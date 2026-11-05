"""
Speech-to-text using faster-whisper (CPU, quantized) - `pip install
faster-whisper sounddevice numpy`.

This module is intentionally decoupled from the audio-capture loop:
`transcribe_audio()` takes raw float32 samples so it can be unit tested
with synthetic arrays without needing a real microphone.

`listen_until_silence()` provides VAD-based continuous recording so
the voice loop does not require press-to-talk or a wake word - `--voice`
starts listening immediately on launch.

--- Latency & interruption notes -------------------------------------------
Two things made the assistant feel slow/unresponsive, both addressed here:

  1. The whisper model used to load lazily on the FIRST utterance of a
     session - a multi-second one-time cost that landed right in the
     middle of the first thing you said, and (worse, in `--all` mode)
     briefly stalled the gesture loop's frame processing too, since model
     deserialization holds the GIL for part of that time. `preload_model()`
     lets the caller pay that cost upfront, before the mic starts listening
     / before the camera loop starts.
  2. There was no way to interrupt a recording in progress - you had to
     wait out the full silence-timeout even if you'd said everything you
     needed already, or wait through `pre_speech_timeout` doing nothing.
     `request_interrupt()` (wired up to a hotkey in voice/hotkeys.py) lets
     any listen-in-progress bail out immediately and return whatever was
     captured so far.
"""

from __future__ import annotations
import threading
import numpy as np
from loguru import logger
from config.loader import load_settings

_model = None
_model_lock = threading.Lock()
SAMPLE_RATE = 16000
_CHUNK = 512           # ~32 ms per chunk at 16 kHz

# Only one thing can hold the physical microphone at a time. This matters
# once voice and gestures run together (`python assistant.py --all`): the
# continuous voice-conversation loop and the gesture module's fist-hold
# confirmation prompt (see gestures/loop.py) both call
# `listen_until_silence()` from different threads. Without this lock they'd
# both try to open an `sd.InputStream` on the same device simultaneously,
# which on most systems either raises a "device busy" error or produces
# silent/garbled audio for one of the two callers. Serializing access means
# a fist-hold that happens mid-conversation simply waits its turn for the
# mic instead of failing outright.
_mic_lock = threading.Lock()

# Set by voice/hotkeys.py's interrupt key (or anything else) to cut a
# `listen_until_silence()` call short. Checked once per audio chunk (~32ms),
# so interruption is essentially instant.
_interrupt_event = threading.Event()


def request_interrupt() -> None:
    """Ask any in-progress listen_until_silence() call to stop immediately."""
    _interrupt_event.set()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:  # re-check inside the lock (another thread may
                             # have already loaded it while we were waiting)
            from faster_whisper import WhisperModel
            cfg = load_settings().get("voice", {})
            _model = WhisperModel(
                cfg.get("stt_model", "base.en"),
                device=cfg.get("stt_device", "cpu"),
                compute_type=cfg.get("stt_compute_type", "int8"),
                # Caps how many CPU threads whisper itself spins up.
                # Left uncapped, CTranslate2 happily grabs every core it
                # can find, which starves the gesture loop's mediapipe
                # processing of CPU during `--all` mode and makes the
                # camera feed stutter every time a command is transcribed.
                # Default of 2 leaves headroom for the gesture loop on
                # typical 4+ core machines; raise it in settings.yaml if
                # you're voice-only and want faster transcription instead.
                cpu_threads=cfg.get("stt_cpu_threads", 2),
            )
            logger.info("faster-whisper model loaded")
    return _model


def preload_model() -> None:
    """Force the whisper model to load right now, on whichever thread calls
    this - call it once at startup (before the mic starts listening / before
    the gesture camera loop starts) so the one-time load cost is predictable
    and up-front instead of stalling the first thing you say."""
    _get_model()


def _chunk_rms(chunk: np.ndarray) -> float:
    """RMS energy of one audio chunk - the actual voice-activity signal
    listen_until_silence() thresholds against. Pulled out as its own
    function (rather than an inline expression in the capture loop) so the
    VAD decision itself is unit-testable on synthetic arrays without
    needing a real microphone or an open sounddevice stream."""
    return float(np.sqrt(np.mean(chunk ** 2)))


def transcribe_audio(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """samples: float32 mono array in [-1, 1]."""
    model = _get_model()
    try:
        segments, _info = model.transcribe(
            samples, language="en", beam_size=1,
            vad_filter=True,               # skips silence within the clip -
                                            # both faster and less prone to
                                            # hallucinating text on dead air
            condition_on_previous_text=False,  # avoids occasional slow
                                                # repetition loops on noisy audio
        )
        text = " ".join(seg.text.strip() for seg in segments)
    except Exception as e:
        # vad_filter needs onnxruntime; fall back cleanly if it's missing
        # rather than losing the whole transcription.
        logger.warning(f"Transcribe with vad_filter failed ({e}); retrying without it.")
        segments, _info = model.transcribe(
            samples, language="en", beam_size=1, condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments)
    logger.info(f"STT transcribed: {text!r}")
    return text.strip()


def record_and_transcribe(duration_seconds: float = 5.0) -> str:
    """Record from the default mic for `duration_seconds` and transcribe.
    Used by the live voice loop; not needed for text-mode / tests."""
    import sounddevice as sd

    with _mic_lock:
        logger.info(f"Recording {duration_seconds}s of audio...")
        audio = sd.rec(int(duration_seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32")
        sd.wait()
    return transcribe_audio(audio.flatten())


def listen_until_silence(
    silence_threshold: float = 0.015,
    silence_duration: float = 1.2,
    max_duration: float = 12.0,
    pre_speech_timeout: float = 6.0,
) -> str:
    """VAD-based recording: waits for speech, records until silence, transcribes.

    Args:
        silence_threshold: RMS energy below which a chunk is considered silence.
        silence_duration:  Seconds of continuous silence that ends the recording.
        max_duration:      Hard cap on total recording (seconds).
        pre_speech_timeout: Give up if no speech heard within this many seconds.

    Can be cut short at any point by request_interrupt() (e.g. the Esc
    hotkey - see voice/hotkeys.py); returns whatever was captured so far
    in that case instead of waiting out the full timeout/silence window.

    Returns:
        Transcribed text (empty string if nothing was heard, or if
        interrupted before any speech started).
    """
    import sounddevice as sd

    _interrupt_event.clear()

    chunks_per_second = SAMPLE_RATE // _CHUNK
    silence_chunks_needed = int(silence_duration * chunks_per_second)
    max_chunks = int(max_duration * chunks_per_second)
    timeout_chunks = int(pre_speech_timeout * chunks_per_second)

    recorded: list[np.ndarray] = []
    silence_count = 0
    speech_started = False
    pre_speech_count = 0
    interrupted = False

    # Held for the whole listen - see the `_mic_lock` docstring above for
    # why this matters once voice + gestures run concurrently.
    with _mic_lock:
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=_CHUNK) as stream:
                while True:
                    if _interrupt_event.is_set():
                        logger.debug("Listen interrupted.")
                        interrupted = True
                        break

                    chunk, _ = stream.read(_CHUNK)
                    chunk = chunk.flatten()
                    rms = _chunk_rms(chunk)

                    if not speech_started:
                        pre_speech_count += 1
                        if rms > silence_threshold:
                            speech_started = True
                            logger.debug("Speech started.")
                            recorded.append(chunk)
                            silence_count = 0
                        elif pre_speech_count >= timeout_chunks:
                            logger.debug("No speech detected within timeout.")
                            return ""
                    else:
                        recorded.append(chunk)
                        if rms < silence_threshold:
                            silence_count += 1
                            if silence_count >= silence_chunks_needed:
                                logger.debug("Silence detected, stopping.")
                                break
                        else:
                            silence_count = 0

                        if len(recorded) >= max_chunks:
                            logger.debug("Max duration reached.")
                            break
        except Exception as e:
            logger.warning(
                f"Microphone stream error ({e}). Ensure a microphone is connected "
                f"and audio permissions are granted."
            )
            return ""

    if interrupted:
        _interrupt_event.clear()
    if not recorded:
        return ""

    audio = np.concatenate(recorded)
    return transcribe_audio(audio)
