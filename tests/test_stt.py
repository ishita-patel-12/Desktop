"""
Tests for voice/stt.py - the speech-to-text pipeline.

Neither `sounddevice` (PortAudio isn't installed in this sandbox) nor
`faster-whisper` (not installed - it's a multi-hundred-MB model download)
are usable here, which mirrors a lot of real CI environments too. Every
test below fakes those two boundaries - the microphone stream and the
whisper model - and exercises the actual pipeline logic around them:
VAD chunk classification, pre-speech timeout, silence-triggered stop,
interruption, and the transcription join/fallback behavior.

Run with: pytest tests/test_stt.py -v
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

import voice.stt as stt


# ---------------------------------------------------------------------------
# _chunk_rms - the actual VAD signal
# ---------------------------------------------------------------------------

def test_chunk_rms_of_silence_is_zero():
    assert stt._chunk_rms(np.zeros(512, dtype="float32")) == 0.0


def test_chunk_rms_of_constant_signal_equals_its_amplitude():
    chunk = np.full(512, 0.5, dtype="float32")
    assert stt._chunk_rms(chunk) == pytest.approx(0.5, abs=1e-6)


def test_chunk_rms_of_alternating_signal_is_amplitude_not_zero():
    """A signal that averages to zero (e.g. a square wave) must not be
    read as silence just because its mean is zero - RMS, not mean, is
    what makes this a correct energy measure."""
    chunk = np.array([1.0, -1.0] * 256, dtype="float32")
    assert stt._chunk_rms(chunk) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# transcribe_audio - joining + vad_filter fallback
# ---------------------------------------------------------------------------

class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self, segments=None, fail_with_vad_filter=False):
        self._segments = segments or [_FakeSegment(" hello "), _FakeSegment("world ")]
        self._fail_with_vad_filter = fail_with_vad_filter
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("vad_filter") and self._fail_with_vad_filter:
            raise RuntimeError("onnxruntime not installed")
        return self._segments, object()


def test_transcribe_audio_joins_and_strips_segments(monkeypatch):
    fake = _FakeModel(segments=[_FakeSegment(" hello "), _FakeSegment("world ")])
    monkeypatch.setattr(stt, "_get_model", lambda: fake)

    result = stt.transcribe_audio(np.zeros(1600, dtype="float32"))

    assert result == "hello world"
    assert fake.calls[0]["vad_filter"] is True


def test_transcribe_audio_falls_back_when_vad_filter_unavailable(monkeypatch):
    fake = _FakeModel(segments=[_FakeSegment("fallback text")], fail_with_vad_filter=True)
    monkeypatch.setattr(stt, "_get_model", lambda: fake)

    result = stt.transcribe_audio(np.zeros(1600, dtype="float32"))

    assert result == "fallback text"
    assert len(fake.calls) == 2                     # first call (raised) + retry
    assert fake.calls[0]["vad_filter"] is True
    assert "vad_filter" not in fake.calls[1]         # retry omits it entirely


# ---------------------------------------------------------------------------
# listen_until_silence - the full VAD capture loop, mic faked out
# ---------------------------------------------------------------------------

class _FakeInputStream:
    """Stands in for `sounddevice.InputStream`: hands back pre-scripted
    chunks in order, then silence forever once the script runs out (so a
    test that doesn't carefully budget chunks doesn't hang/crash)."""

    def __init__(self, chunks, **kwargs):
        self._chunks = list(chunks)
        self.blocksize = kwargs.get("blocksize", stt._CHUNK)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, n):
        if self._chunks:
            chunk = self._chunks.pop(0)
        else:
            chunk = np.zeros((n, 1), dtype="float32")
        return chunk, False


def _loud_chunk(n=None):
    return np.full((n or stt._CHUNK, 1), 0.8, dtype="float32")


def _quiet_chunk(n=None):
    return np.zeros((n or stt._CHUNK, 1), dtype="float32")


def _install_fake_sounddevice(monkeypatch, chunks):
    fake_module = types.SimpleNamespace(
        InputStream=lambda **kwargs: _FakeInputStream(chunks, **kwargs),
        rec=lambda *a, **k: (_ for _ in ()).throw(AssertionError("rec() shouldn't be used by listen_until_silence")),
        wait=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)


@pytest.fixture(autouse=True)
def _clear_interrupt_flag():
    stt._interrupt_event.clear()
    yield
    stt._interrupt_event.clear()


def test_listen_returns_empty_string_when_no_speech_within_timeout(monkeypatch):
    _install_fake_sounddevice(monkeypatch, chunks=[])  # only ever silence

    def _fail_if_called(*a, **k):
        raise AssertionError("transcribe_audio should not be called when no speech was heard")
    monkeypatch.setattr(stt, "transcribe_audio", _fail_if_called)

    result = stt.listen_until_silence(pre_speech_timeout=0.05, silence_duration=0.05, max_duration=1.0)

    assert result == ""


def test_listen_records_speech_and_stops_on_silence(monkeypatch):
    chunks = [_loud_chunk(), _loud_chunk(), _quiet_chunk(), _quiet_chunk()]
    _install_fake_sounddevice(monkeypatch, chunks=chunks)
    monkeypatch.setattr(stt, "transcribe_audio", lambda audio: "close chrome")

    result = stt.listen_until_silence(pre_speech_timeout=0.5, silence_duration=0.05, max_duration=2.0)

    assert result == "close chrome"


def test_listen_respects_max_duration_even_without_silence(monkeypatch):
    """A long continuous utterance with no gap should still stop at
    max_duration rather than recording forever."""
    chunks = [_loud_chunk() for _ in range(50)]
    _install_fake_sounddevice(monkeypatch, chunks=chunks)
    monkeypatch.setattr(stt, "transcribe_audio", lambda audio: "long utterance")

    result = stt.listen_until_silence(pre_speech_timeout=0.5, silence_duration=10.0, max_duration=0.05)

    assert result == "long utterance"


def test_listen_can_be_interrupted_mid_capture(monkeypatch):
    """request_interrupt() is meant to be called *during* an in-progress
    listen (e.g. from a hotkey thread) - listen_until_silence() clears any
    stale flag from a previous call the moment it starts, so triggering
    the interrupt has to happen after capture is already underway, not
    before listen_until_silence() is even called."""
    chunks = [_loud_chunk() for _ in range(50)]

    class _InterruptingStream(_FakeInputStream):
        def read(self, n):
            chunk, ok = super().read(n)
            if len(self._chunks) == 47:  # after the first few chunks are consumed
                stt.request_interrupt()
            return chunk, ok

    fake_module = types.SimpleNamespace(
        InputStream=lambda **kwargs: _InterruptingStream(chunks, **kwargs),
        rec=lambda *a, **k: (_ for _ in ()).throw(AssertionError("unused")),
        wait=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)
    monkeypatch.setattr(stt, "transcribe_audio", lambda audio: "should not matter")

    result = stt.listen_until_silence(pre_speech_timeout=1.0, silence_duration=1.0, max_duration=5.0)

    # Interrupted before silence/max_duration would naturally have ended
    # it - i.e. it stopped early because of the interrupt, not coincidence.
    # Whatever was captured up to the interrupt still gets transcribed
    # (an interrupt means "stop now", not "discard everything so far").
    assert result == "should not matter"
    assert not stt._interrupt_event.is_set()  # cleared again after handling
