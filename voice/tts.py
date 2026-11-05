"""Offline text-to-speech feedback (`pip install pyttsx3`).

Kept offline deliberately - part of the local-first design (see README):
no audio ever leaves the machine, and it works with zero network latency.

--- Why you might not hear anything (fixed here) ---------------------------
pyttsx3 wraps a COM object under the hood on Windows (SAPI5). Two very
common, well-documented failure modes fall straight out of that:

  1. "Only speaks once" - a single cached/reused `pyttsx3.init()` engine
     frequently stops producing any audio after its first
     `say()` + `runAndWait()` cycle, even though no exception is raised.
     This module used to keep one module-level engine alive forever, which
     is exactly the pattern that triggers it. The fix is to build a fresh
     engine for every utterance instead of reusing one.
  2. Thread affinity - the underlying COM/SAPI5 object is tied to the OS
     thread that created it. `assistant.py --all` runs the voice loop on a
     background `threading.Thread` while gestures own the main thread, so a
     single shared engine created on one thread would be invalid to use
     from the other, again failing silently. Building the engine fresh
     inside whichever thread is actually speaking sidesteps this too.

A module-level lock still serializes overlapping calls (the gesture loop
and the voice loop can both call `say()` at the same time in `--all`
mode) since two threads writing to the same audio output device at once
isn't supported and would otherwise garble or drop speech.

`interrupt()` lets another thread stop a reply mid-sentence (wired up to
the Esc hotkey in voice/hotkeys.py) - this is what makes it possible to
cut the assistant off instead of having to sit through a full response
before it starts listening again.
"""

from __future__ import annotations
import threading
from loguru import logger
from config.loader import load_settings

_speak_lock = threading.Lock()

# Tracks the engine currently mid-utterance so interrupt() can reach it from
# another thread (e.g. the Esc hotkey listener in voice/hotkeys.py) while
# say() is blocked inside runAndWait() on the speaking thread. Deliberately
# a SEPARATE lock from _speak_lock: interrupt() must never wait on the same
# lock say() holds for the full duration of speech, or it could never fire
# until the utterance was already finished, which defeats the point.
_engine_ref_lock = threading.Lock()
_current_engine = None


def interrupt() -> None:
    """Stop whatever the assistant is currently saying, immediately."""
    with _engine_ref_lock:
        engine = _current_engine
    if engine is not None:
        try:
            engine.stop()
        except Exception as e:
            logger.debug(f"TTS interrupt: stop() raised ({e}); ignoring.")


def say(text: str) -> None:
    """Speak `text` aloud and print to console. Falls back to console-only
    if pyttsx3 isn't installed or audio playback is unavailable - useful for
    headless dev/testing without an audio backend, and ensures a TTS hiccup
    never silently swallows the assistant's response."""
    global _current_engine
    if not text:
        return

    logger.debug(f"[TTS] {text}")
    print(f"Assistant: {text}")

    try:
        import pyttsx3
    except ImportError:
        logger.debug("pyttsx3 not installed; speech synthesis skipped.")
        return

    # Serialize speech across threads (voice loop + gesture loop can both
    # want to talk at once in `--all` mode); see module docstring.
    with _speak_lock:
        engine = None
        try:
            # Deliberately NOT cached at module scope - see docstring.
            engine = pyttsx3.init()
            with _engine_ref_lock:
                _current_engine = engine
            rate = load_settings().get("voice", {}).get("tts_rate", 175)
            engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.debug(f"TTS audio playback failed ({e}); printed output preserved.")
        finally:
            with _engine_ref_lock:
                if _current_engine is engine:
                    _current_engine = None
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
