"""
Lets the user interrupt the assistant by pressing a key (default: Esc), instead of
having to sit through a full spoken reply or wait out a listening timeout.

Why this exists: the voice loop used to feel like "a recording kind of a
thing" - you'd talk, then be stuck waiting for the full silence-timeout to
elapse, then wait again for however long the reply took to finish
speaking, with no way to cut either short. Pressing Esc now immediately:
  - stops the assistant mid-sentence if it is talking (voice/tts.py's interrupt()), and
  - ends a listen_until_silence() call in progress (voice/stt.py's
    request_interrupt()), returning whatever was captured so far instead of
    waiting out the rest of the timeout.

Uses pynput (already a project dependency for gesture-related keyboard/
mouse control) so it works as a GLOBAL hotkey - it doesn't need the
terminal window focused, which matters in `--gesture`/`--all` mode where
the webcam window usually has focus instead.
"""

from __future__ import annotations
import threading
from loguru import logger

_listener = None
_listener_lock = threading.Lock()


def _on_press(key) -> None:
    try:
        from pynput.keyboard import Key
        if key == Key.esc:
            from voice.tts import interrupt as interrupt_speech
            from voice.stt import request_interrupt
            logger.info("Interrupt key pressed - stopping speech/listening.")
            interrupt_speech()
            request_interrupt()
    except Exception as e:
        logger.debug(f"Hotkey handler error (ignored): {e}")


def start_interrupt_listener() -> bool:
    """Start the global Esc-to-interrupt listener, if it isn't already
    running. Safe to call multiple times (idempotent) and safe to call
    even if pynput isn't installed or no display/input backend is available
    (headless CI, some Linux setups without X) - logs a warning and disables
    the feature rather than crashing the assistant.

    Returns True if the listener is active, False if it couldn't start.
    """
    global _listener
    with _listener_lock:
        if _listener is not None:
            return True
        try:
            from pynput import keyboard
            _listener = keyboard.Listener(on_press=_on_press)
            _listener.daemon = True
            _listener.start()
            logger.info("Interrupt hotkey active: press Esc to stop speech/listening.")
            return True
        except Exception as e:
            logger.warning(
                f"Could not start interrupt hotkey listener ({e}); "
                f"Esc-to-interrupt will be unavailable this session."
            )
            _listener = None
            return False
