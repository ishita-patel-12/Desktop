"""
Regression tests for gestures/loop.py's fist-hold confirmation flow.

Covers the bugs fixed here:
  - it used to block on terminal input() (freezing the camera feed and
    making it unusable without a terminal); it now asks + listens through
    the same speak/listen callables the voice assistant uses, and does the
    asking on a background thread so the caller isn't blocked.
  - a fist held on an app not covered by close_app() (unknown process name)
    used to just report failure; it now falls back to closing whatever
    window currently has focus.

Run with: pytest tests/test_gesture_fist_close.py -v
"""

import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import gestures.loop as loop
from executor.apps import ExecResult


class _ImmediateThread:
    """Stand-in for threading.Thread that runs the target synchronously,
    so tests don't need to sleep/poll for the background confirmation to
    finish."""

    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def setup_function(_):
    # Each test starts clean regardless of execution order.
    loop._confirmation_active = False


def test_fist_hold_closes_after_yes():
    with patch.object(loop, "get_active_window_title", return_value="Untitled - Notepad"), \
         patch.object(loop, "guess_app_name_from_title", return_value="Notepad"), \
         patch.object(loop.threading, "Thread", _ImmediateThread), \
         patch("executor.apps.close_app", return_value=ExecResult(True, "Closed Notepad")) as mock_close:

        spoken = []
        _handle = lambda: loop._handle_fist_hold(
            speak=lambda t: spoken.append(t),
            listen=lambda: "yes",
        )
        _handle()

    mock_close.assert_called_once_with("Notepad", confirmed=True)
    assert any("close notepad" in s.lower() for s in spoken)
    assert any(s == "Closed Notepad" for s in spoken)
    assert loop._confirmation_active is False


def test_fist_hold_leaves_app_open_on_no():
    with patch.object(loop, "get_active_window_title", return_value="Untitled - Notepad"), \
         patch.object(loop, "guess_app_name_from_title", return_value="Notepad"), \
         patch.object(loop.threading, "Thread", _ImmediateThread), \
         patch("executor.apps.close_app") as mock_close:

        spoken = []
        loop._handle_fist_hold(speak=lambda t: spoken.append(t), listen=lambda: "no")

    mock_close.assert_not_called()
    assert spoken[-1] == "Okay, leaving it open."


def test_fist_hold_falls_back_to_direct_window_close_when_process_unknown():
    """App isn't in config/apps.yaml / no matching process - should still
    close the focused window directly instead of just failing."""
    with patch.object(loop, "get_active_window_title", return_value="Some Random Tool"), \
         patch.object(loop, "guess_app_name_from_title", return_value="Some Random Tool"), \
         patch.object(loop.threading, "Thread", _ImmediateThread), \
         patch("executor.apps.close_app", return_value=ExecResult(False, "not found")), \
         patch.object(loop, "_close_active_window_directly",
                       return_value=ExecResult(True, "Closed the current window.")) as mock_fallback:

        spoken = []
        loop._handle_fist_hold(speak=lambda t: spoken.append(t), listen=lambda: "yes")

    mock_fallback.assert_called_once()
    assert spoken[-1] == "Closed the current window."


def test_fist_hold_ignores_second_trigger_while_confirmation_pending():
    """A fist held slightly too long shouldn't stack a second overlapping
    confirmation prompt on top of the first."""
    loop._confirmation_active = True
    try:
        spoken = []
        with patch.object(loop.threading, "Thread", _ImmediateThread):
            loop._handle_fist_hold(speak=lambda t: spoken.append(t), listen=lambda: "yes")
        assert spoken == []  # nothing asked - the earlier confirmation owns the turn
    finally:
        loop._confirmation_active = False


def test_fist_hold_ignores_its_own_gesture_feed_window():
    with patch.object(loop, "get_active_window_title", return_value="Assistant - Gesture Feed"), \
         patch.object(loop.threading, "Thread", _ImmediateThread):
        spoken = []
        loop._handle_fist_hold(speak=lambda t: spoken.append(t), listen=lambda: "yes")
    assert spoken == []
