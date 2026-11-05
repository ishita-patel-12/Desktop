"""
Regression/behavior tests for config-driven, app-aware gesture actions
(gestures/loop.py + config/gesture_profiles.yaml).

This is the mechanism that answers "can gestures mean different things in
different apps" / "can I play a browser game with swipe gestures":
swipe_up/down/left/right normally scroll or switch tabs (the "default"
profile), but map to arrow-key presses instead when the focused window's
title matches a game profile (e.g. "temple run", "poki.com").

Run with: pytest tests/test_gesture_profiles.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import gestures.loop as loop

PROFILES = {
    "default": {
        "swipe_up": {"action": "scroll", "direction": "up"},
        "swipe_down": {"action": "scroll", "direction": "down"},
        "swipe_left": {"action": "keys", "keys": ["ctrl", "shift", "tab"]},
        "swipe_right": {"action": "keys", "keys": ["ctrl", "tab"]},
    },
    "browser_game": {
        "match": {"title_contains": ["temple run", "poki.com"]},
        "swipe_up": {"action": "keys", "keys": ["up"]},
        "swipe_down": {"action": "keys", "keys": ["down"]},
        "swipe_left": {"action": "keys", "keys": ["left"]},
        "swipe_right": {"action": "keys", "keys": ["right"]},
    },
}


def _fake_pyautogui():
    """A stand-in module for pyautogui's public API surface used here.

    Real pyautogui refuses to import at all without a display/X server
    (it pulls in mouseinfo, which reads $DISPLAY at import time) - which
    is a sandbox/headless-CI limitation, not something about the app's
    logic. Injecting a fake module into sys.modules lets the *dispatch
    logic* be tested (did we call scroll vs. press vs. hotkey, with what
    args) independent of whether a real display is available."""
    fake = MagicMock()
    return fake


def test_select_profile_falls_back_to_default_for_unrelated_window():
    profile = loop._select_profile("Untitled - Notepad", PROFILES)
    assert profile is PROFILES["default"]


def test_select_profile_matches_game_by_title_substring():
    profile = loop._select_profile("Temple Run 2 - Poki.com - Google Chrome", PROFILES)
    assert profile is PROFILES["browser_game"]


def test_select_profile_match_is_case_insensitive():
    profile = loop._select_profile("TEMPLE RUN 2 - Google Chrome", PROFILES)
    assert profile is PROFILES["browser_game"]


def test_select_profile_with_no_title_uses_default():
    profile = loop._select_profile(None, PROFILES)
    assert profile is PROFILES["default"]


def test_swipe_up_scrolls_by_default():
    fake_pyautogui = _fake_pyautogui()
    with patch.object(loop, "get_active_window_title", return_value="Untitled - Notepad"), \
         patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        loop._dispatch_gesture_event("swipe_up", PROFILES)
    fake_pyautogui.scroll.assert_called_once()
    fake_pyautogui.press.assert_not_called()


def test_swipe_up_presses_arrow_key_in_game_profile():
    fake_pyautogui = _fake_pyautogui()
    with patch.object(loop, "get_active_window_title",
                       return_value="Temple Run 2 - Poki.com - Google Chrome"), \
         patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        loop._dispatch_gesture_event("swipe_up", PROFILES)
    fake_pyautogui.press.assert_called_once_with("up")
    fake_pyautogui.scroll.assert_not_called()


def test_swipe_left_is_a_hotkey_by_default():
    fake_pyautogui = _fake_pyautogui()
    with patch.object(loop, "get_active_window_title", return_value="Untitled - Notepad"), \
         patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        loop._dispatch_gesture_event("swipe_left", PROFILES)
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "shift", "tab")


def test_swipe_left_is_single_arrow_press_in_game_profile():
    fake_pyautogui = _fake_pyautogui()
    with patch.object(loop, "get_active_window_title", return_value="Poki.com - Chrome"), \
         patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
        loop._dispatch_gesture_event("swipe_left", PROFILES)
    fake_pyautogui.press.assert_called_once_with("left")
