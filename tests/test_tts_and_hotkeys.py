"""
Tests for voice/tts.py and voice/hotkeys.py.
Validates spoken feedback, console printing fallback, interrupt mechanisms,
and global hotkey handling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice import tts
from voice import hotkeys


def test_say_empty_string_does_nothing(capsys):
    tts.say("")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_say_prints_to_stdout_and_uses_engine(capsys):
    mock_engine = MagicMock()
    with patch("pyttsx3.init", return_value=mock_engine):
        tts.say("Hello from Assistant")

    captured = capsys.readouterr()
    assert "Assistant: Hello from Assistant" in captured.out
    mock_engine.say.assert_called_once_with("Hello from Assistant")
    mock_engine.runAndWait.assert_called_once()


def test_say_prints_even_if_pyttsx3_fails(capsys):
    mock_engine = MagicMock()
    mock_engine.runAndWait.side_effect = RuntimeError("Audio device unavailable")

    with patch("pyttsx3.init", return_value=mock_engine):
        tts.say("Fallback message")

    captured = capsys.readouterr()
    assert "Assistant: Fallback message" in captured.out


def test_tts_interrupt_stops_active_engine():
    mock_engine = MagicMock()
    with tts._engine_ref_lock:
        tts._current_engine = mock_engine

    try:
        tts.interrupt()
        mock_engine.stop.assert_called_once()
    finally:
        with tts._engine_ref_lock:
            tts._current_engine = None


def test_hotkey_listener_starts():
    with patch("pynput.keyboard.Listener") as mock_listener_cls:
        instance = MagicMock()
        mock_listener_cls.return_value = instance
        # Reset listener singleton
        hotkeys._listener = None

        started = hotkeys.start_interrupt_listener()
        assert started is True
        instance.start.assert_called_once()


def test_hotkey_on_press_esc_triggers_interrupts():
    from pynput.keyboard import Key

    with patch("voice.tts.interrupt") as mock_tts_interrupt, \
         patch("voice.stt.request_interrupt") as mock_stt_interrupt:
        hotkeys._on_press(Key.esc)

        mock_tts_interrupt.assert_called_once()
        mock_stt_interrupt.assert_called_once()


def test_hotkey_on_press_other_keys_ignored():
    with patch("voice.tts.interrupt") as mock_tts_interrupt, \
         patch("voice.stt.request_interrupt") as mock_stt_interrupt:
        hotkeys._on_press("a")

        mock_tts_interrupt.assert_not_called()
        mock_stt_interrupt.assert_not_called()

