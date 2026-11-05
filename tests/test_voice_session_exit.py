"""
Regression tests for assistant.py's voice-mode session-exit detection.

BUG FIXED: this used to be a substring check (`"exit" in text.lower()`),
so saying "exit chrome" or "close it, exit please" ended the entire voice
session instead of being handled as a command - the assistant would quit
itself rather than closing the app you asked about. It now only treats the
session as over when the WHOLE utterance is a stand-alone exit phrase.

Run with: pytest tests/test_voice_session_exit.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assistant


def test_bare_exit_words_end_the_session():
    for phrase in ("quit", "exit", "stop", "goodbye", "bye"):
        assert assistant._is_session_exit(phrase), phrase


def test_exit_word_embedded_in_a_command_does_not_end_the_session():
    """The exact bug report this fixes: 'exit chrome' / 'close chrome'
    must be left for the intent parser (-> CLOSE_APP), not swallowed as
    a request to end the voice session."""
    for phrase in ("exit chrome", "close chrome", "quit vs code",
                   "please exit", "exit please", "stop the music"):
        assert not assistant._is_session_exit(phrase), phrase


def test_addressed_exit_phrases_still_end_the_session():
    for phrase in ("hey assistant, stop", "assistant exit", "Assistant, goodbye"):
        assert assistant._is_session_exit(phrase), phrase


def test_case_and_punctuation_insensitive():
    assert assistant._is_session_exit("Quit!")
    assert assistant._is_session_exit("GOODBYE.")
