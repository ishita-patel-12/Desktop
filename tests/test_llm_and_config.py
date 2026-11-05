"""
Tests for config loading and LLM fallback intent classification.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import loader
from intent import llm_fallback
from intent.schema import IntentType


# --- Config Loader Tests ---

def test_load_settings_structure():
    settings = loader.load_settings()
    assert isinstance(settings, dict)
    assert "assistant_name" in settings
    assert "paths" in settings
    assert "voice" in settings
    assert "gestures" in settings


def test_load_apps_structure():
    apps = loader.load_apps()
    assert isinstance(apps, dict)
    assert "chrome" in apps
    assert "notepad" in apps
    assert "aliases" in apps["chrome"]


def test_load_gesture_profiles():
    profiles = loader.load_gesture_profiles()
    assert isinstance(profiles, dict)
    assert "default" in profiles
    assert "swipe_up" in profiles["default"]


def test_canonical_app_key():
    assert loader.canonical_app_key("chrome") == "chrome"
    assert loader.canonical_app_key("google chrome") == "chrome"
    assert loader.canonical_app_key("VS Code") == "vscode"
    assert loader.canonical_app_key("  text editor  ") == "notepad"
    assert loader.canonical_app_key("nonexistent_app_xyz") is None
    assert loader.canonical_app_key(None) is None
    assert loader.canonical_app_key("") is None


def test_known_app_names():
    names = loader.known_app_names()
    assert isinstance(names, list)
    assert "chrome" in names
    assert "notepad" in names


# --- LLM Fallback Tests ---

def test_llm_fallback_disabled(monkeypatch):
    monkeypatch.setattr(loader, "load_settings", lambda: {"llm_fallback": {"enabled": False}})
    intent = llm_fallback.classify("play some jazz music")
    assert intent.type == IntentType.UNKNOWN
    assert intent.confidence == 0.0


def test_llm_fallback_missing_ollama_package(monkeypatch):
    monkeypatch.setattr(loader, "load_settings", lambda: {"llm_fallback": {"enabled": True}})
    monkeypatch.setattr(llm_fallback, "_client", MagicMock(side_effect=ImportError("No ollama")))

    intent = llm_fallback.classify("summarize this article")
    assert intent.type == IntentType.UNKNOWN
    assert intent.confidence == 0.0


def test_llm_fallback_parses_valid_json(monkeypatch):
    monkeypatch.setattr(loader, "load_settings", lambda: {
        "llm_fallback": {"enabled": True, "model": "qwen2.5:1.5b"}
    })

    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {
            "content": '{"type": "open_app", "app_name": "chrome", "confidence": 0.95}'
        }
    }
    monkeypatch.setattr(llm_fallback, "_client", lambda: fake_client)

    intent = llm_fallback.classify("launch the google chrome browser")
    assert intent.type == IntentType.OPEN_APP
    assert intent.app_name == "chrome"
    assert intent.confidence == 0.95


def test_llm_fallback_handles_corrupt_json(monkeypatch):
    monkeypatch.setattr(loader, "load_settings", lambda: {
        "llm_fallback": {"enabled": True, "model": "qwen2.5:1.5b"}
    })

    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "not valid json"}}
    monkeypatch.setattr(llm_fallback, "_client", lambda: fake_client)

    intent = llm_fallback.classify("some gibberish command")
    assert intent.type == IntentType.UNKNOWN
    assert intent.confidence == 0.0

