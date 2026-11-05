"""
Tests for os_adapter/ - the platform abstraction layer.

These deliberately don't call real OS window APIs (that would make the
suite fail depending on which OS/session runs it - exactly the problem
this layer exists to contain). Instead:
  - the factory's OS-selection logic is tested by monkeypatching
    platform.system() and each adapter's is_available(), so the actual
    xdotool/pygetwindow calls never run;
  - a minimal FakeAdapter subclassing PlatformAdapter checks that callers
    (active_window.py, gestures/loop.py's close path) correctly delegate
    to whatever get_adapter() returns, without caring what OS is real.

Run with: pytest tests/test_os_adapter.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import os_adapter
from os_adapter.base import PlatformAdapter
from os_adapter.noop_adapter import NoOpAdapter
from os_adapter.windows_adapter import WindowsAdapter
from os_adapter.linux_adapter import LinuxAdapter


@pytest.fixture(autouse=True)
def _reset_adapter_cache():
    """Every test starts and ends with a clean slate - get_adapter() is
    cached at module level, so leaking a fake/mocked adapter between tests
    would make results depend on test order."""
    os_adapter.reset_cache()
    yield
    os_adapter.reset_cache()


class FakeAdapter(PlatformAdapter):
    """Minimal adapter for testing callers in isolation from any real OS."""
    name = "fake"

    def __init__(self, title="Some App - Notepad", close_result=(True, "Closed.")):
        self._title = title
        self._close_result = close_result
        self.close_calls = []

    def get_active_window_title(self):
        return self._title

    def close_active_window(self, expected_title=None):
        self.close_calls.append(expected_title)
        return self._close_result


# ---------------------------------------------------------------------------
# NoOpAdapter - the universal safe fallback
# ---------------------------------------------------------------------------

def test_noop_adapter_never_raises_and_reports_unavailable():
    adapter = NoOpAdapter()
    assert adapter.get_active_window_title() is None
    success, message = adapter.close_active_window(expected_title="anything")
    assert success is False
    assert isinstance(message, str) and message  # a real, non-empty explanation
    assert adapter.is_available() is False


# ---------------------------------------------------------------------------
# get_adapter() factory: OS selection + availability fallback
# ---------------------------------------------------------------------------

def test_factory_selects_windows_adapter(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(WindowsAdapter, "is_available", lambda self: True)
    adapter = os_adapter.get_adapter()
    assert isinstance(adapter, WindowsAdapter)
    assert adapter.name == "windows"


def test_factory_selects_linux_adapter_when_available(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(LinuxAdapter, "is_available", lambda self: True)
    adapter = os_adapter.get_adapter()
    assert isinstance(adapter, LinuxAdapter)
    assert adapter.name == "linux"


def test_factory_falls_back_to_noop_when_linux_backend_unavailable(monkeypatch):
    """E.g. Wayland session or xdotool not installed - LinuxAdapter itself
    reports is_available() == False in that case; the factory must not
    hand back an adapter that will silently fail on every call."""
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(LinuxAdapter, "is_available", lambda self: False)
    adapter = os_adapter.get_adapter()
    assert isinstance(adapter, NoOpAdapter)


def test_factory_falls_back_to_noop_on_unrecognized_platform(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "SomeOtherOS")
    adapter = os_adapter.get_adapter()
    assert isinstance(adapter, NoOpAdapter)


def test_factory_result_is_cached(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(LinuxAdapter, "is_available", lambda self: True)
    first = os_adapter.get_adapter()
    second = os_adapter.get_adapter()
    assert first is second


def test_reset_cache_forces_re_detection(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(LinuxAdapter, "is_available", lambda self: True)
    first = os_adapter.get_adapter()

    os_adapter.reset_cache()
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(WindowsAdapter, "is_available", lambda self: True)
    second = os_adapter.get_adapter()

    assert type(first) is not type(second)


def test_set_adapter_for_testing_overrides_the_cache():
    fake = FakeAdapter(title="Fake Window")
    os_adapter._set_adapter_for_testing(fake)
    assert os_adapter.get_adapter() is fake


# ---------------------------------------------------------------------------
# Callers correctly delegate to whatever adapter get_adapter() returns
# ---------------------------------------------------------------------------

def test_active_window_module_delegates_to_the_adapter():
    from context.active_window import get_active_window_title
    fake = FakeAdapter(title="Inbox - Gmail - Google Chrome")
    os_adapter._set_adapter_for_testing(fake)
    assert get_active_window_title() == "Inbox - Gmail - Google Chrome"


def test_guess_app_name_from_title_is_platform_agnostic():
    """This half of active_window.py is pure string logic - no OS call at
    all - so it should behave identically regardless of adapter."""
    from context.active_window import guess_app_name_from_title
    assert guess_app_name_from_title("Inbox - Gmail - Google Chrome") == "Google Chrome"
    assert guess_app_name_from_title("document.txt | Notepad") == "Notepad"
    assert guess_app_name_from_title(None) is None


def test_gesture_loop_close_path_delegates_to_the_adapter():
    from gestures.loop import _close_active_window_directly
    fake = FakeAdapter(close_result=(True, "Closed the current window."))
    os_adapter._set_adapter_for_testing(fake)

    result = _close_active_window_directly(expected_title="Notepad")

    assert result.success is True
    assert fake.close_calls == ["Notepad"]  # expected_title was threaded through


def test_gesture_loop_close_path_surfaces_adapter_failure():
    from gestures.loop import _close_active_window_directly
    fake = FakeAdapter(close_result=(False, "The focused window changed, so I didn't close anything."))
    os_adapter._set_adapter_for_testing(fake)

    result = _close_active_window_directly(expected_title="Notepad")

    assert result.success is False
    assert "didn't close" in result.message
