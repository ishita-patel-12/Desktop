"""
Public entry point for the OS abstraction layer: `get_adapter()`.

    from os_adapter import get_adapter
    title = get_adapter().get_active_window_title()

Picks an adapter based on `platform.system()`, then confirms it's actually
usable via `is_available()` (the right OS doesn't guarantee the right
backend is present - e.g. Linux without xdotool installed, or a Wayland
session) and falls back to NoOpAdapter rather than handing back something
that will silently fail on every call.

Platform support status:
  - Windows: full support (WindowsAdapter, via pygetwindow)
  - Linux:   full support under X11 with xdotool installed (LinuxAdapter);
             NoOpAdapter under Wayland or without xdotool - see
             linux_adapter.py's docstring for why Wayland specifically
             can't be supported the same generic way
  - macOS:   not implemented yet (NoOpAdapter). Would be a DarwinAdapter
             using `osascript`/AppleScript (`System Events` for the
             focused process name/window title, `tell application ...
             to quit` to close) - the interface in base.py is exactly what
             that adapter would need to implement; the reason it isn't
             built is not knowing macOS's edge cases well enough to trust
             without a real Mac to test against, not architectural
             difficulty.

The result is cached (one process = one platform, this never changes at
runtime) - `reset_cache()` and `_set_adapter_for_testing()` exist so tests
can override it without monkeypatching `platform.system()`.
"""

from __future__ import annotations
import platform
from typing import Optional

from os_adapter.base import PlatformAdapter
from os_adapter.noop_adapter import NoOpAdapter

_cached_adapter: Optional[PlatformAdapter] = None


def _build_adapter() -> PlatformAdapter:
    system = platform.system().lower()

    if system == "windows":
        from os_adapter.windows_adapter import WindowsAdapter
        candidate = WindowsAdapter()
    elif system == "linux":
        from os_adapter.linux_adapter import LinuxAdapter
        candidate = LinuxAdapter()
    else:
        # Covers "darwin" (not implemented yet, see module docstring) and
        # anything else `platform.system()` might return.
        candidate = NoOpAdapter()

    if not candidate.is_available():
        return NoOpAdapter()
    return candidate


def get_adapter() -> PlatformAdapter:
    """Returns the right adapter for the current OS/session, falling back
    to NoOpAdapter if the real one isn't usable. Cached after the first
    call within a process."""
    global _cached_adapter
    if _cached_adapter is None:
        _cached_adapter = _build_adapter()
    return _cached_adapter


def reset_cache() -> None:
    """Forces the next get_adapter() call to re-detect the platform.
    Only needed by tests that mock platform.system()."""
    global _cached_adapter
    _cached_adapter = None


def _set_adapter_for_testing(adapter: Optional[PlatformAdapter]) -> None:
    """Injects a specific adapter (or clears the override with None,
    which causes the next get_adapter() to re-detect normally). Lets
    tests exercise callers (active_window.py, gestures/loop.py) against a
    fake adapter without needing to be run on every OS."""
    global _cached_adapter
    _cached_adapter = adapter
