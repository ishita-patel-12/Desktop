"""
The abstract interface every OS-specific adapter implements.

Why this exists: the OS-touching parts of this project used to be a handful
of direct `import pygetwindow` calls scattered across context/active_window.py
and gestures/loop.py, with no Linux path at all - pygetwindow's Linux support
is partial (no active-window detection or close() on most window managers),
so the assistant simply didn't work for window-focus features off Windows.
Centralizing "what does the OS let us do to windows" behind one small
interface means:
  - adding a new platform is a new file implementing this interface, not a
    hunt through the codebase for every place that assumed Windows,
  - tests can inject a fake adapter (see NoOpAdapter / tests/test_os_adapter.py)
    instead of mocking a platform-specific library,
  - callers (active_window.py, apps.py, loop.py) don't need their own
    platform.system() branching - they just ask os_adapter.get_adapter().

Deliberately small: only the operations the assistant actually needs
(read the focused window's title, close it) are on the interface. Resist
the urge to grow this into a general "window management" API before a
second real caller needs more than that - YAGNI applies here same as
anywhere else.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class PlatformAdapter(ABC):
    """One implementation per OS. `name` must be one of "windows", "linux",
    "darwin" - it's used as the lookup key into config/apps.yaml's
    per-platform launch commands (see executor/apps.py), so it has to match
    those YAML keys exactly."""

    name: str = "unknown"

    @abstractmethod
    def get_active_window_title(self) -> Optional[str]:
        """The title of whichever window currently has OS focus, or None
        if that can't be determined (no window focused, backend
        unavailable, permission denied, etc). Must never raise - callers
        rely on None meaning "unknown" rather than having to wrap every
        call in their own try/except."""
        raise NotImplementedError

    @abstractmethod
    def close_active_window(self, expected_title: Optional[str] = None) -> tuple[bool, str]:
        """Ask the OS to close whichever window currently has focus.

        `expected_title`, when given, is a safety check: if the window
        that currently has focus doesn't match, this must refuse rather
        than close the wrong thing - focus can change in the gap between
        deciding to close something and this actually running (e.g. while
        waiting on a spoken confirmation).

        Returns (success, human_readable_message) - never raises.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this adapter's backend is actually usable right now -
        distinct from "this is the right OS". A Linux machine still fails
        this under Wayland (see linux_adapter.py) since X11-only APIs are
        used; a headless CI runner fails it on any OS. Default True since
        most adapters only find out at call-time; override where a cheap
        upfront check is possible."""
        return True
