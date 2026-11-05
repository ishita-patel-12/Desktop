"""
Fallback adapter used whenever no real platform backend is available:
macOS (not implemented yet - see the module docstring in os_adapter/__init__.py
for what a DarwinAdapter would need), a Linux session under Wayland, a
Linux box without xdotool installed, headless CI, or any OS this project
doesn't recognize.

Every operation reports "unknown"/failure cleanly rather than raising or
guessing - callers (context/active_window.py, gestures/loop.py) already
treat `None`/failure as "this information isn't available right now" and
degrade gracefully (e.g. the fist-hold-to-close flow falls back to
asking "which app?" instead of assuming a window is focused).
"""

from __future__ import annotations
from typing import Optional

from os_adapter.base import PlatformAdapter


class NoOpAdapter(PlatformAdapter):
    name = "unsupported"

    def get_active_window_title(self) -> Optional[str]:
        return None

    def close_active_window(self, expected_title: Optional[str] = None) -> tuple[bool, str]:
        return False, ("Active-window control isn't available on this platform/session yet "
                        "(see os_adapter/README or the project README's platform-support section).")

    def is_available(self) -> bool:
        return False
