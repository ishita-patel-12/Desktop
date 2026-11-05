"""
Windows implementation of PlatformAdapter, using pygetwindow - already a
project dependency, and Windows is the one OS pygetwindow supports fully
(active-window detection AND close()).
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

from os_adapter.base import PlatformAdapter


class WindowsAdapter(PlatformAdapter):
    name = "windows"

    def get_active_window_title(self) -> Optional[str]:
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            return win.title if win else None
        except ImportError:
            logger.warning("pygetwindow not installed - active window detection disabled.")
            return None
        except Exception as e:
            logger.warning(f"Could not get active window: {e}")
            return None

    def close_active_window(self, expected_title: Optional[str] = None) -> tuple[bool, str]:
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if not win:
                return False, "No active window to close."
            if expected_title is not None and win.title != expected_title:
                # Focus moved on since the caller decided to close something
                # (e.g. while waiting on a spoken confirmation) - refuse
                # rather than closing whatever now happens to have focus.
                return False, "The focused window changed, so I didn't close anything."
            win.close()
            return True, "Closed the current window."
        except ImportError:
            logger.warning("pygetwindow not installed - cannot close active window.")
            return False, "pygetwindow isn't installed, so I can't close windows directly."
        except Exception as e:
            logger.warning(f"Could not close active window: {e}")
            return False, "Sorry, I couldn't close that window."

    def is_available(self) -> bool:
        try:
            import pygetwindow  # noqa: F401
            return True
        except ImportError:
            return False
