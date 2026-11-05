"""
Linux implementation of PlatformAdapter, using `xdotool` (a system binary,
not a pip package - `sudo apt install xdotool` / equivalent).

Why xdotool and not pygetwindow: pygetwindow's Linux backend is partial -
it can enumerate windows on some setups but has no working active-window
detection or close() on most window managers, which is exactly the two
operations this adapter needs. xdotool is the de facto standard CLI for
this on X11 and is what most "Linux automation" tooling shells out to
anyway.

Why this only works under X11: xdotool queries X11's window manager
protocol directly. Wayland deliberately does not expose a
cross-application "what window has focus" API for security/sandboxing
reasons - there is no drop-in xdotool equivalent that works the same way
across Wayland compositors. `is_available()` checks the session type and
reports False under Wayland so callers fall back to NoOpAdapter instead of
silently getting `None`/failures from every call with no explanation of
why. A compositor-specific Wayland adapter (e.g. via each desktop's own
IPC - GNOME Shell's D-Bus interface, Sway's `swaymsg`, etc.) is real
future work, not something one adapter can cover generically.
"""

from __future__ import annotations
import os
import shutil
import subprocess
from typing import Optional
from loguru import logger

from os_adapter.base import PlatformAdapter

_TIMEOUT_SECONDS = 1.0  # xdotool talking to a hung/absent X server should never block the assistant


class LinuxAdapter(PlatformAdapter):
    name = "linux"

    def is_available(self) -> bool:
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            return False
        return shutil.which("xdotool") is not None

    def get_active_window_title(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                return None
            title = result.stdout.strip()
            return title or None
        except FileNotFoundError:
            logger.warning("xdotool not installed - active window detection disabled on Linux. "
                            "`sudo apt install xdotool` (or your distro's equivalent).")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("xdotool timed out getting the active window.")
            return None
        except Exception as e:
            logger.warning(f"Could not get active window via xdotool: {e}")
            return None

    def close_active_window(self, expected_title: Optional[str] = None) -> tuple[bool, str]:
        if expected_title is not None:
            current = self.get_active_window_title()
            if current != expected_title:
                return False, "The focused window changed, so I didn't close anything."
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "windowclose"],
                capture_output=True, text=True, timeout=_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return True, "Closed the current window."
            return False, "Sorry, I couldn't close that window."
        except FileNotFoundError:
            return False, "xdotool isn't installed, so I can't close windows directly."
        except subprocess.TimeoutExpired:
            return False, "Closing the window timed out."
        except Exception as e:
            logger.warning(f"Could not close active window via xdotool: {e}")
            return False, "Sorry, I couldn't close that window."
