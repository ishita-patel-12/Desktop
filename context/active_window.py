"""
Detects the currently focused/active window - used by:
  - the fist-to-close confirmation flow ("Should I close [App Name]?")
  - context-aware gesture mapping (gesture_profiles.yaml / teachable gestures)

Delegates the actual OS call to os_adapter.get_adapter() rather than
calling a platform-specific library directly - see os_adapter/base.py for
why. This module now only owns the OS-agnostic part: turning a raw window
title into a guessed app name.
"""

from __future__ import annotations
from typing import Optional

from os_adapter import get_adapter


def get_active_window_title() -> Optional[str]:
    return get_adapter().get_active_window_title()


def guess_app_name_from_title(title: Optional[str]) -> Optional[str]:
    """Very rough heuristic: most window titles end with '- AppName' or
    similar. Good enough for the confirmation prompt; not used for anything
    safety-critical."""
    if not title:
        return None
    for sep in (" - ", " — ", " | "):
        if sep in title:
            return title.split(sep)[-1].strip()
    return title.strip()
