from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import yaml

_CONFIG_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_settings() -> dict:
    with open(_CONFIG_DIR / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_apps() -> dict:
    with open(_CONFIG_DIR / "apps.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["apps"]


@lru_cache(maxsize=None)
def load_gesture_profiles() -> dict:
    """App-aware gesture-to-action mapping (see config/gesture_profiles.yaml
    for the format) - lets the same swipe gesture mean different things in
    different apps, e.g. tab-switching normally but arrow-key presses while
    a browser game has focus."""
    path = _CONFIG_DIR / "gesture_profiles.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def known_app_names() -> list[str]:
    """Flat list of every alias, for the intent parser's fuzzy matching."""
    apps = load_apps()
    aliases = []
    for entry in apps.values():
        aliases.extend(a.lower() for a in entry.get("aliases", []))
    return aliases


def canonical_app_key(alias: str | None) -> str | None:
    """Map a matched alias back to its canonical key in apps.yaml (e.g. 'vs code' -> 'vscode')."""
    if not alias:
        return None
    apps = load_apps()
    alias = alias.lower().strip()
    for key, entry in apps.items():
        if alias in [a.lower() for a in entry.get("aliases", [])] or alias == key:
            return key
    return None
