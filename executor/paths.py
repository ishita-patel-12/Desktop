"""
Resolves things like "the Projects folder" or "Desktop" into an absolute,
safe path under the user's home directory. Never hardcodes machine-specific
paths - everything is relative to Path.home() unless a named location is
configured in settings.yaml.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from loguru import logger

from config.loader import load_settings

_NAMED_LOCATIONS_CACHE: dict[str, Path] | None = None


def sanitize_filename(name: str) -> str:
    """Sanitize a user-provided filename to prevent path traversal and invalid filesystem characters."""
    clean = Path(name.strip()).name.strip()
    clean = re.sub(r'[<>:"/\\|?*]', "_", clean)
    return clean or "untitled"


def _named_locations() -> dict[str, Path]:
    global _NAMED_LOCATIONS_CACHE
    if _NAMED_LOCATIONS_CACHE is None:
        settings = load_settings()
        raw = settings.get("paths", {})
        locations: dict[str, Path] = {}
        home = Path.home()
        for key, value in raw.items():
            if key == "default_create_location":
                continue
            val_str = str(value).strip()
            if val_str.startswith("~/") or val_str.startswith("~\\"):
                locations[key] = home / val_str[2:]
            elif val_str == "~":
                locations[key] = home
            else:
                locations[key] = Path(val_str).expanduser()
        _NAMED_LOCATIONS_CACHE = locations
    return _NAMED_LOCATIONS_CACHE


def default_create_location() -> Path:
    settings = load_settings()
    key = settings.get("paths", {}).get("default_create_location", "desktop")
    return _named_locations().get(key, Path.home() / "Desktop")


def resolve_folder(name: Optional[str], search_depth: int = 2) -> Path:
    """
    Resolve a spoken/typed folder reference to an absolute path.

    Resolution order:
      1. No name given -> the configured default (usually Desktop).
      2. Matches a named location in settings.yaml (desktop/documents/downloads).
      3. Exact subfolder of home directory (~/<name>).
      4. Shallow search under home for a folder matching that name
         (handles "the Projects folder" when Projects lives a level or two deep).
      5. Fallback: default location (never silently writes somewhere unexpected
         without at least logging a warning).
    """
    if not name:
        return default_create_location()

    # Prevent directory traversal
    clean_parts = [p for p in Path(name.strip()).parts if p not in ("..", ".", "/", "\\")]
    if not clean_parts:
        return default_create_location()
    clean_target = clean_parts[-1].strip().strip(".")
    name_clean = clean_target.lower()

    named = _named_locations()
    if name_clean in named:
        path = named[name_clean]
        path.mkdir(parents=True, exist_ok=True)
        return path

    home = Path.home()
    direct = home / clean_target
    if direct.is_dir():
        return direct

    # Shallow BFS search under home for a matching folder name
    match = _search_for_folder(home, name.strip(), max_depth=search_depth)
    if match:
        return match

    logger.warning(
        f"Could not locate folder '{name}' under home directory; "
        f"falling back to default create location."
    )
    return default_create_location()


def _search_for_folder(root: Path, target_name: str, max_depth: int) -> Optional[Path]:
    target_lower = target_name.lower()
    frontier = [(root, 0)]
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}

    while frontier:
        current, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            for child in current.iterdir():
                if not child.is_dir() or child.name.startswith(".") or child.name in skip_dirs:
                    continue
                if child.name.lower() == target_lower:
                    return child
                frontier.append((child, depth + 1))
        except PermissionError:
            continue
    return None


def ensure_unique_path(path: Path) -> Path:
    """If path already exists, append (1), (2)... before the extension so
    we never silently overwrite an existing file."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    candidate = parent / f"{stem} ({counter}){suffix}"
    while candidate.exists():
        counter += 1
        candidate = parent / f"{stem} ({counter}){suffix}"
    return candidate


# Search order for find_file_anywhere(): most-likely locations checked
# first and cheaply (no recursion), before falling back to a bounded walk.
_COMMON_LOCATIONS_ORDER = ("desktop", "documents", "downloads")


def find_file_anywhere(name: str, max_depth: int = 2) -> list[Path]:
    """Fallback file/folder search used when the user doesn't say which
    folder something is in - by far the most common phrasing ("delete
    resume.pdf", not "delete resume.pdf from Downloads"). Without this,
    delete/rename/copy only ever looked in one single default folder
    (usually Desktop) and reported "couldn't find it" for anything sitting
    anywhere else, which is the single biggest cause of the assistant
    failing to recognize files that do exist.

    Checks Desktop, Documents, and Downloads first (flat glob, cheap), and
    stops at the first location that has ANY match - this keeps a same-named
    file on both Desktop and Downloads from being silently resolved to the
    wrong one; the caller sees it as an ambiguous multi-match instead (see
    files.py) rather than this function guessing across locations.
    Only if none of those three have anything does it fall back to a
    depth-bounded walk under the home directory (same skip-list and depth
    cap `resolve_folder`'s folder search already uses, so this stays fast
    even on a large home directory instead of doing an unbounded recursive
    scan of the whole disk).
    """
    home = Path.home()
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}

    named = _named_locations()
    seen: set[Path] = set()
    ordered_roots: list[Path] = []
    for key in _COMMON_LOCATIONS_ORDER:
        loc = named.get(key)
        if loc and loc not in seen:
            ordered_roots.append(loc)
            seen.add(loc)

    for root in ordered_roots:
        if not root.is_dir():
            continue
        matches = [c for c in root.glob(f"{name}*") if c.exists()]
        if matches:
            return matches

    # Last resort: shallow BFS under home, same bound as folder resolution.
    frontier = [(home, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            for child in current.iterdir():
                if child.name.startswith(".") or child.name in skip_dirs:
                    continue
                if child.name.lower().startswith(name.lower()):
                    return [child]
                if child.is_dir():
                    frontier.append((child, depth + 1))
        except PermissionError:
            continue
    return []
