from __future__ import annotations
import platform
import subprocess
import webbrowser
from urllib.parse import quote, urlparse
from dataclasses import dataclass
from loguru import logger

from config.loader import load_apps, canonical_app_key


@dataclass
class ExecResult:
    success: bool
    message: str


def _platform_key() -> str:
    system = platform.system().lower()
    return {"windows": "windows", "darwin": "darwin", "linux": "linux"}.get(system, "linux")


def _normalize_url(url_or_domain: str | None) -> str | None:
    if not url_or_domain:
        return None
    raw = url_or_domain.strip()
    if not raw:
        return None
    # Block unsafe/local schemes
    if raw.lower().startswith(("javascript:", "file:", "data:", "vbscript:")):
        logger.warning(f"Blocked unsafe or unsupported URL scheme: {raw}")
        return None
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return raw


def open_url(url_or_domain: str) -> ExecResult:
    url = _normalize_url(url_or_domain)
    if not url:
        return ExecResult(False, f"Invalid or unsafe URL: '{url_or_domain}'")
    ok = webbrowser.open(url)
    logger.info(f"Opened browser at {url}")
    return ExecResult(ok, f"Opening {url}" if ok else f"Couldn't open a browser for {url}")


def open_url_in_app(url_or_domain: str, app_name: str) -> ExecResult:
    """Opens a URL in a SPECIFIC browser (e.g. 'open youtube inside chrome'),
    instead of whatever the OS default browser happens to be.

    Falls back to open_url() (default browser) if the requested app isn't a
    recognized browser in config/apps.yaml - this keeps "open X inside Y"
    working even for a browser we don't have launch config for, rather than
    failing outright.
    """
    url = _normalize_url(url_or_domain)
    if not url:
        return ExecResult(False, f"Invalid or unsafe URL: '{url_or_domain}'")

    key = canonical_app_key(app_name) or app_name.strip().lower()
    if key in (None, "browser", ""):
        return open_url(url)

    apps = load_apps()
    entry = apps.get(key)
    platform_cfg = entry.get(_platform_key()) if entry else None
    if not platform_cfg:
        logger.warning(f"No launch config for browser '{key}'; using default browser instead.")
        return open_url(url)

    try:
        cmd = list(platform_cfg["cmd"]) + [url]
        subprocess.Popen(cmd)
        logger.info(f"Opened {url} in {key}")
        return ExecResult(True, f"Opening {url} in {key.title()}")
    except FileNotFoundError:
        logger.warning(f"'{key}' not found on this system; falling back to default browser.")
        return open_url(url)
    except Exception as e:
        logger.exception(e)
        return ExecResult(False, f"Couldn't open {url} in {key}: {e}")


def web_search(query: str | None) -> ExecResult:
    if not query or not query.strip():
        return ExecResult(False, "What would you like me to search for?")
    clean_query = query.strip()
    url = f"https://www.google.com/search?q={quote(clean_query)}"
    ok = webbrowser.open(url)
    logger.info(f"Searched web for: {clean_query}")
    return ExecResult(ok, f"Searching for {clean_query}" if ok else "Couldn't open a browser")