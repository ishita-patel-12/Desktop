"""
Open/close applications. Cross-platform launch commands come from
config/apps.yaml so adding a new app never requires touching code.
"""

from __future__ import annotations
import platform
import subprocess
from dataclasses import dataclass
from loguru import logger

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False
    logger.warning("psutil not installed - close_app will not work. `pip install psutil`.")

from config.loader import load_apps, canonical_app_key


@dataclass
class ExecResult:
    success: bool
    message: str


def _platform_key() -> str:
    system = platform.system().lower()
    return {"windows": "windows", "darwin": "darwin", "linux": "linux"}.get(system, "linux")


def open_app(app_name: str | None) -> ExecResult:
    if not app_name:
        return ExecResult(False, "Which app would you like me to open?")

    key = canonical_app_key(app_name)
    apps = load_apps()

    if not key or key not in apps:
        return ExecResult(False, f"I don't know how to open '{app_name}' yet. "
                                  f"Add it to config/apps.yaml.")

    platform_cfg = apps[key].get(_platform_key())
    if not platform_cfg:
        return ExecResult(False, f"No launch command configured for '{key}' on this OS.")

    try:
        subprocess.Popen(platform_cfg["cmd"])
        logger.info(f"Launched {key}")
        return ExecResult(True, f"Opening {key.title()}")
    except FileNotFoundError:
        return ExecResult(False, f"Couldn't find '{key}' installed on this system.")
    except Exception as e:
        logger.exception(e)
        return ExecResult(False, f"Failed to open {key}: {e}")


def close_app(app_name: str | None, confirmed: bool = False) -> ExecResult:
    """
    NOTE: per the Tier-1 safety requirement, callers (voice pipeline / gesture
    fist-close flow) MUST set confirmed=True only after the user has explicitly
    said yes / given a confirming gesture. This function does not itself ask -
    that's a UX concern that belongs one layer up.
    """
    if not confirmed:
        return ExecResult(False, "Close requires confirmation - not executed.")

    if not app_name:
        return ExecResult(False, "Which app would you like me to close?")

    if not _HAVE_PSUTIL:
        return ExecResult(False, "psutil is required to close apps.")

    key = canonical_app_key(app_name) or app_name.lower()
    apps = load_apps()
    process_names = apps.get(key, {}).get("process_names", [key])

    closed_any = False
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = (proc.info["name"] or "").lower()
            pstem = pname.removesuffix(".exe")
            matches = False
            for target in process_names:
                t = target.lower()
                t_stem = t.removesuffix(".exe")
                # Exact name or stem match, or space-separated bundle names (e.g. "Code Helper")
                if pname == t or pstem == t_stem or (" " in t and t in pname):
                    matches = True
                    break
            if matches:
                proc.terminate()
                closed_any = True
                logger.info(f"Terminated process {pname} (pid {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if closed_any:
        return ExecResult(True, f"Closed {key.title()}")
    return ExecResult(False, f"Couldn't find a running process for '{key}'")


def find_running_app_display_name(app_name: str | None) -> str:
    """Best-effort friendly name for the confirmation prompt, e.g.
    'Should I close Chrome?'"""
    if not app_name:
        return "the app"
    key = canonical_app_key(app_name)
    return key.title() if key else app_name.title()
