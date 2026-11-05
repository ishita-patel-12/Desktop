"""
Single entry point the assistant calls with a finished Intent. This is the
only place that knows how an Intent maps to executor modules - keeping that
mapping in one file makes it trivial to add new intent types later.

Confirmation-requiring intents (delete, close-app) return a result with
`needs_confirmation=True` instead of executing immediately; the caller
(voice loop / gesture flow) is responsible for asking the user and calling
back in with `confirmed=True`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from intent.schema import Intent, IntentType
from executor import files, apps, browser
from config.loader import load_settings


@dataclass
class DispatchResult:
    success: bool
    message: str
    needs_confirmation: bool = False
    confirmation_prompt: Optional[str] = None


_UNDO_STACK: list[dict] = []  # Tier 2 - simple in-memory undo stack


def dispatch(intent: Intent, confirmed: bool = False) -> DispatchResult:
    settings = load_settings()
    safety = settings.get("safety", {})

    if intent.needs_clarification:
        missing = ", ".join(intent.missing_slots) or "some details"
        return DispatchResult(False, f"I need more info: {missing}. Could you clarify?")

    if intent.type == IntentType.CREATE_FILE:
        result = files.create_file(intent.name, intent.extension, intent.target_folder)
        if result.success:
            _UNDO_STACK.append({"type": "create_file", "path": str(result.path)})
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.CREATE_FOLDER:
        result = files.create_folder(intent.name, intent.target_folder)
        if result.success:
            _UNDO_STACK.append({"type": "create_folder", "path": str(result.path)})
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.DELETE_FILE:
        if safety.get("require_confirmation_for_delete", True) and not confirmed:
            return DispatchResult(
                False, "", needs_confirmation=True,
                confirmation_prompt=f"Are you sure you want to delete '{intent.name}'?"
            )
        result = files.delete_file(intent.name, intent.target_folder)
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.RENAME_FILE:
        result = files.rename_file(intent.name, intent.new_name, intent.target_folder)
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.COPY_FILE:
        result = files.copy_file(intent.name, intent.target_folder, intent.source_folder)
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.OPEN_APP:
        result = apps.open_app(intent.app_name)
        if not result.success:
            # Unrecognized "app" name is very often actually a website the
            # person wants opened ("open youtube", "open netflix") rather
            # than an installed desktop app we have launch config for.
            # This is a generic heuristic (name -> name.com), not a
            # hardcoded list of sites, so it covers new site names too.
            from intent.rules import site_to_url
            url = site_to_url(intent.app_name)
            web_result = browser.open_url(url)
            if web_result.success:
                logger.info(f"'{intent.app_name}' wasn't a known app; opened {url} as a website instead.")
                return DispatchResult(
                    True, f"I didn't recognize '{intent.app_name}' as an installed app, "
                          f"so I opened {url} instead."
                )
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.CLOSE_APP:
        if safety.get("require_confirmation_for_close", True) and not confirmed:
            display_name = apps.find_running_app_display_name(intent.app_name)
            return DispatchResult(
                False, "", needs_confirmation=True,
                confirmation_prompt=f"Should I close {display_name}?"
            )
        result = apps.close_app(intent.app_name, confirmed=True)
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.OPEN_WEBSITE:
        if intent.app_name:
            result = browser.open_url_in_app(intent.url_or_query, intent.app_name)
        else:
            result = browser.open_url(intent.url_or_query)
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.WEB_SEARCH:
        result = browser.web_search(intent.url_or_query)
        return DispatchResult(result.success, result.message)

    if intent.type == IntentType.UNDO:
        return _undo_last()

    logger.warning(f"No handler for intent type {intent.type}; raw text: {intent.raw_text!r}")
    return DispatchResult(False, "Sorry, I didn't understand that.")


def _undo_last() -> DispatchResult:
    if not _UNDO_STACK:
        return DispatchResult(False, "Nothing to undo.")
    action = _UNDO_STACK.pop()
    from pathlib import Path
    try:
        if action["type"] in ("create_file", "create_folder"):
            path = Path(action["path"])
            if path.exists():
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
            return DispatchResult(True, f"Undid creation of {path.name}")
    except Exception as e:
        logger.exception(e)
        return DispatchResult(False, f"Couldn't undo: {e}")
    return DispatchResult(False, "Don't know how to undo that action type.")