"""
Structured intent representation.

This is the ONLY contract between "decide what to do" (rules / LLM) and
"actually do it" (executor). Keeping it a validated pydantic model means:
  - the executor never has to guess about missing/malformed fields
  - swapping the rule parser for an LLM later requires zero executor changes
  - it's trivially loggable / testable as plain dict-like data
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    CREATE_FILE = "create_file"
    CREATE_FOLDER = "create_folder"
    DELETE_FILE = "delete_file"
    RENAME_FILE = "rename_file"
    COPY_FILE = "copy_file"
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    OPEN_WEBSITE = "open_website"
    WEB_SEARCH = "web_search"
    UNDO = "undo"
    UNKNOWN = "unknown"


class Intent(BaseModel):
    type: IntentType = IntentType.UNKNOWN

    # File / folder related slots
    name: Optional[str] = None            # e.g. "calculator"
    extension: Optional[str] = None        # e.g. "py", "txt"
    target_folder: Optional[str] = None    # raw text like "Projects" or "Desktop"
    source_folder: Optional[str] = None    # raw text; only set when the user names
                                            # an explicit source, e.g. "copy X from Y to Z"
    new_name: Optional[str] = None         # for rename

    # App / web related slots
    app_name: Optional[str] = None
    url_or_query: Optional[str] = None

    # Bookkeeping
    raw_text: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    missing_slots: list[str] = Field(default_factory=list)

    def is_confident(self, threshold: float = 0.6) -> bool:
        return self.type != IntentType.UNKNOWN and self.confidence >= threshold
