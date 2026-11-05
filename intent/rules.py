"""
Rule-based intent parser.

Design rationale (put this in the README too):
  Most voice-assistant commands are highly templated ("create a file called X",
  "open Chrome", "close this"). Running every utterance through an LLM for
  that is slow, non-deterministic, and wasteful. Regex/keyword matching
  handles the large majority of real commands instantly and deterministically.
  Only genuinely ambiguous phrasing falls through to the LLM fallback
  (see intent/llm_fallback.py). This keeps the assistant fast, debuggable,
  and cheap, while still handling the long tail gracefully.
"""

from __future__ import annotations
import re
from typing import Optional

from intent.schema import Intent, IntentType

try:
    from rapidfuzz import fuzz
    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False


# --- Known app aliases (kept in sync with config/apps.yaml at runtime by
# the caller passing `known_apps`; these are just a fallback default set) ---
_DEFAULT_APPS = ["chrome", "vscode", "vs code", "excel", "whatsapp", "notepad"]

# Bare site names ("youtube", "gmail") people say without ".com" attached.
# Falls back to appending ".com" for anything not in this table, so new
# sites work generically without needing an entry here.
_COMMON_SITES = {
    "youtube": "youtube.com",
    "gmail": "mail.google.com",
    "google": "google.com",
    "github": "github.com",
    "amazon": "amazon.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "reddit": "reddit.com",
    "twitter": "twitter.com",
    "x": "x.com",
}


def site_to_url(site: str) -> str:
    key = site.strip().lower()
    if key in _COMMON_SITES:
        return _COMMON_SITES[key]
    if "." in key:
        return key
    return f"{key}.com"


# kept as an alias - internal call sites in this file already use the
# underscore name, and it's harmless to keep both.
_site_to_url = site_to_url


# Conversational preambles people say before the actual command.
# These are stripped so patterns don't need to handle every combination.
_PREAMBLE = re.compile(
    r"^(?:"
    r"i(?:'d| would) like (?:you )?to\s+|"
    r"i want (?:you )?to\s+|"
    r"i need (?:you )?to\s+|"
    r"can you\s+|"
    r"could you\s+|"
    r"please\s+|"
    r"would you\s+|"
    r"hey assistant[,.]?\s+|"
    r"assistant[,.]?\s+|"
    r"go ahead and\s+|"
    r"just\s+"
    r")+",
    re.I,
)

# Common Whisper mishearings for frequent words.
_STT_FIXUPS: list[tuple[re.Pattern, str]] = [
    # "create her/the/your/an folder" → "create a folder"
    (re.compile(r"\bcreate\s+(?:her|the|your|an|my)\s+", re.I), "create a "),
    # "make her/the folder" → "make a folder"
    (re.compile(r"\bmake\s+(?:her|the|your|an|my)\s+", re.I), "make a "),
    # trailing punctuation that confuses name extraction
    (re.compile(r"[.!?,;]+\s*$"), ""),
    # "period" spoken literally at end
    (re.compile(r"\s+period\s*$", re.I), ""),
]


def normalize_stt(text: str) -> str:
    """Clean up common Whisper STT artifacts before regex matching."""
    for pattern, replacement in _STT_FIXUPS:
        text = pattern.sub(replacement, text)
    return text.strip()

_EXTENSION_WORDS = {
    "python": "py", "python file": "py",
    "text": "txt", "text file": "txt", "notepad": "txt",
    "word": "docx", "word document": "docx",
    "excel": "xlsx", "spreadsheet": "xlsx",
    "markdown": "md",
    "javascript": "js",
    "java": "java",
    "html": "html",
    "css": "css",
    "json": "json",
    "c++": "cpp", "cpp": "cpp",
    "c#": "cs", "csharp": "cs",
}

# Ordered: more specific patterns first.
# NOTE: CREATE_FILE / CREATE_FOLDER / DELETE_FILE / RENAME_FILE / COPY_FILE
# are NOT handled via this table - each needs its own location-clause
# handling first (see _parse_create/_parse_delete/_parse_rename/_parse_copy
# below), for the same reason CREATE does: a single greedy `name` capture
# group can't tell "the object's name" apart from "the folder it lives in"
# when both are just bare words in the same sentence, e.g. "delete report
# from Downloads" was previously captured whole as name="report from
# Downloads" (folder silently ignored, lookup then failed).
_PATTERNS: list[tuple[IntentType, re.Pattern]] = [

    # Close: handles 'close/quit/exit X', 'can you close X', 'please close X'
    (IntentType.CLOSE_APP, re.compile(
        r"(?:(?:can\s+you|please|could\s+you)\s+)?\b(close|quit|exit|shut\s+down|kill)\b\s+(?P<app>[\w\- ]+)", re.I)),

    # NOTE: must come BEFORE the generic OPEN_APP pattern below.
    (IntentType.OPEN_WEBSITE, re.compile(
        r"(?:(?:can\s+you|please|could\s+you)\s+)?\b(?:open|go to|launch)\b\s+(?P<site>[\w\- .]+?)\s+"
        r"(?:in|inside|using|on)\s+(?P<app>chrome|google chrome|edge|firefox|browser)\b\.?\s*$",
        re.I)),

    # Open app: handles 'open X', 'launch X', 'start X', 'can you open X', 'please open X for me'
    (IntentType.OPEN_APP, re.compile(
        r"(?:(?:can\s+you|please|could\s+you)\s+)?\b(open|launch|start|run)\b\s+(?P<app>[\w\- ]+?)(?:\s+(?:app|application|for\s+me))?$", re.I)),

    (IntentType.WEB_SEARCH, re.compile(
        r"\bsearch\b(?:\s+for)?\s+(?P<query>.+?)(?:\s+on\s+(?:google|the\s+web))?$", re.I)),

    (IntentType.OPEN_WEBSITE, re.compile(
        r"\b(go to|open)\b\s+(?P<url>[\w\-]+\.[\w./\-]+)", re.I)),

    (IntentType.UNDO, re.compile(r"\bundo\b(?:\s+that|\s+last\s+action)?", re.I)),
]


# ---------------------------------------------------------------------------
# Location-clause handling shared by delete/rename/copy.
#
# BUG FIXED HERE: DELETE_FILE/RENAME_FILE previously had no location-clause
# support at all, and COPY_FILE only captured a *destination*. That meant
# "delete report.txt from Downloads" matched name = "report.txt from
# Downloads" verbatim (the greedy name group doesn't stop at "from"), which
# then failed to find anything on disk since no file is literally named
# that. Same problem hit "rename X to Y in the Z folder" - the trailing
# location clause got swallowed into `new_name`. The fix mirrors what
# _parse_create already does for CREATE_FILE/CREATE_FOLDER: strip the
# location clause off the sentence FIRST, then run the name-extraction
# regex against what's left.
# ---------------------------------------------------------------------------
_LOCATION_TAIL = re.compile(
    r"\s+(?:from|in(?:side)?)\s+(?:the\s+)?(?P<folder>[\w\- ]+?)(?:\s+folder)?\.?\s*$",
    re.I,
)


def _strip_trailing_location(text: str) -> tuple[str, Optional[str]]:
    """Strip a trailing 'from/in (the) X (folder)' clause off a sentence,
    returning (remaining_core_text, folder_or_None)."""
    m = _LOCATION_TAIL.search(text)
    if m:
        return text[: m.start()].rstrip(), m.group("folder").strip()
    return text, None


_DELETE_CORE = re.compile(
    r"\b(delete|remove|trash)\b\s+(?:the\s+)?(?:file\s+|folder\s+)?(?:called\s+|named\s+)?(?P<name>[\w\- .]+)", re.I)

_RENAME_CORE = re.compile(
    r"\brename\b\s+(?:the\s+)?(?:file\s+|folder\s+)?(?:called\s+|named\s+)?"
    r"(?P<name>[\w\- .]+?)\s+to\s+(?P<new_name>[\w\- .]+)", re.I)

# Two copy shapes: with an explicit source ("copy X from Y to Z") and
# without one ("copy X to Z", source resolved generically at execution time).
_COPY_WITH_SOURCE = re.compile(
    r"\bcopy\b\s+(?:the\s+)?(?:file\s+|folder\s+)?(?:called\s+|named\s+)?"
    r"(?P<name>[\w\- .]+?)\s+from\s+(?:the\s+)?(?P<src>[\w\- ]+?)\s+to\s+(?:the\s+)?(?P<dst>[\w\- ]+)", re.I)

_COPY_CORE = re.compile(
    r"\bcopy\b\s+(?:the\s+)?(?:file\s+|folder\s+)?(?:called\s+|named\s+)?"
    r"(?P<name>[\w\- .]+?)\s+to\s+(?:the\s+)?(?P<dst>[\w\- ]+)", re.I)


def _parse_delete(text: str) -> Optional[Intent]:
    if not re.search(r"\b(delete|remove|trash)\b", text, re.I):
        return None
    core, folder = _strip_trailing_location(text)
    m = _DELETE_CORE.search(core)
    if not m:
        return None
    intent = Intent(type=IntentType.DELETE_FILE, raw_text=text, confidence=0.9)
    intent.name = _clean_delete_name(m.group("name"))
    intent.target_folder = folder
    intent.needs_clarification = not intent.name
    if intent.needs_clarification:
        intent.missing_slots = ["name"]
    return intent


def _parse_rename(text: str) -> Optional[Intent]:
    if not re.search(r"\brename\b", text, re.I):
        return None
    core, folder = _strip_trailing_location(text)
    m = _RENAME_CORE.search(core)
    if not m:
        return None
    intent = Intent(type=IntentType.RENAME_FILE, raw_text=text, confidence=0.9)
    name = _clean_delete_name(m.group("name"))
    new_name = m.group("new_name").strip()
    intent.name = name or None
    intent.new_name = new_name or None
    intent.target_folder = folder
    missing = []
    if not intent.name:
        missing.append("name")
    if not intent.new_name:
        missing.append("new_name")
    if missing:
        intent.missing_slots = missing
        intent.needs_clarification = True
    return intent


def _parse_copy(text: str) -> Optional[Intent]:
    if not re.search(r"\bcopy\b", text, re.I):
        return None

    m = _COPY_WITH_SOURCE.search(text)
    if m:
        intent = Intent(type=IntentType.COPY_FILE, raw_text=text, confidence=0.9)
        name = _clean_delete_name(m.group("name"))
        intent.name = name or None
        intent.source_folder = m.group("src").strip()
        intent.target_folder = m.group("dst").strip()
        missing = []
        if not intent.name:
            missing.append("name")
        if not intent.target_folder:
            missing.append("target_folder")
        if missing:
            intent.missing_slots = missing
            intent.needs_clarification = True
        return intent

    m = _COPY_CORE.search(text)
    if not m:
        return None
    intent = Intent(type=IntentType.COPY_FILE, raw_text=text, confidence=0.9)
    name = _clean_delete_name(m.group("name"))
    intent.name = name or None
    intent.target_folder = m.group("dst").strip()
    missing = []
    if not intent.name:
        missing.append("name")
    if not intent.target_folder:
        missing.append("target_folder")
    if missing:
        intent.missing_slots = missing
        intent.needs_clarification = True
    return intent


def _extract_extension(text: str, name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Pull a file extension out of the name or surrounding words.
    Returns (cleaned_name, extension)."""
    text_l = text.lower()

    # explicit ".ext" already in the name
    if name:
        m = re.search(r"\.(\w{1,5})$", name.strip())
        if m:
            return name.strip()[: m.start()].strip(), m.group(1).lower()

    for phrase, ext in _EXTENSION_WORDS.items():
        if phrase in text_l:
            return name, ext

    return name, None


def _clean_delete_name(raw_name: str) -> str:
    """'delete the Projects folder' was matching name='Projects folder'
    (trailing filler word attached), which then failed to find anything on
    disk since nothing is literally named 'Projects folder'. Strip these
    filler words so the name matches what's actually on disk.

    Bug fix: only 'called' was stripped, so 'delete folder named X' left
    the name as 'named X' (lookup then failed, e.g. finding 'namedX').
    'called' and 'named' are used interchangeably by people, so both must
    be handled - and in either order relative to the file/folder keyword
    ('delete folder named X' vs 'delete the file called X')."""
    name = raw_name.strip()
    name = re.sub(r"^(?:file|folder)\s+", "", name, flags=re.I)
    name = re.sub(r"^(?:called|named)\s+", "", name, flags=re.I)
    name = re.sub(r"\s+(?:file|folder)$", "", name, flags=re.I)
    clean = name.strip()
    if clean.lower() in ("file", "folder", "the file", "the folder", "it", "this"):
        return ""
    return clean


def _parse_create(text: str) -> Optional[Intent]:
    """Handles 'create a file called X' and 'create a folder called X',
    including an optional trailing '... in the Y folder' location clause.

    Location is stripped FIRST, then file-vs-folder is decided from what's
    left. This matters because "create a java file called Test in the
    Projects folder" contains the word "folder" (as part of the location
    clause) - deciding file-vs-folder before stripping that clause would
    misread it as "create a folder".

    Two orderings of the location clause are supported, since people phrase
    it both ways:
      - "... in the Projects folder"           (name, then the word "folder")
      - "... in folder called Projects"        (the word "folder", then the name)
    Without the second pattern, "create a java file in folder called
    Ishita2" fell through with no location match at all, silently
    defaulting to the Desktop instead of the Ishita2 folder.
    """
    if not re.search(r"\b(create|make|new)\b", text, re.I):
        return None

    _LOCATION_PATTERNS = [
        re.compile(r"\bin(?:side)?\s+(?:the\s+)?(?P<folder>[\w\- ]+?)\s*folder\b\.?\s*$", re.I),
        re.compile(r"\bin(?:side)?\s+(?:the\s+)?folder\s+(?:called|named)\s+(?P<folder>[\w\- .]+?)\.?\s*$", re.I),
        re.compile(r"\bin(?:side)?\s+(?:the\s+)?(?P<folder>desktop|documents|downloads)\b\.?\s*$", re.I),
    ]
    loc_match = None
    for pattern in _LOCATION_PATTERNS:
        loc_match = pattern.search(text)
        if loc_match:
            break

    folder = loc_match.group("folder").strip() if loc_match else None
    core = text[: loc_match.start()].strip() if loc_match else text.strip()

    is_folder = bool(re.search(r"\b(create|make|new)\b\s*(?:a|an)?\s*folder\b", core, re.I))
    is_file = bool(re.search(r"\b(create|make|new)\b\s*(?:a|an)?\s*(?:[\w+#]+\s+){0,2}file\b", core, re.I))

    if not is_folder and not is_file:
        return None
    itype = IntentType.CREATE_FOLDER if (is_folder and not is_file) else IntentType.CREATE_FILE

    name_match = re.search(r"\b(?:called|named)\s+(?P<name>[\w\- .]+?)\s*$", core, re.I)
    name = name_match.group("name").strip() if name_match else None

    intent = Intent(type=itype, raw_text=text, confidence=0.9)
    name, ext = _extract_extension(text, name)
    intent.name = name.strip() if name else None
    intent.extension = ext
    intent.target_folder = folder

    missing = []
    if not intent.name:
        missing.append("name")
    if itype == IntentType.CREATE_FILE and not intent.extension:
        missing.append("extension")
    intent.missing_slots = missing
    intent.needs_clarification = bool(missing)
    return intent



def _match_known_app(candidate: str, known_apps: list[str]) -> Optional[str]:
    candidate = candidate.strip().lower()
    if candidate in known_apps:
        return candidate
    if _HAVE_RAPIDFUZZ:
        best, score = None, 0
        for app in known_apps:
            s = fuzz.ratio(candidate, app)
            if s > score:
                best, score = app, s
        if score >= 75:
            return best
    else:
        for app in known_apps:
            if candidate in app or app in candidate:
                return app
    return None


def parse(text: str, known_apps: Optional[list[str]] = None) -> Intent:
    """Try to classify raw text into a structured Intent using regex rules.

    If no pattern matches confidently, returns IntentType.UNKNOWN with
    confidence 0.0 so the caller knows to fall back to the LLM.
    """
    known_apps = known_apps or _DEFAULT_APPS
    text = text.strip()
    if not text:
        return Intent(raw_text=text, confidence=0.0)

    # Fix STT artifacts first, then strip conversational preamble.
    original_text = text
    text = normalize_stt(text)
    text = _PREAMBLE.sub("", text).strip() or text

    create_intent = _parse_create(text)
    if create_intent:
        create_intent.raw_text = original_text
        return create_intent

    for parser in (_parse_delete, _parse_rename, _parse_copy):
        intent = parser(text)
        if intent:
            intent.raw_text = original_text
            return intent

    for itype, pattern in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue

        groups = m.groupdict()
        intent = Intent(type=itype, raw_text=text, confidence=0.9)

        if itype in (IntentType.OPEN_APP, IntentType.CLOSE_APP):
            raw_app = groups.get("app", "").strip()
            matched = _match_known_app(raw_app, known_apps)
            intent.app_name = matched or raw_app
            if not matched:
                # Keep confidence high for OPEN_APP: dispatch() has a generic
                # website fallback so even unrecognized names are handled safely.
                if itype == IntentType.OPEN_APP or raw_app.lower() in _COMMON_SITES:
                    intent.confidence = 0.85
                else:
                    intent.confidence = 0.4  # truly unrecognized, lower confidence

        elif itype == IntentType.WEB_SEARCH:
            intent.url_or_query = groups.get("query", "").strip()

        elif itype == IntentType.OPEN_WEBSITE:
            if groups.get("site"):
                # "open youtube inside chrome" shape: site + target browser
                intent.url_or_query = _site_to_url(groups.get("site", ""))
                intent.app_name = groups.get("app", "").strip().lower()
            else:
                intent.url_or_query = groups.get("url", "").strip()

        return intent

    # Nothing matched -> caller should try the LLM fallback
    return Intent(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)