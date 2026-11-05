"""
Regression tests for the file/folder recognition fixes in intent/rules.py,
executor/files.py, and executor/paths.py.

Covers the reported bug: "it still has trouble recognising files and
folders". Two separate root causes were found and are each tested here:

  1. intent/rules.py: DELETE_FILE/RENAME_FILE had no location-clause
     parsing at all, and COPY_FILE only captured a destination - so a
     trailing "from/in X" got swallowed into the file name itself instead
     of being recognized as a folder to search.
  2. executor/files.py: delete/rename/copy only ever looked in ONE folder
     (the explicit one, or the default create location) and gave up
     instead of checking the other common places a file is likely to be.

Run with: pytest tests/test_file_search.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from intent import rules
from intent.schema import IntentType
from executor import files as files_executor

KNOWN_APPS = ["chrome", "vscode", "vs code", "excel", "whatsapp", "notepad"]


# --- intent parsing: location clauses no longer get swallowed into names ---

def test_delete_with_from_clause_extracts_folder():
    intent = rules.parse("delete report.txt from Downloads", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert intent.name == "report.txt"
    assert intent.target_folder == "Downloads"


def test_delete_with_in_the_folder_clause():
    intent = rules.parse("delete resume.pdf in the Documents folder", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert intent.name == "resume.pdf"
    assert intent.target_folder == "Documents"


def test_rename_with_trailing_location_clause():
    intent = rules.parse("rename notes.txt to journal.txt in the Projects folder", KNOWN_APPS)
    assert intent.type == IntentType.RENAME_FILE
    assert intent.name == "notes.txt"
    assert intent.new_name == "journal.txt"  # not "journal.txt in the Projects folder"
    assert intent.target_folder == "Projects"


def test_copy_without_source_has_no_source_folder():
    intent = rules.parse("copy resume.pdf to Documents", KNOWN_APPS)
    assert intent.type == IntentType.COPY_FILE
    assert intent.name == "resume.pdf"
    assert intent.source_folder is None
    assert intent.target_folder == "Documents"


def test_copy_with_explicit_source_and_destination():
    intent = rules.parse("copy resume.pdf from Downloads to Documents", KNOWN_APPS)
    assert intent.type == IntentType.COPY_FILE
    assert intent.name == "resume.pdf"
    assert intent.source_folder == "Downloads"
    assert intent.target_folder == "Documents"


def test_delete_without_location_clause_is_unaffected():
    """Plain 'delete X' (no location words at all) must keep working
    exactly as before - target_folder should just be None."""
    intent = rules.parse("delete calculator.py", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert intent.name == "calculator.py"
    assert intent.target_folder is None


# --- executor: falls back to Desktop/Documents/Downloads when no folder given ---

def test_delete_file_falls_back_to_other_locations(tmp_path):
    desktop = tmp_path / "Desktop"
    downloads = tmp_path / "Downloads"
    documents = tmp_path / "Documents"
    for d in (desktop, downloads, documents):
        d.mkdir()
    target = downloads / "report.txt"
    target.write_text("hi")

    fake_settings = {
        "paths": {"desktop": str(desktop), "documents": str(documents),
                   "downloads": str(downloads), "default_create_location": "desktop"}
    }
    with patch("executor.paths.load_settings", return_value=fake_settings), \
         patch("executor.paths._NAMED_LOCATIONS_CACHE", None), \
         patch("executor.files._HAVE_SEND2TRASH", False):
        result = files_executor.delete_file("report.txt")  # no folder specified

    assert result.success, result.message
    assert not target.exists()


def test_copy_file_uses_explicit_source_folder(tmp_path):
    desktop = tmp_path / "Desktop"
    downloads = tmp_path / "Downloads"
    documents = tmp_path / "Documents"
    for d in (desktop, downloads, documents):
        d.mkdir()
    src = downloads / "resume.pdf"
    src.write_text("resume contents")

    fake_settings = {
        "paths": {"desktop": str(desktop), "documents": str(documents),
                   "downloads": str(downloads), "default_create_location": "desktop"}
    }
    with patch("executor.paths.load_settings", return_value=fake_settings), \
         patch("executor.paths._NAMED_LOCATIONS_CACHE", None):
        # BUG FIXED: source_folder_name used to be silently ignored and the
        # Desktop (default) was always searched instead, so this would have
        # failed to find resume.pdf (it's in Downloads, not Desktop).
        result = files_executor.copy_file("resume.pdf", "documents", "downloads")

    assert result.success, result.message
    assert (documents / "resume.pdf").exists()
