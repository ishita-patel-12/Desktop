import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intent import rules
from intent.schema import IntentType

KNOWN_APPS = ["chrome", "vscode", "vs code", "excel", "whatsapp", "notepad"]


def test_create_python_file():
    intent = rules.parse("create a python file called calculator", KNOWN_APPS)
    assert intent.type == IntentType.CREATE_FILE
    assert intent.extension == "py"
    assert "calculator" in intent.name.lower()


def test_create_file_missing_name_needs_clarification():
    intent = rules.parse("create a file", KNOWN_APPS)
    assert intent.type == IntentType.CREATE_FILE
    assert intent.needs_clarification


def test_create_folder_with_name():
    intent = rules.parse("create a folder called Projects", KNOWN_APPS)
    assert intent.type == IntentType.CREATE_FOLDER
    assert intent.name.lower() == "projects"


def test_open_known_app():
    intent = rules.parse("open chrome", KNOWN_APPS)
    assert intent.type == IntentType.OPEN_APP
    assert intent.app_name == "chrome"
    assert intent.confidence >= 0.8


def test_close_app():
    intent = rules.parse("close vs code", KNOWN_APPS)
    assert intent.type == IntentType.CLOSE_APP
    assert intent.app_name in ("vscode", "vs code")


def test_delete_file():
    intent = rules.parse("delete calculator.py", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert "calculator" in intent.name


def test_rename_file():
    intent = rules.parse("rename notes.txt to journal.txt", KNOWN_APPS)
    assert intent.type == IntentType.RENAME_FILE
    assert intent.new_name == "journal.txt"


def test_undo():
    intent = rules.parse("undo that", KNOWN_APPS)
    assert intent.type == IntentType.UNDO


def test_unknown_input_falls_through():
    intent = rules.parse("what's the meaning of life", KNOWN_APPS)
    assert intent.type == IntentType.UNKNOWN
    assert intent.confidence == 0.0


# --- Regression tests for reported bugs ---

def test_delete_folder_phrasing_strips_filler_word():
    intent = rules.parse("delete the Projects folder", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert intent.name == "Projects"  # not "Projects folder"


def test_delete_folder_called_phrasing():
    intent = rules.parse("delete folder called Projects", KNOWN_APPS)
    assert intent.name == "Projects"


def test_create_java_file_with_target_folder_not_misread_as_create_folder():
    intent = rules.parse("create a java file called Test in the Projects folder", KNOWN_APPS)
    assert intent.type == IntentType.CREATE_FILE  # NOT create_folder
    assert intent.extension == "java"
    assert intent.name == "Test"
    assert intent.target_folder == "Projects"


def test_create_folder_still_works_with_target_folder():
    intent = rules.parse("create a folder called Assets in the Documents folder", KNOWN_APPS)
    assert intent.type == IntentType.CREATE_FOLDER
    assert intent.name == "Assets"
    assert intent.target_folder == "Documents"


# --- Regression tests: bugs reported after initial build ---

def test_open_site_inside_specific_browser():
    intent = rules.parse("open youtube inside chrome", KNOWN_APPS)
    assert intent.type == IntentType.OPEN_WEBSITE
    assert intent.app_name == "chrome"
    assert intent.url_or_query == "youtube.com"


def test_create_file_with_folder_keyword_before_name():
    # "in folder called X" (folder keyword BEFORE the name) was previously
    # unmatched entirely, silently defaulting to the Desktop.
    intent = rules.parse("create a java file in folder called Ishita2", KNOWN_APPS)
    assert intent.type == IntentType.CREATE_FILE
    assert intent.extension == "java"
    assert intent.target_folder == "Ishita2"


def test_delete_folder_named_phrasing():
    # "named" was not stripped (only "called" was), leaving name="named Ishita2"
    intent = rules.parse("delete folder named Ishita2", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert intent.name == "Ishita2"


def test_delete_file_named_phrasing_with_extension():
    intent = rules.parse("delete file named report.txt", KNOWN_APPS)
    assert intent.type == IntentType.DELETE_FILE
    assert intent.name == "report.txt"