"""
Security, edge-case, and safety tests for AI Desktop Assistant.
Validates path traversal prevention, pickle deserialization safety,
URL scheme restrictions, process matching safety, and slot clarification.
"""

import sys
import pickle
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from executor.paths import sanitize_filename, resolve_folder
from executor.browser import _normalize_url, open_url
from gestures.teachable import _RestrictedUnpickler, TrainedGestureModel, TeachableGestureStore
from intent import rules
from intent.schema import IntentType, Intent


# --- Security: Insecure Deserialization Protection ---

def test_restricted_unpickler_blocks_dangerous_payload():
    """Ensure that arbitrary code execution via pickle (e.g. os.system) is blocked."""
    import io
    class Exploit:
        def __reduce__(self):
            import os
            return (os.system, ("echo exploited",))

    payload = pickle.dumps(Exploit())
    with pytest.raises(pickle.UnpicklingError) as exc_info:
        _RestrictedUnpickler(io.BytesIO(payload)).load()
    assert "forbidden" in str(exc_info.value).lower()


def test_restricted_unpickler_allows_trained_gesture_model(tmp_path):
    """Ensure that legitimate TrainedGestureModel instances can be saved and loaded."""
    from gestures.teachable import GestureBinding
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    # Fit dummy 1-sample data
    knn.fit([[0.1] * 63], ["test_gesture"])
    model = TrainedGestureModel(
        scope="testapp",
        labels=["test_gesture"],
        knn=knn,
        bindings={"test_gesture": GestureBinding(description="Test action")},
    )
    store = TeachableGestureStore(directory=tmp_path)
    store.save(model)
    loaded = store.load("testapp")
    assert loaded is not None
    assert loaded.scope == "testapp"
    assert "test_gesture" in loaded.labels


# --- Security: Path Traversal Prevention ---

def test_sanitize_filename_strips_traversal():
    """Verify that ../ or ..\\ cannot traverse directories."""
    assert sanitize_filename("../../../secret.txt") == "secret.txt"
    assert sanitize_filename("..\\..\\passwords.txt") == "passwords.txt"
    assert sanitize_filename("my:invalid*name?.txt") == "my_invalid_name_.txt"
    assert sanitize_filename("") == "untitled"


def test_resolve_folder_blocks_directory_traversal(tmp_path, monkeypatch):
    """Ensure resolve_folder does not resolve to an arbitrary directory outside home."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    # A traversal attempt should either be sanitized to the base folder name or fallback safely
    resolved = resolve_folder("../../etc/passwd")
    # Must not traverse out of tmp_path or point to root
    assert not str(resolved).startswith(str(Path("/etc")))


# --- Security: Browser URL Scheme Restriction ---

def test_browser_url_validation():
    """Ensure unsafe schemes (javascript:, file:, data:) are rejected."""
    assert _normalize_url("javascript:alert(1)") is None
    assert _normalize_url("file:///etc/passwd") is None
    assert _normalize_url("data:text/html,<script>alert(1)</script>") is None
    assert _normalize_url("google.com") == "https://google.com"
    assert _normalize_url("https://github.com/test") == "https://github.com/test"

    res = open_url("javascript:alert(1)")
    assert not res.success
    assert "invalid or unsafe" in res.message.lower()


# --- Intent Edge Cases & Slot Validation ---

def test_intent_filename_with_spaces():
    intent = rules.parse("create a python file called my test script in Desktop")
    assert intent.type == IntentType.CREATE_FILE
    assert intent.name == "my test script"
    assert intent.extension == "py"
    assert intent.target_folder == "Desktop"


def test_intent_direct_named_location():
    intent = rules.parse("create a text file called todo in Downloads")
    assert intent.type == IntentType.CREATE_FILE
    assert intent.name == "todo"
    assert intent.extension == "txt"
    assert intent.target_folder == "Downloads"


def test_intent_missing_slots_clarification():
    # File creation without name
    intent = rules.parse("create a python file")
    assert intent.type == IntentType.CREATE_FILE
    assert intent.needs_clarification is True
    assert "name" in intent.missing_slots


def test_intent_delete_missing_name():
    intent = rules.parse("delete the file")
    assert intent.type == IntentType.DELETE_FILE
    assert intent.needs_clarification is True


def test_intent_rename_missing_slots():
    intent = rules.parse("rename the file to final.txt")
    assert intent.type == IntentType.RENAME_FILE
    assert intent.needs_clarification is True
    assert "name" in intent.missing_slots


def test_intent_copy_missing_slots():
    intent = rules.parse("copy the file to Documents")
    assert intent.type == IntentType.COPY_FILE
    assert intent.needs_clarification is True
    assert "name" in intent.missing_slots


# --- Security: File Operations Directory Traversal Prevention ---

def test_create_file_sanitizes_traversal(tmp_path, monkeypatch):
    from executor import files, paths
    paths._NAMED_LOCATIONS_CACHE = None
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "Desktop").mkdir()

    res = files.create_file("../../../secret", "txt")
    assert res.success
    # The created file must be inside Desktop, not outside
    assert res.path.parent == tmp_path / "Desktop"
    assert res.path.name == "secret.txt"


def test_create_folder_sanitizes_traversal(tmp_path, monkeypatch):
    from executor import files, paths
    paths._NAMED_LOCATIONS_CACHE = None
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "Desktop").mkdir()

    res = files.create_folder("../../hacked_folder")
    assert res.success
    assert res.path.parent == tmp_path / "Desktop"
    assert res.path.name == "hacked_folder"


# --- Robustness: Graceful Handling of None / Invalid Inputs ---

def test_executor_none_inputs_handled_cleanly():
    from executor import apps, files, browser

    # apps
    assert not apps.open_app(None).success
    assert not apps.close_app(None).success
    assert apps.find_running_app_display_name(None) == "the app"

    # files
    assert not files.create_file(None, None).success
    assert not files.create_folder(None).success
    assert not files.delete_file(None).success
    assert not files.rename_file(None, None).success
    assert not files.copy_file(None, None).success

    # browser
    assert not browser.open_url(None).success
    assert not browser.web_search(None).success

