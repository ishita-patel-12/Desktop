"""
These tests exercise the executor against a temp "fake home" directory
(monkeypatched via Path.home) so they never touch your real Desktop/Documents.
Run with: pytest tests/test_executor.py -v
"""

import sys
import shutil
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at a scratch directory and pre-create Desktop/Documents,
    matching what settings.yaml expects."""
    (tmp_path / "Desktop").mkdir()
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Downloads").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Path.expanduser() reads HOME on POSIX and USERPROFILE on Windows,
    # so both need to be patched or "~/Desktop" in settings.yaml won't resolve
    # into the fake sandbox.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    # executor.paths caches named locations on first call - clear the cache
    # so it re-reads against the fake home for each test.
    import executor.paths as paths_mod
    paths_mod._NAMED_LOCATIONS_CACHE = None

    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_create_file_default_location(fake_home):
    from executor import files
    result = files.create_file("calculator", "py")
    assert result.success
    assert result.path.exists()
    assert result.path.parent == fake_home / "Desktop"
    assert result.path.name == "calculator.py"
    assert "def main" in result.path.read_text()


def test_create_file_in_named_folder(fake_home):
    from executor import files
    result = files.create_file("notes", "txt", target_folder_name="Documents")
    assert result.success
    assert result.path.parent == fake_home / "Documents"


def test_create_file_never_overwrites(fake_home):
    from executor import files
    r1 = files.create_file("test", "txt")
    r2 = files.create_file("test", "txt")
    assert r1.path != r2.path
    assert r1.path.exists() and r2.path.exists()


def test_create_folder(fake_home):
    from executor import files
    result = files.create_folder("Projects")
    assert result.success
    assert result.path.is_dir()
    assert result.path == fake_home / "Desktop" / "Projects"


def test_create_folder_duplicate_fails_gracefully(fake_home):
    from executor import files
    files.create_folder("Projects")
    result = files.create_folder("Projects")
    assert not result.success


def test_delete_missing_file_reports_failure(fake_home):
    from executor import files
    result = files.delete_file("does_not_exist.txt")
    assert not result.success


def test_delete_uses_trash_not_permanent(fake_home):
    from executor import files
    created = files.create_file("throwaway", "txt")
    result = files.delete_file("throwaway.txt")
    assert result.success
    # File should be gone from its original location either way (trash or unlink)
    assert not created.path.exists()


def test_rename_file(fake_home):
    from executor import files
    created = files.create_file("draft", "txt")
    result = files.rename_file("draft.txt", "final.txt")
    assert result.success
    assert result.path.name == "final.txt"
    assert not created.path.exists()


def test_resolve_folder_falls_back_to_default_when_not_found(fake_home):
    from executor.paths import resolve_folder, default_create_location
    resolved = resolve_folder("SomeFolderThatDoesNotExist")
    assert resolved == default_create_location()


def test_resolve_folder_finds_nested_folder(fake_home):
    from executor.paths import resolve_folder
    nested = fake_home / "Documents" / "Projects"
    nested.mkdir()
    resolved = resolve_folder("Projects")
    assert resolved == nested


# --- Regression tests for reported bugs: "can't delete a folder I created" /
# "can't create a java file inside that folder" ---

def test_delete_folder_created_by_assistant(fake_home):
    """Bug: intent parser was capturing name='Projects folder' (trailing
    filler word attached) instead of 'Projects', so nothing matched on disk."""
    from intent import rules
    from executor.dispatch import dispatch

    create_intent = rules.parse("create a folder called Projects", [])
    dispatch(create_intent)
    assert (fake_home / "Desktop" / "Projects").is_dir()

    delete_intent = rules.parse("delete the Projects folder", [])
    assert delete_intent.name == "Projects"  # not "Projects folder"
    result = dispatch(delete_intent, confirmed=True)
    assert result.success
    assert not (fake_home / "Desktop" / "Projects").exists()


def test_create_java_file_inside_folder(fake_home):
    """Bug: "create a java file called X in the Y folder" was misclassified
    as create_folder because the CREATE_FOLDER pattern matched on the word
    'folder' appearing anywhere in the sentence, including in the location
    clause. Also: 'java' wasn't in the recognized extension list at all."""
    from intent import rules
    from executor.dispatch import dispatch

    (fake_home / "Desktop" / "Projects").mkdir()
    intent = rules.parse("create a java file called Test in the Projects folder", [])
    assert intent.type.value == "create_file"
    assert intent.extension == "java"
    assert intent.target_folder == "Projects"

    result = dispatch(intent)
    assert result.success
    assert (fake_home / "Desktop" / "Projects" / "Test.java").exists()


def test_delete_permanent_fallback_handles_directories(fake_home):
    """Bug: the no-send2trash fallback called .unlink() unconditionally,
    which raises IsADirectoryError for folders."""
    from executor import files
    (fake_home / "Desktop" / "SomeFolder").mkdir()
    result = files.delete_file("SomeFolder")
    assert result.success
    assert not (fake_home / "Desktop" / "SomeFolder").exists()


def test_delete_exact_match_does_not_catch_prefix_siblings(fake_home):
    """Deleting 'Test' should never also catch 'TestBackup'."""
    from executor import files
    (fake_home / "Desktop" / "Test").mkdir()
    (fake_home / "Desktop" / "TestBackup").mkdir()
    result = files.delete_file("Test")
    assert result.success
    assert not (fake_home / "Desktop" / "Test").exists()
    assert (fake_home / "Desktop" / "TestBackup").exists()