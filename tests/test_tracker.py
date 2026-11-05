"""
Tests for gestures/tracker.py - the camera -> hand-landmark pipeline.

Neither `mediapipe` nor `opencv-python` are installed in this sandbox
(mediapipe is a large native-code dependency, and headless CI environments
frequently don't have it either) so these tests fake both at the
`sys.modules` level with the minimal surface HandTracker.process() actually
touches: `mp.solutions.hands.Hands.process()`'s legacy result shape, and
`cv2.cvtColor()`. This exercises the real logic under test - converting a
MediaPipe result object into plain HandFrame tuples - without needing a
real camera, a real model file, or the actual native libraries installed.

Run with: pytest tests/test_tracker.py -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ---------------------------------------------------------------------------
# Fakes for mediapipe's legacy `solutions.hands` API
# ---------------------------------------------------------------------------

class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeHandLandmarks:
    """Stands in for mediapipe's `NormalizedLandmarkList` - only the
    `.landmark` attribute (an iterable of 21 x/y points) is used by
    HandTracker.process()."""
    def __init__(self, points):
        self.landmark = [_FakeLandmark(x, y) for (x, y) in points]


class _FakeClassification:
    def __init__(self, label):
        self.classification = [types.SimpleNamespace(label=label)]


class _FakeResults:
    def __init__(self, hands: list[list[tuple[float, float]]], handedness: list[str]):
        self.multi_hand_landmarks = [_FakeHandLandmarks(pts) for pts in hands] if hands else None
        self.multi_handedness = [_FakeClassification(h) for h in handedness] if hands else None


def _make_fake_mediapipe(results_to_return):
    """Builds a fake `mediapipe` module exposing just enough of
    `solutions.hands` for HandTracker.__init__ + .process() to run against,
    with `Hands().process()` pre-programmed to return `results_to_return`."""
    fake_hands_instance = MagicMock()
    fake_hands_instance.process.return_value = results_to_return

    fake_hands_class = MagicMock(return_value=fake_hands_instance)

    fake_mp = types.SimpleNamespace()
    fake_mp.solutions = types.SimpleNamespace(
        hands=types.SimpleNamespace(Hands=fake_hands_class, HAND_CONNECTIONS=object()),
        drawing_utils=MagicMock(),
    )
    return fake_mp, fake_hands_instance


def _install_fakes(monkeypatch, results_to_return):
    fake_mp, fake_hands_instance = _make_fake_mediapipe(results_to_return)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    fake_cv2 = types.SimpleNamespace(
        cvtColor=lambda frame, code: frame,  # identity - color conversion isn't what's under test
        COLOR_BGR2RGB=0,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return fake_hands_instance


@pytest.fixture(autouse=True)
def _isolate_module_cache(monkeypatch):
    """gestures.tracker imports mediapipe/cv2 lazily inside methods, but
    make sure a real import never sneaks in from a previous test's cached
    module state."""
    monkeypatch.delitem(sys.modules, "gestures.tracker", raising=False)
    yield
    monkeypatch.delitem(sys.modules, "gestures.tracker", raising=False)


# ---------------------------------------------------------------------------
# HandTracker.process()
# ---------------------------------------------------------------------------

def test_process_returns_empty_list_when_no_hand_detected(monkeypatch):
    _install_fakes(monkeypatch, _FakeResults(hands=[], handedness=[]))
    from gestures.tracker import HandTracker

    tracker = HandTracker()
    result = tracker.process(frame_bgr="fake_frame_object")

    assert result == []


def test_process_converts_one_hand_into_a_hand_frame(monkeypatch):
    points = [(0.1 * i, 0.2 * i) for i in range(21)]  # 21 landmarks, MediaPipe's fixed count
    _install_fakes(monkeypatch, _FakeResults(hands=[points], handedness=["Right"]))
    from gestures.tracker import HandTracker

    tracker = HandTracker()
    result = tracker.process(frame_bgr="fake_frame_object")

    assert len(result) == 1
    assert result[0].handedness == "Right"
    assert len(result[0].landmarks) == 21
    assert result[0].landmarks[0] == pytest.approx((0.0, 0.0))
    assert result[0].landmarks[5] == pytest.approx((0.5, 1.0))


def test_process_converts_landmarks_to_plain_tuples_not_mediapipe_objects(monkeypatch):
    """Downstream code (recognizer.py, teachable.py) works with plain
    (x, y) tuples and must never need to import mediapipe itself - this is
    the whole reason tracker.py exists as a boundary."""
    points = [(0.5, 0.5)] * 21
    _install_fakes(monkeypatch, _FakeResults(hands=[points], handedness=["Left"]))
    from gestures.tracker import HandTracker

    tracker = HandTracker()
    result = tracker.process(frame_bgr="fake_frame_object")

    assert isinstance(result[0].landmarks, list)
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in result[0].landmarks)


def test_process_respects_max_hands_limit_from_the_underlying_result(monkeypatch):
    """max_hands is passed through to MediaPipe itself (which does the
    actual limiting) - this just confirms tracker.py doesn't drop or
    duplicate hands MediaPipe already returned."""
    hand_a = [(0.1, 0.1)] * 21
    hand_b = [(0.9, 0.9)] * 21
    _install_fakes(monkeypatch, _FakeResults(hands=[hand_a, hand_b], handedness=["Left", "Right"]))
    from gestures.tracker import HandTracker

    tracker = HandTracker(max_hands=2)
    result = tracker.process(frame_bgr="fake_frame_object")

    assert len(result) == 2
    assert result[0].handedness == "Left"
    assert result[1].handedness == "Right"


def test_process_defaults_handedness_when_classification_list_is_shorter_than_hands(monkeypatch):
    """Defensive case: if MediaPipe ever returns landmarks without a
    matching handedness entry, this must not crash - it should default
    rather than index out of range."""
    points = [(0.3, 0.3)] * 21
    results = _FakeResults(hands=[points], handedness=[])
    results.multi_handedness = []  # explicitly empty, not None
    _install_fakes(monkeypatch, results)
    from gestures.tracker import HandTracker

    tracker = HandTracker()
    result = tracker.process(frame_bgr="fake_frame_object")

    assert len(result) == 1
    assert result[0].handedness == "Right"  # documented default


def test_init_raises_a_clear_error_when_mediapipe_is_not_installed(monkeypatch):
    monkeypatch.delitem(sys.modules, "mediapipe", raising=False)
    monkeypatch.delitem(sys.modules, "gestures.tracker", raising=False)

    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "mediapipe":
            raise ImportError("No module named 'mediapipe'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    from gestures.tracker import HandTracker

    with pytest.raises(ImportError, match="mediapipe is required"):
        HandTracker()
