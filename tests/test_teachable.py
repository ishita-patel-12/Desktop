"""
Tests for gestures/teachable.py (Phase 3 - teachable custom gestures).
Uses synthetic hand-landmark coordinates, same approach as
tests/test_gestures.py - no camera, no mediapipe install required.

Run with: pytest tests/test_teachable.py -v
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gestures.teachable import (
    CustomGestureClassifier,
    GLOBAL_SCOPE,
    GestureBinding,
    GestureSample,
    MIN_SAMPLES_PER_GESTURE,
    TeachableGestureStore,
    featurize,
    train_classifier,
)

WRIST = (0.5, 0.9)
MIDDLE_MCP_PT = (0.5, 0.6)


def _hand_with_shape(tip_offsets: dict, jitter: float = 0.0, seed: int = 0) -> list:
    """Builds a full 21-point landmark list from a small dict of
    {landmark_index: (x, y)} overrides on top of a relaxed default hand,
    optionally jittered a little to simulate natural pose variation across
    captured samples (real samples are never pixel-identical frame to
    frame)."""
    rng = random.Random(seed)
    points = [(0.5, 0.55)] * 21
    points[0] = WRIST
    points[9] = MIDDLE_MCP_PT  # MIDDLE_MCP, used as the normalization axis

    for idx, (x, y) in tip_offsets.items():
        if jitter:
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        points[idx] = (x, y)
    return points


def _shape_a(jitter=0.0, seed=0):
    # "thumbs up"-ish: thumb tip far above the wrist, everything else tucked in
    return _hand_with_shape({4: (0.5, 0.15), 8: (0.48, 0.58), 12: (0.5, 0.58),
                              16: (0.52, 0.58), 20: (0.54, 0.58)}, jitter, seed)


def _shape_b(jitter=0.0, seed=0):
    # "peace sign"-ish: index+middle far out, ring/pinky tucked
    return _hand_with_shape({4: (0.4, 0.5), 8: (0.35, 0.2), 12: (0.55, 0.2),
                              16: (0.52, 0.58), 20: (0.54, 0.58)}, jitter, seed)


def _samples(shape_fn, label, n=MIN_SAMPLES_PER_GESTURE + 4) -> list[GestureSample]:
    return [GestureSample(label=label, landmarks=shape_fn(jitter=0.01, seed=i)) for i in range(n)]


# ---------------------------------------------------------------------------
# featurize()
# ---------------------------------------------------------------------------

def test_featurize_needs_all_21_landmarks():
    with pytest.raises(ValueError):
        featurize([(0.0, 0.0)] * 5)


def test_featurize_is_translation_invariant():
    """Shifting the whole hand in frame shouldn't change the feature
    vector - only the shape relative to the wrist should matter."""
    hand = _shape_a()
    shifted = [(x + 0.2, y - 0.1) for (x, y) in hand]
    f1 = featurize(hand)
    f2 = featurize(shifted)
    for a, b in zip(f1, f2):
        assert a == pytest.approx(b, abs=1e-6)


def test_featurize_is_scale_invariant():
    """A hand held closer/farther from the camera (uniformly scaled)
    should produce (nearly) the same normalized feature vector."""
    hand = _shape_a()
    wrist = hand[0]
    scaled = [(wrist[0] + (x - wrist[0]) * 2.0, wrist[1] + (y - wrist[1]) * 2.0) for (x, y) in hand]
    f1 = featurize(hand)
    f2 = featurize(scaled)
    for a, b in zip(f1, f2):
        assert a == pytest.approx(b, abs=1e-6)


def test_featurize_distinguishes_different_shapes():
    fa = featurize(_shape_a())
    fb = featurize(_shape_b())
    dist = sum((a - b) ** 2 for a, b in zip(fa, fb)) ** 0.5
    assert dist > 0.5  # clearly different poses should be far apart


# ---------------------------------------------------------------------------
# train_classifier()
# ---------------------------------------------------------------------------

def test_train_succeeds_with_a_single_gesture():
    """A single taught gesture is now a valid, saveable model - it works
    as a yes/no distance-based detector for that one pose (see
    train_classifier()'s docstring for why >= 2 labels is no longer
    required). This was the reported bug: teaching just one gesture used
    to discard everything and save nothing."""
    samples = _samples(_shape_a, "one_gesture")
    model = train_classifier(samples, scope=GLOBAL_SCOPE)
    assert model.labels == ["one_gesture"]
    assert model.max_distance > 0


def test_single_gesture_model_accepts_the_trained_pose_and_rejects_another():
    samples = _samples(_shape_a, "shape_a", n=15)
    model = train_classifier(samples, scope=GLOBAL_SCOPE)
    clf = CustomGestureClassifier(model, hold_seconds=0.1, cooldown_seconds=0.1)

    clf.update(_shape_a(jitter=0.005, seed=999), timestamp=0.0)
    accepted = clf.update(_shape_a(jitter=0.005, seed=999), timestamp=0.2)
    assert accepted is not None
    assert accepted[0] == "shape_a"

    clf2 = CustomGestureClassifier(model, hold_seconds=0.1, cooldown_seconds=0.1)
    clf2.update(_shape_b(jitter=0.005, seed=999), timestamp=0.0)
    rejected = clf2.update(_shape_b(jitter=0.005, seed=999), timestamp=0.2)
    assert rejected is None


def test_train_requires_at_least_one_sample():
    with pytest.raises(ValueError):
        train_classifier([], scope=GLOBAL_SCOPE)


def test_train_requires_minimum_samples_per_label():
    samples = _samples(_shape_a, "a", n=3) + _samples(_shape_b, "b")
    with pytest.raises(ValueError):
        train_classifier(samples, scope=GLOBAL_SCOPE)


def test_train_succeeds_with_two_well_separated_gestures():
    samples = _samples(_shape_a, "shape_a") + _samples(_shape_b, "shape_b")
    model = train_classifier(samples, scope=GLOBAL_SCOPE)
    assert set(model.labels) == {"shape_a", "shape_b"}
    assert model.samples_per_label["shape_a"] >= MIN_SAMPLES_PER_GESTURE
    assert model.max_distance > 0


def test_trained_model_correctly_classifies_a_held_out_sample():
    train_samples = _samples(_shape_a, "shape_a", n=15) + _samples(_shape_b, "shape_b", n=15)
    model = train_classifier(train_samples, scope=GLOBAL_SCOPE)

    held_out_a = featurize(_shape_a(jitter=0.005, seed=999))
    held_out_b = featurize(_shape_b(jitter=0.005, seed=999))
    assert model.knn.predict([held_out_a])[0] == "shape_a"
    assert model.knn.predict([held_out_b])[0] == "shape_b"


# ---------------------------------------------------------------------------
# TeachableGestureStore (save/load round trip)
# ---------------------------------------------------------------------------

def test_store_round_trip(tmp_path):
    samples = _samples(_shape_a, "a") + _samples(_shape_b, "b")
    model = train_classifier(
        samples, scope="notepad",
        bindings={
            "a": GestureBinding(description="Save File", action={"action": "keys", "keys": ["ctrl", "s"]}),
            "b": GestureBinding(description="Recognized only, no action yet"),
        },
    )

    store = TeachableGestureStore(directory=tmp_path)
    path = store.save(model)
    assert path.exists()

    loaded = store.load("notepad")
    assert loaded is not None
    assert loaded.labels == model.labels
    assert loaded.bindings["a"].description == "Save File"
    assert loaded.bindings["a"].action == {"action": "keys", "keys": ["ctrl", "s"]}
    assert loaded.bindings["b"].action is None


def test_store_returns_none_for_missing_scope(tmp_path):
    store = TeachableGestureStore(directory=tmp_path)
    assert store.load("nonexistent_scope") is None


def test_store_list_scopes(tmp_path):
    samples = _samples(_shape_a, "a") + _samples(_shape_b, "b")
    store = TeachableGestureStore(directory=tmp_path)
    store.save(train_classifier(samples, scope="vscode"))
    store.save(train_classifier(samples, scope=GLOBAL_SCOPE))
    assert store.list_scopes() == sorted(["vscode", GLOBAL_SCOPE])


def test_store_sanitizes_scope_names_for_the_filesystem(tmp_path):
    samples = _samples(_shape_a, "a") + _samples(_shape_b, "b")
    store = TeachableGestureStore(directory=tmp_path)
    store.save(train_classifier(samples, scope="Some App / Weird:Name"))
    # Should not have created a path traversal or crashed on odd characters.
    assert store.load("Some App / Weird:Name") is not None


def test_train_gives_every_label_a_default_binding_when_none_is_supplied():
    """A gesture taught without an explicit description still needs a
    non-crashing binding at runtime - it should describe itself rather
    than KeyError."""
    samples = _samples(_shape_a, "a") + _samples(_shape_b, "b")
    model = train_classifier(samples, scope=GLOBAL_SCOPE)  # no bindings= at all
    assert "a" in model.bindings
    assert "unlabeled" in model.bindings["a"].description
    assert model.bindings["a"].action is None


def test_describe_all_lists_every_scope_with_its_bindings(tmp_path):
    samples = _samples(_shape_a, "a") + _samples(_shape_b, "b")
    store = TeachableGestureStore(directory=tmp_path)
    store.save(train_classifier(
        samples, scope="vscode",
        bindings={"a": GestureBinding(description="Save File", action={"action": "keys", "keys": ["ctrl", "s"]})},
    ))
    store.save(train_classifier(
        samples, scope=GLOBAL_SCOPE,
        bindings={"a": GestureBinding(description="Scroll Up", action={"action": "scroll", "direction": "up"})},
    ))

    all_models = store.describe_all()
    assert set(all_models.keys()) == {"vscode", GLOBAL_SCOPE}
    assert all_models["vscode"].bindings["a"].description == "Save File"
    assert all_models[GLOBAL_SCOPE].bindings["a"].description == "Scroll Up"


# ---------------------------------------------------------------------------
# CustomGestureClassifier (runtime hold + cooldown + rejection)
# ---------------------------------------------------------------------------

def _model():
    samples = _samples(_shape_a, "shape_a", n=15) + _samples(_shape_b, "shape_b", n=15)
    return train_classifier(
        samples, scope=GLOBAL_SCOPE,
        bindings={
            "shape_a": GestureBinding(description="Play/Pause Media", action={"action": "keys", "keys": ["space"]}),
            "shape_b": GestureBinding(description="Recognized only, no action yet"),
        },
    )


def test_classifier_does_not_fire_immediately():
    clf = CustomGestureClassifier(_model(), hold_seconds=0.5, cooldown_seconds=0.5)
    result = clf.update(_shape_a(), timestamp=0.0)
    assert result is None


def test_classifier_fires_after_hold_and_returns_bound_action():
    clf = CustomGestureClassifier(_model(), hold_seconds=0.5, cooldown_seconds=0.5)
    clf.update(_shape_a(), timestamp=0.0)
    result = clf.update(_shape_a(), timestamp=0.6)
    assert result == ("shape_a", GestureBinding(description="Play/Pause Media", action={"action": "keys", "keys": ["space"]}))


def test_classifier_does_not_refire_while_still_held():
    clf = CustomGestureClassifier(_model(), hold_seconds=0.3, cooldown_seconds=0.5)
    clf.update(_shape_a(), timestamp=0.0)
    first = clf.update(_shape_a(), timestamp=0.4)
    second = clf.update(_shape_a(), timestamp=0.5)
    assert first is not None
    assert second is None


def test_switching_pose_resets_the_hold_timer():
    clf = CustomGestureClassifier(_model(), hold_seconds=0.5, cooldown_seconds=0.1)
    clf.update(_shape_a(), timestamp=0.0)
    clf.update(_shape_b(), timestamp=0.1)  # switched pose - resets timer
    result = clf.update(_shape_a(), timestamp=0.3)  # only 0.2s into the new 'a' hold
    assert result is None


def test_untrained_pose_is_rejected_not_forced_into_nearest_label():
    """A hand shape far from both trained gestures should return None
    (rejected via max_distance), not be forced into whichever trained
    label happens to be nearest."""
    clf = CustomGestureClassifier(_model(), hold_seconds=0.1, cooldown_seconds=0.1)
    # Open palm: nothing like the trained thumbs-up/peace-sign shapes.
    open_palm = _hand_with_shape({8: (0.3, 0.1), 12: (0.5, 0.05), 16: (0.6, 0.1), 20: (0.68, 0.2)})
    clf.update(open_palm, timestamp=0.0)
    result = clf.update(open_palm, timestamp=0.2)
    assert result is None
