"""
Phase 3 — Teachable custom gestures (few-shot, on-device, no deep learning).

This is the feature the README flagged as "designed, not yet implemented".
It follows the design written up there:

  1. Capture   - record N samples of a new pose's 21 hand landmarks via
                 gestures/tracker.py while the user holds it in front of
                 the camera (see `run_teach_mode()` at the bottom).
  2. Featurize - convert each frame's raw (x, y) landmarks into a
                 representation that's stable across hand size, distance
                 from the camera, and rotation (`featurize()`).
  3. Train     - fit a small k-NN classifier per gesture set
                 (`train_classifier()`). k-NN trains in milliseconds on
                 20-30 samples and needs no GPU/deep learning, matching the
                 project's local-first, no-deep-learning goal. A single
                 taught gesture is enough to save and use (it works as a
                 yes/no distance-based detector); a second, different
                 gesture is optional but can sharpen accuracy.
  4. Store     - saved per scope (usually one app) as
                 `gestures/custom/<scope>.pkl`, alongside which action each
                 label should trigger, so a gesture can mean nothing in
                 general but do something specific only while a particular
                 app has focus (`TeachableGestureStore`).
  5. Predict   - `CustomGestureClassifier` loads a saved model and turns
                 live landmarks into (label, action) at runtime, with a
                 distance-based rejection threshold so an untrained pose
                 doesn't get silently misclassified as the nearest trained
                 one, and its own hold+cooldown timing so it behaves like
                 the built-in gestures in recognizer.py.

Zero dependency on mediapipe/opencv in the featurize/train/predict path -
same design principle as recognizer.py - so all of this is unit-testable
on synthetic landmark data with no camera. Only `run_teach_mode()` (the
interactive capture UI) touches the camera.
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from gestures.recognizer import WRIST, MIDDLE_MCP

Point = tuple[float, float]

CUSTOM_DIR = Path(__file__).parent / "custom"
GLOBAL_SCOPE = "global"  # applies regardless of which app has focus

MIN_SAMPLES_PER_GESTURE = 8       # below this a k-NN split isn't meaningful
DEFAULT_SAMPLES_PER_GESTURE = 25  # matches the README's "~20-30 samples"
DEFAULT_K = 3
# Normalized-feature-space distance beyond which a live pose is treated as
# "doesn't match any trained gesture" rather than forced into the nearest
# one. Features are scale/rotation-normalized (see featurize()) so this is
# a fairly stable default across cameras/hand sizes; run_teach_mode() can
# tighten it per-model from the captured samples' own spread.
DEFAULT_MAX_DISTANCE = 0.55
DEFAULT_HOLD_SECONDS = 0.5   # a custom gesture must be held briefly, like fist_hold
DEFAULT_COOLDOWN_SECONDS = 0.8


# ---------------------------------------------------------------------------
# 2. Featurize
# ---------------------------------------------------------------------------

def featurize(landmarks: list[Point]) -> list[float]:
    """21 raw (x, y) landmarks -> a 63-dim feature vector: each point's
    distance from the wrist (normalized by hand span) plus sin/cos of its
    angle relative to the wrist->middle-MCP axis (the hand's own "up").

    This is the same idea recognizer.py's classify_static_pose already
    relies on (distance-from-wrist beats raw coordinates for robustness to
    hand size/camera distance) generalized into a numeric vector a
    classifier can fit, instead of hand-tuned thresholds for a fixed set
    of poses. Using sin/cos of the angle (rather than the raw angle) avoids
    the -pi/+pi wraparound discontinuity that would otherwise make two
    nearly-identical poses look far apart to a distance-based classifier.
    """
    if len(landmarks) < 21:
        raise ValueError("featurize() needs all 21 hand landmarks")

    wrist = landmarks[WRIST]
    mcp = landmarks[MIDDLE_MCP]
    span = math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1]) or 1e-6
    axis_angle = math.atan2(mcp[1] - wrist[1], mcp[0] - wrist[0])

    features: list[float] = []
    for (x, y) in landmarks:
        dx, dy = x - wrist[0], y - wrist[1]
        dist = math.hypot(dx, dy) / span
        angle = math.atan2(dy, dx) - axis_angle
        features.append(dist)
        features.append(math.sin(angle))
        features.append(math.cos(angle))
    return features


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

@dataclass
class GestureSample:
    label: str
    landmarks: list[Point]


@dataclass
class GestureBinding:
    """What a taught gesture actually *means* for one app: a human-readable
    function name/description (e.g. "Save File", "Scroll Up", "Mute Mic")
    plus the machine-executable action, if any. Keeping the description
    alongside the action - rather than just an anonymous action dict - is
    what makes `--list-gestures` (see assistant.py) and the runtime log
    lines legible: "peace_sign -> Scroll Up" means something to a human,
    {"action": "scroll", "direction": "up"} on its own doesn't."""
    description: str
    action: Optional[dict] = None


@dataclass
class TrainedGestureModel:
    """Everything needed to classify at runtime AND to know what a match
    should *do* - the binding lives with the model so a taught gesture is
    immediately useful and self-documenting, not just recognized."""
    scope: str
    labels: list[str]                        # unique gesture names, fit order
    knn: object                              # fitted sklearn KNeighborsClassifier
    bindings: dict[str, GestureBinding]      # label -> what it means/does in this scope
    max_distance: float = DEFAULT_MAX_DISTANCE
    k: int = DEFAULT_K
    created_at: float = field(default_factory=time.time)
    samples_per_label: dict[str, int] = field(default_factory=dict)


def train_classifier(
    samples: list[GestureSample],
    scope: str,
    bindings: Optional[dict[str, GestureBinding]] = None,
    k: int = DEFAULT_K,
    max_distance: Optional[float] = None,
) -> TrainedGestureModel:
    """Fits a k-NN classifier over featurized samples. Requires at least
    one label with >= MIN_SAMPLES_PER_GESTURE samples, so the model has
    something meaningful to generalize from.

    A single taught gesture is a valid, useful model on its own: it works
    as a pure yes/no distance-based detector - "does this pose look close
    enough to what was taught" - via `max_distance` (see
    `_estimate_max_distance`, which only ever needs same-label neighbor
    distances, so it works fine with one label). Teaching a second,
    different gesture in the same session is still worth encouraging (it
    gives the classifier a contrasting shape to discriminate against,
    which can improve real-world accuracy) but it is no longer required
    to save anything - the old "need >= 2 distinct gestures or nothing
    gets saved" behavior threw away a real, working single-gesture
    session for no functional reason.
    """
    from sklearn.neighbors import KNeighborsClassifier

    by_label: dict[str, list[GestureSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    if len(by_label) < 1:
        raise ValueError("No samples to train on.")
    for label, group in by_label.items():
        if len(group) < MIN_SAMPLES_PER_GESTURE:
            raise ValueError(
                f"Gesture '{label}' only has {len(group)} sample(s); "
                f"need at least {MIN_SAMPLES_PER_GESTURE}."
            )

    X = [featurize(s.landmarks) for s in samples]
    y = [s.label for s in samples]

    effective_k = min(k, min(len(g) for g in by_label.values()))
    knn = KNeighborsClassifier(n_neighbors=effective_k)
    knn.fit(X, y)

    resolved_threshold = max_distance if max_distance is not None else _estimate_max_distance(X, y, knn)

    resolved_bindings = dict(bindings) if bindings else {}
    for label in by_label:
        # Every label gets a binding even if the caller didn't provide
        # one, so `.bindings[label]` is never a KeyError at runtime - an
        # untitled gesture just describes itself as "(unlabeled)" rather
        # than crashing the dispatch path.
        resolved_bindings.setdefault(label, GestureBinding(description=f"{label} (unlabeled)", action=None))

    return TrainedGestureModel(
        scope=scope,
        labels=sorted(by_label.keys()),
        knn=knn,
        bindings=resolved_bindings,
        max_distance=resolved_threshold,
        k=effective_k,
        samples_per_label={label: len(group) for label, group in by_label.items()},
    )


def _estimate_max_distance(X: list[list[float]], y: list[str], knn) -> float:
    """Leave-one-out-ish estimate: for each training point, the distance to
    its nearest same-label neighbor (excluding itself). The rejection
    threshold is set a bit above the worst of those, so genuine repeats of
    a trained pose comfortably pass while poses far from every class get
    rejected as "no match" rather than forced into the closest label.
    Falls back to DEFAULT_MAX_DISTANCE if the math is degenerate (e.g. only
    one sample per label, or scikit-learn isn't behaving as expected).
    """
    try:
        distances, indices = knn.kneighbors(X, n_neighbors=min(len(X), 5))
        worst_same_label = 0.0
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            for d, j in zip(dists, idxs):
                if j != i and y[j] == y[i]:
                    worst_same_label = max(worst_same_label, d)
                    break
        if worst_same_label <= 0:
            return DEFAULT_MAX_DISTANCE
        return worst_same_label * 1.6 + 0.05
    except Exception as e:
        logger.warning(f"Could not estimate rejection threshold, using default: {e}")
        return DEFAULT_MAX_DISTANCE


# ---------------------------------------------------------------------------
# 4. Store
# ---------------------------------------------------------------------------

class _RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows safe classes required for TrainedGestureModel.
    Prevents arbitrary code execution when loading untrusted model files.
    """
    _SAFE_BUILTINS = {
        "dict", "list", "set", "tuple", "float", "int", "str", "bool", "bytes", "getattr"
    }

    def find_class(self, module: str, name: str):
        if module == "builtins" and name in self._SAFE_BUILTINS:
            return getattr(__import__("builtins"), name)
        if module in ("gestures.teachable", "teachable") or module.endswith(".teachable"):
            if name in {"TrainedGestureModel", "GestureBinding", "GestureSample"}:
                import gestures.teachable as t_mod
                return getattr(t_mod, name)
        if (module.startswith("sklearn.") or module.startswith("numpy.")
                or module == "numpy" or module == "scipy.sparse"):
            mod = __import__(module, fromlist=[name])
            return getattr(mod, name)
        if module == "_codecs" and name == "encode":
            import _codecs
            return _codecs.encode
        raise pickle.UnpicklingError(f"Security: forbidden class in model file: {module}.{name}")


class TeachableGestureStore:
    """Loads/saves TrainedGestureModel objects as
    gestures/custom/<scope>.pkl. `scope` is normally an app key from
    config/apps.yaml (e.g. "vscode"), or GLOBAL_SCOPE for a gesture that
    should work regardless of which app has focus."""

    def __init__(self, directory: Path = CUSTOM_DIR):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, scope: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in scope.lower()) or GLOBAL_SCOPE
        return self.directory / f"{safe}.pkl"

    def save(self, model: TrainedGestureModel) -> Path:
        path = self._path(model.scope)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(
            f"Saved teachable-gesture model for scope={model.scope!r}: "
            f"{model.labels} -> {path}"
        )
        return path

    def load(self, scope: str) -> Optional[TrainedGestureModel]:
        path = self._path(scope)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                model = _RestrictedUnpickler(f).load()
            if not isinstance(model, TrainedGestureModel):
                logger.warning(f"File {path} did not deserialize into a TrainedGestureModel.")
                return None
            return model
        except Exception as e:
            logger.warning(f"Could not load teachable-gesture model {path}: {e}")
            return None

    def delete(self, scope: str) -> bool:
        path = self._path(scope)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_scopes(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.pkl"))

    def describe_all(self) -> dict[str, TrainedGestureModel]:
        """Loads every saved model, keyed by scope - what `--list-gestures`
        (assistant.py) iterates over to print a full picture of what's
        taught across every app, not just one scope at a time."""
        result = {}
        for scope in self.list_scopes():
            model = self.load(scope)
            if model:
                result[scope] = model
        return result


# ---------------------------------------------------------------------------
# 5. Predict (runtime)
# ---------------------------------------------------------------------------

class CustomGestureClassifier:
    """Runtime wrapper around a TrainedGestureModel with the same
    hold-then-fire + cooldown shape as GestureRecognizer's fist_hold, so a
    custom gesture behaves consistently with the built-ins: it must be held
    briefly (avoids one noisy frame triggering an action) and then won't
    re-fire every frame while still held.
    """

    def __init__(self, model: TrainedGestureModel,
                 hold_seconds: float = DEFAULT_HOLD_SECONDS,
                 cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS):
        self.model = model
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None
        self._fired = False
        self._last_event_time = float("-inf")  # no previous fire yet, so cooldown never blocks the first one

    def _raw_predict(self, landmarks: list[Point]) -> Optional[str]:
        try:
            features = featurize(landmarks)
        except ValueError:
            return None
        distances, indices = self.model.knn.kneighbors([features], n_neighbors=1)
        nearest_distance = distances[0][0]
        if nearest_distance > self.model.max_distance:
            return None
        return self.model.knn.predict([features])[0]

    def update(self, landmarks: list[Point], timestamp: Optional[float] = None
               ) -> Optional[tuple[str, GestureBinding]]:
        """Feed one frame's landmarks in. Returns (label, binding) once,
        when a trained pose has been held for `hold_seconds` and isn't in
        cooldown from a previous fire - or None most frames. `binding`
        carries both the human-readable description of what this gesture
        means in its scope and the action to actually run, if any."""
        t = timestamp if timestamp is not None else time.monotonic()
        label = self._raw_predict(landmarks)

        if label is None:
            self._candidate = None
            self._candidate_since = None
            self._fired = False
            return None

        if label != self._candidate:
            self._candidate = label
            self._candidate_since = t
            self._fired = False
            return None

        held_long_enough = (t - self._candidate_since) >= self.hold_seconds
        in_cooldown = (t - self._last_event_time) < self.cooldown_seconds

        if held_long_enough and not self._fired and not in_cooldown:
            self._fired = True
            self._last_event_time = t
            binding = self.model.bindings.get(label) or GestureBinding(description=label, action=None)
            return label, binding
        return None


# ---------------------------------------------------------------------------
# 1. Capture (interactive, camera-driven - not covered by unit tests)
# ---------------------------------------------------------------------------

def print_all_taught_gestures() -> None:
    """`python assistant.py --list-gestures`

    Prints every taught gesture across every app scope in one place:
    what it's called, what it DOES (the human-readable function label a
    recruiter/teammate/future-you can actually read), and whether it's
    bound to a real action yet. Doesn't touch the camera - pure disk
    read - so it's instant and safe to run anytime, including to sanity-
    check what a `--teach-gesture` session actually saved.
    """
    store = TeachableGestureStore()
    all_models = store.describe_all()

    if not all_models:
        print("No custom gestures taught yet. Run `python assistant.py --teach-gesture` to add one.")
        return

    print("\n--- Taught gestures ---")
    for scope in sorted(all_models):
        model = all_models[scope]
        scope_label = "global (all apps)" if scope == GLOBAL_SCOPE else scope
        print(f"\n[{scope_label}]")
        for label in model.labels:
            binding = model.bindings.get(label, GestureBinding(description=label))
            action_summary = _describe_action(binding.action)
            n = model.samples_per_label.get(label, "?")
            print(f"  '{label}' ({n} samples) -> {binding.description}  [{action_summary}]")


def _describe_action(action: Optional[dict]) -> str:
    if not action:
        return "no action bound - recognized only"
    kind = action.get("action")
    if kind == "keys":
        return f"press {'+'.join(action.get('keys', []))}"
    if kind == "scroll":
        return f"scroll {action.get('direction', '?')}"
    return f"action: {kind}"


def run_teach_mode(samples_per_gesture: int = DEFAULT_SAMPLES_PER_GESTURE,
                    k: int = DEFAULT_K) -> None:
    """`python assistant.py --teach-gesture`

    Walks through capture -> label -> (optional action binding) -> train ->
    save entirely from the terminal + a live camera preview window, so
    teaching a new gesture never requires editing code or YAML by hand.
    One gesture is enough to save something useful (it works as a yes/no
    detector for that pose - see train_classifier()'s docstring); teaching
    a second, different gesture in the same sitting is optional but can
    improve accuracy by giving the classifier a contrasting shape.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python not installed. `pip install opencv-python mediapipe`.")
        return

    from gestures.tracker import HandTracker
    from config.loader import known_app_names, canonical_app_key  # noqa: F401 (kept close to loop.py's imports)

    print("\n--- Teach a new custom gesture ---")
    print("Scope: which app should this gesture apply in?")
    print("  Leave blank for 'global' (works no matter what has focus),")
    print("  or type an app key from config/apps.yaml (e.g. 'vscode', 'chrome').")
    scope = input("Scope > ").strip().lower() or GLOBAL_SCOPE

    samples: list[GestureSample] = []
    bindings: dict[str, GestureBinding] = {}

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open the camera for teach mode.")
        tracker.close()
        return

    try:
        while True:
            trained_so_far = len({s.label for s in samples})
            hint = ("blank to finish and save"
                    if trained_so_far >= 1
                    else "at least one gesture is required - don't leave this blank yet")
            label = input(f"\nGesture name / internal id, e.g. 'peace_sign' ({hint}) > ").strip()
            if not label:
                if trained_so_far >= 1:
                    break
                print("Need at least one gesture before there's anything to save - "
                      "give this one a name to continue.")
                continue

            description = input(
                f"What should '{label}' DO in '{scope}'? A short human-readable "
                f"function label, e.g. 'Save File' or 'Scroll Up' > "
            ).strip() or label
            action = _prompt_for_action()
            bindings[label] = GestureBinding(description=description, action=action)

            print(f"Show '{label}' to the camera. Capturing {samples_per_gesture} samples - "
                  f"hold the pose steady; a window will show progress. Press 'q' to abort this gesture.")
            captured = _capture_samples(cap, tracker, label, samples_per_gesture)
            if len(captured) < MIN_SAMPLES_PER_GESTURE:
                print(f"Only captured {len(captured)} usable sample(s) - need at least "
                      f"{MIN_SAMPLES_PER_GESTURE}. Try again with a steadier pose / better lighting.")
                continue
            samples.extend(captured)
            print(f"Captured {len(captured)} samples for '{label}' -> \"{description}\".")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    labels_seen = {s.label for s in samples}
    if not labels_seen:
        print("\nNo gestures were captured, so there's nothing to train. Nothing saved.")
        return
    if len(labels_seen) == 1:
        print(f"\nOnly one gesture ('{next(iter(labels_seen))}') was taught this session - "
              f"that's fine, it'll work as a yes/no detector for that pose. Teaching a "
              f"second, different gesture in the same session (next time) can improve "
              f"accuracy by giving the classifier something to contrast against, but "
              f"isn't required.")

    try:
        model = train_classifier(samples, scope=scope, bindings=bindings, k=k)
    except ValueError as e:
        print(f"\nCouldn't train: {e}")
        return

    store = TeachableGestureStore()
    path = store.save(model)
    print(f"\nSaved! Active for scope '{scope}' ({path}):")
    for label in sorted(labels_seen):
        print(f"  '{label}' -> {model.bindings[label].description}")
    print("Run `python assistant.py --gesture` to try them - they're checked "
          "automatically alongside the built-in gestures. Run "
          "`python assistant.py --list-gestures` any time to see everything "
          "taught so far, across every app.")


def _prompt_for_action() -> Optional[dict]:
    """Minimal terminal UI for binding a taught gesture to something that
    actually happens - same action schema gesture_profiles.yaml already
    uses, so run_gesture_loop's existing _run_gesture_action() handles
    both without new dispatch logic."""
    print("What should this gesture do?")
    print("  1) Press a key/hotkey (e.g. space, ctrl+s)")
    print("  2) Scroll up / down")
    print("  3) Nothing yet (just recognize it for now)")
    choice = input("Choice [1/2/3] > ").strip()

    if choice == "1":
        raw = input("Key(s), comma-separated (e.g. 'ctrl, s') > ").strip()
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return {"action": "keys", "keys": keys}
    elif choice == "2":
        direction = input("Direction [up/down] > ").strip().lower()
        if direction in ("up", "down"):
            return {"action": "scroll", "direction": direction}
    return None


def _capture_samples(cap, tracker, label: str, n: int) -> list[GestureSample]:
    """Grabs frames until `n` valid single-hand samples are collected, with
    a short pacing delay between samples so consecutive captures aren't
    just the same frame's jitter (encourages natural pose variation), and
    a live preview window so the user can see what's being captured."""
    import cv2

    collected: list[GestureSample] = []
    last_capture = 0.0
    min_gap = 0.15  # seconds between accepted samples

    while len(collected) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)

        now = time.monotonic()
        if hands and (now - last_capture) >= min_gap:
            collected.append(GestureSample(label=label, landmarks=hands[0].landmarks))
            last_capture = now

        cv2.putText(frame, f"'{label}': {len(collected)}/{n}  (q to abort)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Teach Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return collected
