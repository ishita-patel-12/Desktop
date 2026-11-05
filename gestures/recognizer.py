"""
Turns a stream of hand landmarks (from tracker.py) into discrete gesture
events the rest of the app can act on. Deliberately has ZERO dependency on
mediapipe/opencv - it only ever sees plain (x, y) tuples - so this whole
module is unit-testable with synthetic landmark data and no camera.

Design choices carried over from the project blueprint:
  - Fist does NOT immediately close anything. It must be HELD for
    fist_hold_seconds before it fires a single "fist_hold" event; the
    caller (assistant.py / gesture loop) is responsible for turning that
    into "Should I close X?" + waiting for confirmation. This avoids
    accidental closes from people just resting their hand as a fist.
  - A cooldown after any fired event prevents the same gesture re-firing
    every single frame while held.
  - Swipes are detected from fingertip displacement over a short rolling
    window, not single-frame deltas, to avoid false positives from
    tracking jitter.
"""

from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# MediaPipe Hands landmark indices (21-point model)
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_extended(landmarks: list[Point], tip: int, mcp: int) -> bool:
    """A finger counts as 'extended' if its tip is farther from the wrist
    than its own MCP (knuckle) joint is. Using distance-from-wrist rather
    than a simple y-coordinate comparison makes this robust to the hand
    being rotated/tilted in front of the camera, not just held upright."""
    wrist = landmarks[WRIST]
    return _dist(landmarks[tip], wrist) > _dist(landmarks[mcp], wrist) * 1.15


def classify_static_pose(landmarks: list[Point]) -> str:
    """Classifies a single frame's hand shape. Returns one of:
    'fist', 'open_palm', 'point', 'pinch', 'unknown'."""
    if len(landmarks) < 21:
        return "unknown"

    index_ext = _finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    middle_ext = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring_ext = _finger_extended(landmarks, RING_TIP, RING_MCP)
    pinky_ext = _finger_extended(landmarks, PINKY_TIP, PINKY_MCP)

    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # Pinch: thumb tip and index tip close together, checked before the
    # other poses since a pinch can otherwise look like a partial fist.
    hand_span = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) or 1e-6
    if _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) < hand_span * 0.35:
        return "pinch"

    if extended_count == 0:
        return "fist"
    if extended_count >= 4:
        return "open_palm"
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "point"
    return "unknown"


@dataclass
class _SwipeTracker:
    window_seconds: float = 0.35
    min_travel_fraction: float = 0.18
    # How much bigger the dominant axis's movement must be than the other
    # axis before committing to a direction. Without this, a swipe that's
    # mostly-vertical-but-a-little-diagonal (extremely common - hardly
    # anyone swipes in a perfectly straight line) could tip either way on
    # a single noisy frame, which is the main reason up/down and
    # left/right used to feel unreliable/swapped.
    dominance_ratio: float = 1.3
    # Exponential-moving-average smoothing applied to the tracked point
    # before it's used for direction math (1.0 = no smoothing, fully raw;
    # lower = smoother but laggier). Raw MediaPipe landmarks jitter by a
    # percent or two of frame size even when the hand is perfectly still,
    # which is enough noise to flip a close dx-vs-dy call frame to frame.
    smoothing_alpha: float = 0.55
    # BUG FIX: MediaPipe normalizes x by frame WIDTH and y by frame HEIGHT
    # independently. At a non-square resolution (640x480 default: 4:3),
    # that means the same physical hand movement produces a SMALLER
    # normalized dx than the identical movement produces as dy - dx is
    # divided by the bigger number (640), dy by the smaller one (480). A
    # single min_travel_fraction/dominance_ratio applied directly to both
    # therefore silently made horizontal swipes need proportionally more
    # real motion than vertical ones to register - not a tuning issue,
    # an axis-scale mismatch. aspect_ratio (width/height) rescales dx back
    # into the same physical-equivalent units as dy before any comparison,
    # so a swipe of a given real size is equally easy to detect regardless
    # of direction. Defaults to 640/480; GestureRecognizer/gestures/loop.py
    # pass the actual configured camera resolution through.
    aspect_ratio: float = 640 / 480
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    _smoothed: Optional[Point] = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self.history.clear()
        self._smoothed = None

    def add(self, point: Point, t: float) -> Optional[str]:
        if self._smoothed is None:
            self._smoothed = point
        else:
            a = self.smoothing_alpha
            self._smoothed = (
                a * point[0] + (1 - a) * self._smoothed[0],
                a * point[1] + (1 - a) * self._smoothed[1],
            )
        point = self._smoothed

        self.history.append((point, t))
        while self.history and t - self.history[0][1] > self.window_seconds:
            self.history.popleft()
        if len(self.history) < 2:
            return None

        start_point, start_t = self.history[0]
        raw_dx = point[0] - start_point[0]
        dy = point[1] - start_point[1]
        # Rescale dx into the same physical-equivalent unit as dy (see the
        # aspect_ratio docstring above) - direction sign is unaffected
        # since aspect_ratio is always positive, only magnitude changes.
        dx = raw_dx * self.aspect_ratio

        if abs(dx) < self.min_travel_fraction and abs(dy) < self.min_travel_fraction:
            return None

        if abs(dx) > abs(dy) * self.dominance_ratio:
            direction = "right" if raw_dx > 0 else "left"
        elif abs(dy) > abs(dx) * self.dominance_ratio:
            direction = "down" if dy > 0 else "up"
        else:
            # Movement isn't clearly dominant on either axis yet (too
            # diagonal to call confidently) - keep accumulating rather than
            # guessing; this naturally resolves a frame or two later as
            # more travel comes in, or the window ages the ambiguous data out.
            return None

        self.reset()
        return f"swipe_{direction}"


class GestureRecognizer:
    def __init__(self, fist_hold_seconds: float = 0.7, cooldown_seconds: float = 0.6,
                 swipe_window_seconds: float = 0.35, swipe_min_travel_fraction: float = 0.18,
                 swipe_dominance_ratio: float = 1.3, swipe_smoothing_alpha: float = 0.55,
                 camera_width: int = 640, camera_height: int = 480):
        self.fist_hold_seconds = fist_hold_seconds
        self.cooldown_seconds = cooldown_seconds

        self._fist_since: Optional[float] = None
        self._fist_fired = False
        self._last_event_time: float = 0.0
        self._swipe_tracker = _SwipeTracker(
            window_seconds=swipe_window_seconds,
            min_travel_fraction=swipe_min_travel_fraction,
            dominance_ratio=swipe_dominance_ratio,
            smoothing_alpha=swipe_smoothing_alpha,
            # See _SwipeTracker.aspect_ratio's docstring: must match the
            # actual camera resolution swipes are being tracked against,
            # or the correction over/under-shoots instead of cancelling
            # the axis-scale mismatch out.
            aspect_ratio=camera_width / camera_height if camera_height else 1.0,
        )
        self._last_pose: str = "unknown"
        self._last_swipe_point: Optional[Point] = None

    def update(self, landmarks: list[Point], timestamp: Optional[float] = None) -> Optional[str]:
        """Feed one frame's landmarks in. Returns an event name if one
        fired this frame, else None. Event names:
          'fist_hold', 'open_palm', 'point', 'pinch',
          'swipe_left', 'swipe_right', 'swipe_up', 'swipe_down'
        """
        t = timestamp if timestamp is not None else time.monotonic()
        pose = classify_static_pose(landmarks)
        self._last_pose = pose

        in_cooldown = (t - self._last_event_time) < self.cooldown_seconds

        # --- fist hold-to-confirm ---
        if pose == "fist":
            self._swipe_tracker.reset()
            self._last_swipe_point = None
            if self._fist_since is None:
                self._fist_since = t
                self._fist_fired = False
            elif not self._fist_fired and (t - self._fist_since) >= self.fist_hold_seconds:
                self._fist_fired = True
                self._last_event_time = t
                return "fist_hold"
            return None
        else:
            self._fist_since = None
            self._fist_fired = False

        # --- swipe (tracked from a stable palm point, not the fingertip) ---
        # Tracked from the middle-finger MCP knuckle rather than a fingertip:
        # a fingertip's position changes both when the hand physically
        # moves AND when the finger itself bends/curls (e.g. transitioning
        # between an open palm and a point), which added noise unrelated
        # to the swipe itself. The MCP knuckle is a rigid part of the palm,
        # so it isolates "the hand moved" from "a finger moved" - and
        # crucially this also means swipes are now recognized whether the
        # hand is doing a flat open-palm swipe (the natural way most people
        # swipe) or pointing, not only while held in a strict one-finger
        # "point" pose. Not tracked during 'fist' (handled above) or
        # 'pinch' (a distinct one-shot gesture) so those keep their own
        # separate meaning.
        #
        # Not gated by `in_cooldown`: a static pose fired as an event on
        # the first frame of a swipe would otherwise put the whole
        # recognizer in cooldown and swallow the swipe that follows it a
        # few frames later. Swipes self-limit via the required travel
        # distance and clearing their history once fired, so a separate
        # cooldown isn't needed here.
        if pose in ("point", "open_palm", "unknown") and len(landmarks) > MIDDLE_MCP:
            self._last_swipe_point = landmarks[MIDDLE_MCP]
            swipe = self._swipe_tracker.add(landmarks[MIDDLE_MCP], t)
            if swipe:
                self._last_event_time = t
                return swipe
        else:
            self._swipe_tracker.reset()

        if in_cooldown:
            return None

        if pose in ("open_palm", "point", "pinch"):
            self._last_event_time = t
            return pose

        return None

    @property
    def last_pose(self) -> str:
        return self._last_pose

    @property
    def last_swipe_point(self) -> Optional[Point]:
        """The most recent point fed to the swipe tracker (a hand's
        MIDDLE_MCP knuckle), if any - exposed purely so the gesture loop's
        debug overlay can draw a marker on exactly the point direction
        detection is based on, instead of guessing."""
        return self._last_swipe_point

    def set_camera_resolution(self, width: int, height: int) -> None:
        """Updates the swipe tracker's axis-scale correction (see
        _SwipeTracker.aspect_ratio) to match the camera's ACTUAL resolution.
        Needed because webcams frequently ignore a requested resolution -
        gestures/loop.py constructs this recognizer before opening the
        camera (config says what to ask for), then calls this once the
        camera reports what it actually granted, so the correction is
        never silently wrong for a webcam that didn't honor the request."""
        if height:
            self._swipe_tracker.aspect_ratio = width / height