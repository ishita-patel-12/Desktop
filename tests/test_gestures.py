"""
Synthetic-landmark tests for gestures/recognizer.py. No camera or mediapipe
needed - these build plain (x, y) point lists by hand to exercise the pure
classification math. Run with: pytest tests/test_gestures.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gestures.recognizer import classify_static_pose, GestureRecognizer

WRIST = (0.5, 0.9)


def _make_hand(finger_states: dict, thumb_tip=(0.35, 0.75)) -> list:
    """Builds a 21-point landmark list. finger_states maps
    'index'/'middle'/'ring'/'pinky' -> 'extended' or 'curled'.
    MCP joints are fixed; tip position depends on extended/curled."""
    mcp_x = {"index": 0.45, "middle": 0.5, "ring": 0.55, "pinky": 0.6}
    points = [None] * 21
    points[0] = WRIST
    points[3] = thumb_tip  # thumb_ip, unused directly
    points[4] = thumb_tip  # thumb_tip

    layout = {
        "index": (5, 6, 8),
        "middle": (9, 10, 12),
        "ring": (13, 14, 16),
        "pinky": (17, 18, 20),
    }
    for finger, (mcp_i, pip_i, tip_i) in layout.items():
        x = mcp_x[finger]
        mcp = (x, 0.6)
        points[mcp_i] = mcp
        points[pip_i] = mcp  # not used by current classifier, filler
        if finger_states.get(finger) == "extended":
            points[tip_i] = (x - 0.03, 0.2)  # far from wrist
        else:
            points[tip_i] = (x, 0.62)  # curled back toward wrist

    return points


def test_open_palm():
    hand = _make_hand({"index": "extended", "middle": "extended",
                        "ring": "extended", "pinky": "extended"})
    assert classify_static_pose(hand) == "open_palm"


def test_fist():
    hand = _make_hand({"index": "curled", "middle": "curled",
                        "ring": "curled", "pinky": "curled"})
    assert classify_static_pose(hand) == "fist"


def test_point():
    hand = _make_hand({"index": "extended", "middle": "curled",
                        "ring": "curled", "pinky": "curled"})
    assert classify_static_pose(hand) == "point"


def test_pinch():
    hand = _make_hand(
        {"index": "extended", "middle": "curled", "ring": "curled", "pinky": "curled"},
        thumb_tip=(0.42, 0.21),  # right next to the extended index tip
    )
    assert classify_static_pose(hand) == "pinch"


def test_fist_does_not_fire_immediately():
    """Core safety requirement from the blueprint: a fist must be HELD,
    not fire the instant it's detected."""
    hand = _make_hand({"index": "curled", "middle": "curled",
                        "ring": "curled", "pinky": "curled"})
    recognizer = GestureRecognizer(fist_hold_seconds=0.7, cooldown_seconds=0.6)

    event = recognizer.update(hand, timestamp=0.0)
    assert event is None  # not held long enough yet

    event = recognizer.update(hand, timestamp=0.3)
    assert event is None

    event = recognizer.update(hand, timestamp=0.75)
    assert event == "fist_hold"  # now it's been held >= 0.7s


def test_fist_hold_fires_only_once_per_hold():
    hand = _make_hand({"index": "curled", "middle": "curled",
                        "ring": "curled", "pinky": "curled"})
    recognizer = GestureRecognizer(fist_hold_seconds=0.5, cooldown_seconds=0.6)

    recognizer.update(hand, timestamp=0.0)
    first = recognizer.update(hand, timestamp=0.6)
    second = recognizer.update(hand, timestamp=0.9)  # still holding fist

    assert first == "fist_hold"
    assert second is None  # doesn't re-fire every frame while still held


def test_open_hand_breaks_fist_hold_timer():
    fist = _make_hand({"index": "curled", "middle": "curled",
                        "ring": "curled", "pinky": "curled"})
    palm = _make_hand({"index": "extended", "middle": "extended",
                        "ring": "extended", "pinky": "extended"})
    recognizer = GestureRecognizer(fist_hold_seconds=0.5, cooldown_seconds=0.1)

    recognizer.update(fist, timestamp=0.0)
    recognizer.update(palm, timestamp=0.2)  # briefly opens hand - resets timer
    event = recognizer.update(fist, timestamp=0.6)  # only 0.4s into new fist
    assert event is None


def test_swipe_left_detected_from_palm_movement():
    """Swipes are tracked from the middle-finger MCP knuckle (a stable
    part of the palm), not a fingertip - moving landmark 9 here, not 8."""
    recognizer = GestureRecognizer()
    events = []
    for i in range(6):
        x = 0.8 - i * 0.12  # palm moving left across frames
        hand = _make_hand({"index": "extended", "middle": "curled",
                            "ring": "curled", "pinky": "curled"})
        hand[9] = (x, 0.6)
        events.append(recognizer.update(hand, timestamp=i * 0.05))

    assert "swipe_left" in events


def test_swipe_detected_during_open_palm_not_just_point():
    """Regression test for the main reported bug: swipes used to only be
    tracked while the hand held a strict one-finger 'point' pose, so a
    natural open-palm swipe (how most people actually swipe) silently did
    nothing. They should now be detected during open_palm too."""
    recognizer = GestureRecognizer()
    events = []
    for i in range(6):
        x = 0.8 - i * 0.12
        hand = _make_hand({"index": "extended", "middle": "extended",
                            "ring": "extended", "pinky": "extended"})
        assert classify_static_pose(hand) == "open_palm"  # sanity check on the fixture
        hand[9] = (x, 0.6)
        events.append(recognizer.update(hand, timestamp=i * 0.05))

    assert "swipe_left" in events


def test_swipe_not_tracked_during_fist():
    """A fist has its own separate meaning (hold-to-confirm-close) and
    should never also register as a swipe just because the hand moved
    while balled up."""
    recognizer = GestureRecognizer(fist_hold_seconds=10.0)  # won't fire fist_hold mid-test
    events = []
    for i in range(6):
        x = 0.8 - i * 0.12
        hand = _make_hand({"index": "curled", "middle": "curled",
                            "ring": "curled", "pinky": "curled"})
        hand[9] = (x, 0.6)
        events.append(recognizer.update(hand, timestamp=i * 0.05))

    assert "swipe_left" not in events
    assert all(e is None for e in events)


def test_diagonal_movement_does_not_fire_until_one_axis_dominates():
    """A true 45-degree diagonal - equal REAL-WORLD movement along both
    axes - shouldn't be coin-flipped into up/down or left/right. Note this
    is raw dy_step vs raw dx_step = dy_step * (height/width), NOT equal
    raw units - equal raw units would no longer be an actual diagonal
    once the aspect-ratio correction is applied (that correction is the
    whole point: raw units alone don't represent equal physical distance
    at a non-square resolution)."""
    recognizer = GestureRecognizer(camera_width=640, camera_height=480, swipe_dominance_ratio=1.3)
    events = []
    for i in range(6):
        x = 0.8 - i * 0.075   # 0.075 = 0.1 * (480/640): true-diagonal partner to dy's 0.1/step
        y = 0.2 + i * 0.1
        hand = _make_hand({"index": "extended", "middle": "curled",
                            "ring": "curled", "pinky": "curled"})
        hand[9] = (x, y)
        events.append(recognizer.update(hand, timestamp=i * 0.05))

    assert all(e is None for e in events)


# --- Regression tests: configurable swipe sensitivity (performance fix) ---
# Added alongside camera-resolution/FPS fixes for reported "gesture
# recognition degraded" - settings.yaml now exposes swipe_window_seconds /
# swipe_min_travel_fraction so users can tune sensitivity for their own
# camera/setup without touching code. These confirm the knobs actually work.

def test_small_movement_is_not_a_swipe_with_default_threshold():
    """A movement well under the default 18%-of-frame threshold should NOT
    register as a swipe."""
    recognizer = GestureRecognizer()
    events = []
    for i in range(6):
        x = 0.5 - i * 0.01  # tiny movement: 5% of frame width total
        hand = _make_hand({"index": "extended", "middle": "curled",
                            "ring": "curled", "pinky": "curled"})
        hand[9] = (x, 0.6)
        events.append(recognizer.update(hand, timestamp=i * 0.05))
    assert "swipe_left" not in events


def test_lowering_min_travel_fraction_makes_small_movements_register():
    """The same small movement DOES register once the sensitivity is
    turned up (lower min_travel_fraction) - this is the knob a user with
    reduced camera performance / a smaller motion range would turn."""
    recognizer = GestureRecognizer(swipe_min_travel_fraction=0.03)
    events = []
    for i in range(6):
        x = 0.5 - i * 0.01  # same tiny movement as above
        hand = _make_hand({"index": "extended", "middle": "curled",
                            "ring": "curled", "pinky": "curled"})
        hand[9] = (x, 0.6)
        events.append(recognizer.update(hand, timestamp=i * 0.05))
    # The tracker clears its history the instant a swipe fires, so it may
    # register on an earlier frame rather than the last one - check across
    # the whole gesture, not just the final frame's return value.
    assert "swipe_left" in events


def test_swipe_window_seconds_is_configurable():
    """A movement spread out over more time than the (shortened) window
    allows should NOT register, since the tracker only looks at motion
    within the last `swipe_window_seconds`."""
    recognizer = GestureRecognizer(swipe_window_seconds=0.1, swipe_min_travel_fraction=0.18)
    events = []
    for i in range(6):
        x = 0.8 - i * 0.12
        hand = _make_hand({"index": "extended", "middle": "curled",
                            "ring": "curled", "pinky": "curled"})
        hand[9] = (x, 0.6)
        # Spread frames out over 0.3s total - wider than the 0.1s window,
        # so older points keep getting dropped before enough travel
        # accumulates within the window.
        events.append(recognizer.update(hand, timestamp=i * 0.06))
    assert "swipe_left" not in events


# ---------------------------------------------------------------------------
# Aspect-ratio correction (bug fix: left/right used to need proportionally
# more real hand motion than up/down at non-square camera resolutions)
# ---------------------------------------------------------------------------

def _swipe_hand(x: float, y: float) -> list:
    hand = _make_hand({"index": "extended", "middle": "curled",
                        "ring": "curled", "pinky": "curled"})
    hand[9] = (x, y)
    return hand


def test_horizontal_swipe_of_realistic_size_now_registers_at_default_resolution():
    """At 640x480, MediaPipe normalizes x by 640 and y by 480 - so a real
    hand movement that produces dy=0.267 (moving vertically) produces only
    dx=0.20 for the identical physical movement made horizontally instead
    (0.267 * 480/640 = 0.20). Before the aspect-ratio fix, that 0.20 was
    compared directly against the min_travel threshold with no correction
    - see test_without_aspect_correction_the_same_horizontal_swipe_would_
    have_failed() below for the direct side-by-side. This is the reported
    bug: left/right swipes needing more effort than up/down for no
    functional reason."""
    recognizer = GestureRecognizer(camera_width=640, camera_height=480,
                                    swipe_min_travel_fraction=0.18)
    events = []
    for i in range(1, 6):  # start at i=1: the very first frame only seeds
        x = 0.8 - i * 0.05  # the smoothing filter, it never has a delta yet
        events.append(recognizer.update(_swipe_hand(x, 0.6), timestamp=i * 0.05))
    assert "swipe_left" in events


def test_vertical_swipe_of_the_same_physical_size_also_registers():
    """Companion to the test above: the vertically-equivalent movement of
    the same real size (dy_step = dx_step * width/height) should register
    just as easily - confirming parity between axes, not just that the
    horizontal case was independently loosened."""
    recognizer = GestureRecognizer(camera_width=640, camera_height=480,
                                    swipe_min_travel_fraction=0.18)
    dy_step = 0.05 * (640 / 480)  # same real-world distance as the dx_step above
    events = []
    for i in range(1, 6):
        y = 0.2 + i * dy_step
        events.append(recognizer.update(_swipe_hand(0.5, y), timestamp=i * 0.05))
    assert "swipe_down" in events


def test_without_aspect_correction_the_same_horizontal_swipe_would_have_failed():
    """Guards against the fix silently regressing: a 'square camera'
    recognizer (width == height, so aspect_ratio == 1.0 - equivalent to
    the old, uncorrected behavior) must NOT register the identical dx
    movement that the corrected recognizer above confirms DOES register.
    This documents that the correction is doing real work, not a no-op."""
    recognizer = GestureRecognizer(camera_width=480, camera_height=480,
                                    swipe_min_travel_fraction=0.18)
    events = []
    for i in range(1, 6):
        x = 0.8 - i * 0.05  # identical raw movement to the corrected test above
        events.append(recognizer.update(_swipe_hand(x, 0.6), timestamp=i * 0.05))
    assert "swipe_left" not in events


def test_set_camera_resolution_updates_the_correction():
    """gestures/loop.py constructs the recognizer before opening the
    camera (config says what to *request*), then calls this once the
    camera reports what it actually granted - webcams frequently ignore
    the requested resolution. Checking the tracker's resolved
    aspect_ratio directly (rather than round-tripping through behavior)
    keeps this test independent of swipe-timing/smoothing specifics."""
    recognizer = GestureRecognizer(camera_width=480, camera_height=480)
    assert recognizer._swipe_tracker.aspect_ratio == pytest.approx(1.0)

    recognizer.set_camera_resolution(640, 480)
    assert recognizer._swipe_tracker.aspect_ratio == pytest.approx(640 / 480)