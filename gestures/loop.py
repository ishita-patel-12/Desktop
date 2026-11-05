"""
Phase 2 deliverable: live webcam window, finger tracking, scroll/swipe,
and fist-hold-then-confirm window close.

  python assistant.py --gesture

Kept separate from assistant.py's text/voice loops since it owns an
OpenCV window and its own frame loop, rather than being event-driven.
"""

from __future__ import annotations
import threading
import time
from typing import Optional
from loguru import logger

from config.loader import load_settings, load_gesture_profiles, canonical_app_key
from context.active_window import get_active_window_title, guess_app_name_from_title
from executor import apps as apps_executor
from gestures.teachable import (
    CustomGestureClassifier, TeachableGestureStore, GLOBAL_SCOPE,
)

INDEX_TIP_INDEX = 8  # MediaPipe hand-landmark index, mirrored here so this
                      # module doesn't need to import recognizer.py's
                      # internals just to draw a reference marker.


def _select_profile(title: Optional[str], profiles: dict) -> dict:
    """Pick which gesture-action profile applies for the current window.
    Checks every non-"default" profile's `match.title_contains` list against
    the active window title (case-insensitive substring), top to bottom in
    config file order; falls back to "default" if nothing matches."""
    title_lower = (title or "").lower()
    for key, profile in profiles.items():
        if key == "default" or not isinstance(profile, dict):
            continue
        match = profile.get("match") or {}
        needles = [s.lower() for s in match.get("title_contains", [])]
        if needles and any(n in title_lower for n in needles):
            return profile
    return profiles.get("default", {})


def _run_gesture_action(action: Optional[dict]) -> None:
    """Executes one mapped action (see config/gesture_profiles.yaml)."""
    if not action:
        return
    try:
        import pyautogui
    except ImportError:
        logger.warning("pyautogui not installed - gesture actions disabled.")
        return

    kind = action.get("action")
    if kind == "scroll":
        amount = 3 * 40
        pyautogui.scroll(amount if action.get("direction") == "up" else -amount)
    elif kind == "keys":
        keys = action.get("keys") or []
        if len(keys) == 1:
            pyautogui.press(keys[0])
        elif len(keys) > 1:
            pyautogui.hotkey(*keys)
    else:
        logger.warning(f"Unknown gesture action kind: {kind!r}")


def _dispatch_gesture_event(event: str, profiles: dict) -> None:
    """Look up the active window, pick its profile, and run whatever action
    that profile maps this swipe event to - this is what makes the same
    swipe mean 'switch tab' normally but 'press the Up arrow' in a browser
    game (see config/gesture_profiles.yaml)."""
    title = get_active_window_title()
    profile = _select_profile(title, profiles)
    action = profile.get(event) or profiles.get("default", {}).get(event)
    _run_gesture_action(action)
    logger.info(f"Gesture: {event} -> {action}")


def _close_active_window_directly(expected_title: Optional[str]) -> apps_executor.ExecResult:
    """Fallback close path used when the focused app either isn't listed in
    config/apps.yaml or its process name doesn't match anything psutil can
    find (e.g. a UWP app, a browser variant, a random utility). Rather than
    reporting failure, ask the OS to close whichever window currently has
    focus directly - this is what makes "close the current window" work
    generically instead of only for apps we happen to have config for.

    Delegates to os_adapter.get_adapter() rather than a platform-specific
    library directly, so this works the same way on Windows and Linux
    (X11) and degrades cleanly (a clear failure message, not a crash) on
    platforms/sessions the adapter layer doesn't support yet - see
    os_adapter/base.py.

    `expected_title` guards against closing the wrong window if focus moved
    on to something else (e.g. this app's own gesture-feed window) during
    the brief pause it took to get a spoken answer - the adapter itself
    re-checks this, but it's threaded through explicitly here too so the
    intent is visible at the call site.
    """
    from os_adapter import get_adapter
    success, message = get_adapter().close_active_window(expected_title=expected_title)
    return apps_executor.ExecResult(success, message)


def _scope_for_title(title: Optional[str]) -> str:
    """Resolve the active window into a teachable-gesture scope key: the
    canonical app key from config/apps.yaml when it's a known app (so a
    model trained while 'python' was the label in apps.yaml also applies
    to windows titled slightly differently), else a sanitized version of
    the raw title, else the global scope."""
    guessed = guess_app_name_from_title(title)
    if not guessed:
        return GLOBAL_SCOPE
    return canonical_app_key(guessed) or guessed.lower().strip() or GLOBAL_SCOPE


def _draw_hand_overlay(frame, cv2, landmarks, tracked_point, gesture_name: str,
                        last_swipe: Optional[str], fps: float) -> None:
    """Debug overlay so tracking is visible instead of a black box: a
    bounding box around the whole hand, a marker on the point swipes are
    actually tracked from (the middle-finger MCP knuckle - see
    recognizer.py for why that point instead of a fingertip), and a
    smaller marker on the index fingertip for reference since that's what
    people naturally look at when "pointing". All drawing here is a
    handful of cv2 primitives on an already-decoded frame - negligible
    cost (sub-millisecond) next to MediaPipe's inference time, so this
    does not meaningfully affect latency; see the perf notes in
    run_gesture_loop() for what actually costs time.
    """
    h, w = frame.shape[:2]
    xs = [p[0] * w for p in landmarks]
    ys = [p[1] * h for p in landmarks]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    pad = 12
    cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (255, 200, 0), 2)

    if tracked_point is not None:
        tx, ty = int(tracked_point[0] * w), int(tracked_point[1] * h)
        cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)
        cv2.putText(frame, "swipe anchor", (tx + 12, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    if len(landmarks) > INDEX_TIP_INDEX:
        fx, fy = int(landmarks[INDEX_TIP_INDEX][0] * w), int(landmarks[INDEX_TIP_INDEX][1] * h)
        cv2.circle(frame, (fx, fy), 5, (0, 255, 255), -1)

    cv2.putText(frame, f"FPS: {fps:.0f}  Gesture: {gesture_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if last_swipe:
        cv2.putText(frame, f"Last swipe: {last_swipe}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)


# Guards against a fist held slightly too long re-triggering a second
# confirmation flow while the first one is still waiting on an answer.
_confirmation_active = False
_confirmation_lock = threading.Lock()


def _handle_fist_hold(speak, listen) -> None:
    """Fist held -> identify active window -> ASK before closing anything.

    Never closes on its own; a spoken "yes" is required to actually close
    the window (a false-positive gesture should never cost unsaved work).

    `speak(text)` and `listen() -> str` are injected so this stays testable
    and so the caller decides how confirmation is delivered - in practice
    this is the same TTS/mic pipeline the voice assistant uses, which is
    what lets voice and gestures run side by side: closing your fist asks
    the question out loud and listens for your spoken answer, instead of
    requiring a keyboard (which used to block the whole camera feed while
    it waited on terminal input).
    """
    global _confirmation_active

    with _confirmation_lock:
        if _confirmation_active:
            return  # already asking about a previous fist-hold
        title = get_active_window_title()
        if title and ("Assistant - Gesture Feed" in title or title == "Assistant"):
            return
        app_name = guess_app_name_from_title(title)
        if not app_name or app_name.lower() in ("gesture feed", "assistant", "assistant - gesture feed"):
            return
        _confirmation_active = True

    def _confirm_and_close() -> None:
        global _confirmation_active
        try:
            speak(f"Should I close {app_name}? Say yes or no.")
            answer = (listen() or "").strip().lower()
            logger.info(f"Fist-close confirmation heard: {answer!r}")

            if any(w in answer for w in ("yes", "yeah", "yep", "sure", "do it", "confirm")):
                result = apps_executor.close_app(app_name, confirmed=True)
                if not result.success:
                    result = _close_active_window_directly(title)
                speak(result.message)
            else:
                speak("Okay, leaving it open.")
        finally:
            _confirmation_active = False

    # Runs on its own thread so the camera keeps grabbing/showing frames
    # (and other gestures like swipes keep working) while we wait for the
    # user's spoken answer, instead of freezing the whole gesture loop.
    threading.Thread(target=_confirm_and_close, daemon=True).start()


def run_gesture_loop(stop_event=None) -> None:
    """
    `stop_event`: optional threading.Event, shared with the voice loop in
    `--all` mode. Checked each frame so pressing 'q' here also tells the
    voice thread to stop (set in the `finally` block below), and is itself
    checked so an external shutdown request (e.g. the voice side ending
    first) closes the camera window too instead of the two loops being
    unable to affect each other at all.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python not installed. `pip install opencv-python mediapipe`.")
        return

    from gestures.tracker import HandTracker
    from gestures.recognizer import GestureRecognizer

    settings = load_settings()
    gesture_cfg = settings.get("gestures", {})
    camera_index = gesture_cfg.get("camera_index", 0)

    def speak(text: str) -> None:
        try:
            from voice.tts import say
            say(text)
        except Exception:
            print(f"Assistant: {text}")

    def listen() -> str:
        """Short yes/no listen for the fist-close confirmation. Shares the
        same mic-access lock as the voice conversation loop (see
        voice/stt.py) so this is safe to call even while `--all` mode's
        voice loop is running concurrently on another thread."""
        try:
            from voice.stt import listen_until_silence
            return listen_until_silence(
                silence_duration=0.8,
                pre_speech_timeout=5.0,
                max_duration=4.0,
            )
        except Exception as e:
            logger.warning(f"Could not listen for voice confirmation: {e}")
            return ""

    tracker = HandTracker(
        detection_confidence=gesture_cfg.get("detection_confidence", 0.7),
        tracking_confidence=gesture_cfg.get("tracking_confidence", 0.6),
    )
    recognizer = GestureRecognizer(
        fist_hold_seconds=gesture_cfg.get("fist_hold_seconds", 0.7),
        cooldown_seconds=gesture_cfg.get("cooldown_seconds", 0.6),
        swipe_window_seconds=gesture_cfg.get("swipe_window_seconds", 0.35),
        swipe_min_travel_fraction=gesture_cfg.get("swipe_min_travel_fraction", 0.18),
        swipe_dominance_ratio=gesture_cfg.get("swipe_dominance_ratio", 1.3),
        swipe_smoothing_alpha=gesture_cfg.get("swipe_smoothing_alpha", 0.55),
        camera_width=gesture_cfg.get("camera_width", 640),
        camera_height=gesture_cfg.get("camera_height", 480),
    )
    # --- Phase 3: teachable custom gestures ---
    # Loaded lazily and cached per scope (app key or "global") so a missing
    # model file is only ever checked-for once per scope, not every frame.
    # Re-running --teach-gesture while this loop is running won't be picked
    # up until restart - acceptable for a dev-loop feature like this.
    custom_store = TeachableGestureStore()
    custom_classifiers: dict[str, Optional[CustomGestureClassifier]] = {}

    def _get_custom_classifier(scope: str) -> Optional[CustomGestureClassifier]:
        if scope not in custom_classifiers:
            model = custom_store.load(scope)
            custom_classifiers[scope] = CustomGestureClassifier(model) if model else None
        return custom_classifiers[scope]

    def _dispatch_custom_gesture(landmarks, timestamp: float) -> Optional[str]:
        """Only reached when the built-in classifier didn't recognize the
        pose as one of its fixed set (fist/open_palm/point/pinch/swipe) -
        this is what lets a taught gesture like 'thumbs up' mean nothing in
        general but do something specific only while its trained app has
        focus, without colliding with the built-ins or with gestures
        trained for other apps. Returns the fired gesture's human-readable
        description (for the on-screen overlay), or None if nothing fired."""
        title = get_active_window_title()
        scope = _scope_for_title(title)

        fired = None
        classifier = _get_custom_classifier(scope)
        if classifier:
            fired = classifier.update(landmarks, timestamp=timestamp)
        if not fired and scope != GLOBAL_SCOPE:
            global_classifier = _get_custom_classifier(GLOBAL_SCOPE)
            if global_classifier:
                fired = global_classifier.update(landmarks, timestamp=timestamp)

        if fired:
            label, binding = fired
            if binding.action:
                _run_gesture_action(binding.action)
            logger.info(f"Custom gesture: '{label}' (scope={scope}) -> {binding.description}")
            return binding.description
        return None

    gesture_profiles = load_gesture_profiles() or {
        "default": {
            "swipe_up": {"action": "scroll", "direction": "up"},
            "swipe_down": {"action": "scroll", "direction": "down"},
            "swipe_left": {"action": "keys", "keys": ["ctrl", "shift", "tab"]},
            "swipe_right": {"action": "keys", "keys": ["ctrl", "tab"]},
        }
    }

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Could not open camera index {camera_index}.")
        return

    # --- Explicit capture settings ---
    # PERFORMANCE FIX: this used to leave the camera at whatever its default
    # resolution/FPS happened to be. Many webcams default to 720p/1080p,
    # and MediaPipe's per-frame hand-tracking cost scales heavily with
    # resolution - left uncapped, that alone can drag FPS down far enough
    # that swipes (which are timed over a ~0.35s window) only get 1-2 stale
    # samples to work with and stop registering reliably. Requesting a
    # modest, fixed resolution/FPS fixes that directly.
    #
    # CAP_PROP_BUFFERSIZE=1 additionally tells the backend not to queue up
    # frames faster than they're consumed - without it, if processing ever
    # falls behind capture (e.g. under CPU load from other work), OpenCV
    # can silently serve increasingly STALE frames from an internal buffer,
    # which looks exactly like "the camera can't keep up with my hand
    # anymore" even though the live feed itself is fine. Not every backend
    # honors this property, so it's set best-effort.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, gesture_cfg.get("camera_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, gesture_cfg.get("camera_height", 480))
    cap.set(cv2.CAP_PROP_FPS, gesture_cfg.get("camera_fps", 30))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass  # not all backends support this - harmless if ignored

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        f"Camera opened: requested {gesture_cfg.get('camera_width', 640)}x"
        f"{gesture_cfg.get('camera_height', 480)}, got {actual_w}x{actual_h} "
        f"@ {actual_fps:.0f}fps (some webcams ignore requested settings - "
        f"if this doesn't match, that's the camera/driver's choice, not a bug here)"
    )
    if actual_w and actual_h:
        # Re-derive the swipe-tracker's axis correction from what the
        # camera ACTUALLY granted, not just what was requested - see
        # GestureRecognizer.set_camera_resolution()'s docstring.
        recognizer.set_camera_resolution(actual_w, actual_h)

    logger.info("Gesture loop running. Press 'q' in the video window to quit.")
    prev_time = time.monotonic()

    # --- PERFORMANCE: run MediaPipe on a downscaled copy of the frame ---
    # MediaPipe's per-frame inference cost scales roughly with pixel count,
    # so detecting on a smaller copy is the single biggest lever for lower
    # latency short of a faster machine. This is safe to do because
    # HandTracker returns landmarks already normalized to [0, 1] - they
    # apply identically to the full-resolution frame regardless of what
    # size was actually fed to the model, so the display and the overlay
    # drawn on it stay full quality; only the (invisible) inference pass
    # gets smaller/faster. Defaults to 0.7 (30% fewer pixels each axis,
    # noticeably faster on modest CPUs) - set gestures.process_scale to
    # 1.0 in settings.yaml to disable, or lower it further (e.g. 0.5) on
    # slower hardware if hand detection still lags.
    process_scale = gesture_cfg.get("process_scale", 0.7)
    process_scale = min(max(process_scale, 0.2), 1.0)  # guard against nonsensical config values

    last_swipe_label: Optional[str] = None
    last_swipe_time = 0.0
    SWIPE_LABEL_DISPLAY_SECONDS = 1.0

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            ok, frame = cap.read()
            if not ok:
                logger.warning("Failed to read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror, so movement matches what the user sees

            if process_scale < 1.0:
                small = cv2.resize(frame, None, fx=process_scale, fy=process_scale,
                                    interpolation=cv2.INTER_LINEAR)
                hands = tracker.process(small)
            else:
                hands = tracker.process(frame)

            gesture_name = "none"
            if hands:
                landmarks = hands[0].landmarks
                event = recognizer.update(landmarks, timestamp=time.monotonic())
                gesture_name = recognizer.last_pose

                if event == "fist_hold":
                    _handle_fist_hold(speak, listen)
                elif event in ("swipe_up", "swipe_down", "swipe_left", "swipe_right"):
                    _dispatch_gesture_event(event, gesture_profiles)
                    last_swipe_label = event
                    last_swipe_time = time.monotonic()
                elif gesture_name == "unknown":
                    fired_description = _dispatch_custom_gesture(landmarks, time.monotonic())
                    if fired_description:
                        last_swipe_label = f"Custom: {fired_description}"
                        last_swipe_time = time.monotonic()

                if time.monotonic() - last_swipe_time > SWIPE_LABEL_DISPLAY_SECONDS:
                    last_swipe_label = None

                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                _draw_hand_overlay(frame, cv2, landmarks, recognizer.last_swipe_point,
                                    gesture_name, last_swipe_label, fps)
            else:
                now = time.monotonic()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.0f}  Gesture: none",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Assistant - Gesture Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if stop_event is not None:
            stop_event.set()