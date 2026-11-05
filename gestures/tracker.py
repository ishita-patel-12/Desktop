"""
Thin wrapper around MediaPipe Hands. This module's ONLY job is turning
webcam frames into hand landmark coordinates - it knows nothing about what
a "fist" or "swipe" means. That classification logic lives in recognizer.py,
kept separate so tracker.py stays swappable (e.g. for a different CV
backend later) without touching gesture logic, and so recognizer.py is
unit-testable on canned landmark data without a real camera.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class HandFrame:
    """One frame's worth of hand landmark data, already converted to
    plain (x, y) tuples in normalized [0,1] image coordinates - so
    downstream code never has to import mediapipe just to read a point."""
    landmarks: list[tuple[float, float]]  # 21 points, MediaPipe's hand model order
    handedness: str  # "Left" or "Right"


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.6):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for gesture tracking. "
                "`pip install mediapipe opencv-python`."
            ) from e

        self._legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

        if self._legacy:
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
        else:
            import os
            import urllib.request
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(model_dir, "hand_landmarker.task")
            if not os.path.exists(model_path):
                logger.info("Downloading hand_landmarker.task model for MediaPipe Tasks API...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=tracking_confidence,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            self._mp = mp
            self._vision = vision

        logger.info(f"HandTracker initialized (legacy={self._legacy}, max_hands={max_hands})")

    def process(self, frame_bgr) -> list[HandFrame]:
        """Takes a raw OpenCV BGR frame, returns 0+ detected hands as
        HandFrame objects. Caller owns the camera loop; this is a pure
        per-frame function so it's easy to test and easy to swap out."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands: list[HandFrame] = []

        if self._legacy:
            results = self._hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return hands

            handedness_list = results.multi_handedness or []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                label = "Right"
                if i < len(handedness_list):
                    label = handedness_list[i].classification[0].label
                hands.append(HandFrame(landmarks=points, handedness=label))
        else:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            results = self._detector.detect(mp_image)

            if not results.hand_landmarks:
                return hands

            for i, hand_landmarks in enumerate(results.hand_landmarks):
                points = [(lm.x, lm.y) for lm in hand_landmarks]
                label = "Right"
                if i < len(results.handedness) and results.handedness[i]:
                    label = results.handedness[i][0].category_name
                hands.append(HandFrame(landmarks=points, handedness=label))

        return hands

    def draw_landmarks(self, frame_bgr, hand_frame_index: int, results) -> None:
        """Optional: draws MediaPipe's landmark skeleton onto the frame."""
        if self._legacy:
            if not results or not getattr(results, "multi_hand_landmarks", None):
                return
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame_bgr, hand_landmarks, self._mp_hands.HAND_CONNECTIONS
                )

    def close(self) -> None:
        if self._legacy:
            self._hands.close()
        else:
            self._detector.close()