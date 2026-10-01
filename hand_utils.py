import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# which landmark indices connect to which, grouped by finger, with a BGR color each
FINGERS = {
    "thumb":  ([(0, 1), (1, 2), (2, 3), (3, 4)],         (0, 200, 255)),   # orange
    "index":  ([(0, 5), (5, 6), (6, 7), (7, 8)],         (0, 255, 0)),     # green
    "middle": ([(0, 9), (9, 10), (10, 11), (11, 12)],    (255, 200, 0)),   # cyan
    "ring":   ([(0, 13), (13, 14), (14, 15), (15, 16)],  (255, 0, 200)),   # purple
    "pinky":  ([(0, 17), (17, 18), (18, 19), (19, 20)],  (0, 100, 255)),   # red-orange
    "palm":   ([(5, 9), (9, 13), (13, 17)],              (200, 200, 200)), # gray
}


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr):
    """Run MediaPipe on one frame. Returns 21 landmarks or None if no hand."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    if not result.hand_landmarks:
        return None
    return result.hand_landmarks[0]


def to_feature_vector(landmarks):
    """Turn 21 landmarks into 63 numbers the classifier can learn from."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                                  # wrist becomes (0,0,0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale                               # hand size no longer matters
    return pts.flatten()


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    k = h / 480                                    # scale thickness with frame size
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for connections, color in FINGERS.values():
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], color, int(3 * k), cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, int(6 * k), (40, 40, 40), -1, cv2.LINE_AA)     # dark outline
        cv2.circle(frame, p, int(4 * k), (255, 255, 255), -1, cv2.LINE_AA)  # white joint


def draw_label(frame, text, color):
    """Text with a dark outline so it's readable over any background."""
    k = frame.shape[0] / 480                      # scale everything with frame height
    pos = (int(16 * k), int(52 * k))
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, pos, font, 1.4 * k, (0, 0, 0), int(6 * k), cv2.LINE_AA)   # outline
    cv2.putText(frame, text, pos, font, 1.4 * k, color, int(2 * k), cv2.LINE_AA)       # fill
