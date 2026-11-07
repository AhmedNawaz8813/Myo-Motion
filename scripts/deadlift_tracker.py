"""
Deadlift Tracker — MyoMotion FYP
=================================
Uses MediaPipe Pose Landmarker (Tasks API, VIDEO mode) to track hip and
knee angles during a deadlift in real time from a webcam or video file.

Usage (always run from the project root):
    source .venv/bin/activate
    python scripts/deadlift_tracker.py              # webcam
    python scripts/deadlift_tracker.py video.mp4    # pre-recorded file

Camera orientation: person facing LEFT (right side perpendicular to camera).
Press Q to quit.
"""

import sys
import math
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve model path relative to this file so the script works regardless
# of the working directory it is launched from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "pose_landmarker_heavy.task")

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Pose — 33 landmarks)
# Right side (person facing left → right side faces the camera)
# ---------------------------------------------------------------------------
RIGHT_SHOULDER = 12
RIGHT_HIP      = 24
RIGHT_KNEE     = 26
RIGHT_ANKLE    = 28


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Returns the interior joint angle (0–180°) at vertex *b*.

    Args:
        a: (x, y) of the first point.
        b: (x, y) of the vertex.
        c: (x, y) of the third point.
    """
    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(math.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def pixel_coords(landmark, w: int, h: int) -> tuple:
    """Convert a normalised landmark to (x, y) pixel coordinates."""
    return (landmark.x * w, landmark.y * h)


def main():
    # ------------------------------------------------------------------
    # 1. Validate model file
    # ------------------------------------------------------------------
    if not os.path.isfile(MODEL_PATH):
        sys.exit(
            f"[ERROR] Model not found: {MODEL_PATH}\n"
            "Download pose_landmarker_heavy.task and place it in models/."
        )

    # ------------------------------------------------------------------
    # 2. Build the PoseLandmarker
    # ------------------------------------------------------------------
    # CPU delegate is required on macOS — the GPU delegate is Linux-only
    # and causes a Metal crash if used here.
    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )

    # VIDEO mode applies temporal smoothing across frames (better than
    # per-frame IMAGE mode for angle stability).
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    # ------------------------------------------------------------------
    # 3. Open video source
    # ------------------------------------------------------------------
    source = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = default webcam
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video source: {source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0  # Webcams sometimes report 0 — fall back to 30 fps

    print(f"[INFO] Source: {source!r}  |  FPS: {fps:.1f}")
    print("[INFO] Press Q to quit.")

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1

            # MediaPipe Tasks API requires RGB; OpenCV captures BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # VIDEO mode requires a strictly increasing timestamp in ms
            timestamp_ms = int((frame_index * 1000) / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first (only) person
                h, w, _ = frame.shape

                # ---- Draw the full 33-landmark skeleton ----
                drawing_utils.draw_landmarks(
                    frame,
                    landmarks,
                    pose_connections,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=4
                    ),
                    connection_drawing_spec=drawing_utils.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )

                # ---- Extract right-side joint positions ----
                shoulder = pixel_coords(landmarks[RIGHT_SHOULDER], w, h)
                hip      = pixel_coords(landmarks[RIGHT_HIP],      w, h)
                knee     = pixel_coords(landmarks[RIGHT_KNEE],     w, h)
                ankle    = pixel_coords(landmarks[RIGHT_ANKLE],    w, h)

                # ---- Compute angles ----
                hip_angle  = calculate_angle(shoulder, hip,  knee)
                knee_angle = calculate_angle(hip,      knee, ankle)

                # ---- Overlay angle text ----
                cv2.putText(
                    frame, f"Hip: {int(hip_angle)}\u00b0",
                    (int(hip[0]) - 90, int(hip[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame, f"Knee: {int(knee_angle)}\u00b0",
                    (int(knee[0]) - 90, int(knee[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            cv2.imshow("MyoMotion — Deadlift Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
