"""
Deadlift Tracker — MyoMotion FYP
=================================
Tracks hip and knee joint angles during a deadlift using MediaPipe Pose,
classifies biomechanical phases via a state machine, and counts completed
repetitions.

Input mode: toggle the VIDEO_SOURCE lines below (live camera vs video file).
Camera orientation: person facing LEFT (right side toward camera).
Press Q to quit.
"""

import sys
import math
import os
from enum import Enum

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

# ---------------------------------------------------------------------------
# Input Mode — uncomment ONE of the two lines below
# ---------------------------------------------------------------------------
VIDEO_SOURCE = 0                                # Live camera mode
# VIDEO_SOURCE = "path/to/your/video.mp4"       # Video file mode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "pose_landmarker_heavy.task")

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose — 33 landmarks, right side)
# ---------------------------------------------------------------------------
RIGHT_SHOULDER = 12
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

# ---------------------------------------------------------------------------
# Phase-detection angle thresholds (degrees)
# ---------------------------------------------------------------------------
# These thresholds partition the deadlift range of motion into phases based
# on the hip angle (shoulder–hip–knee) and knee angle (hip–knee–ankle).
#
# Bottom position (setup / post-eccentric):
#   Hip  ≈  70–110°   (trunk nearly horizontal, thigh angled back)
#   Knee ≈ 100–140°   (moderate bend for conventional deadlift)
#
# Top position (lockout):
#   Hip  ≈ 165–180°   (trunk and thigh nearly aligned, standing tall)
#   Knee ≈ 165–180°   (leg fully straight)
#
# *_FLEXED thresholds: below these the joint is considered "bent".
# *_EXTENDED thresholds: above these the joint is considered "straight".
# The gap between flexed and extended creates a hysteresis band that
# prevents rapid toggling near boundary angles.
HIP_FLEXED_THRESH = 120
HIP_EXTENDED_THRESH = 160
KNEE_FLEXED_THRESH = 140
KNEE_EXTENDED_THRESH = 160

# Exponential moving average factor for angle smoothing (0–1).
# Higher → more responsive but noisier; lower → smoother but laggier.
EMA_ALPHA = 0.4

# The lockout must be held for this many consecutive frames before the
# state machine will accept a transition into eccentric. This prevents a
# fast pass-through at the top from being mis-classified as a hold.
LOCKOUT_MIN_FRAMES = 5


# ---------------------------------------------------------------------------
# Phase enum and display colours
# ---------------------------------------------------------------------------
class Phase(Enum):
    SETUP = "Setup"
    CONCENTRIC = "Concentric"
    LOCKOUT = "Lockout"
    ECCENTRIC = "Eccentric"


# BGR colours for the on-screen phase badge
PHASE_COLOURS = {
    Phase.SETUP:      (180, 180, 180),   # grey
    Phase.CONCENTRIC: (0, 200, 0),       # green  — ascending
    Phase.LOCKOUT:    (0, 180, 255),     # amber  — holding
    Phase.ECCENTRIC:  (0, 100, 255),     # red-orange — descending
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Interior joint angle (0–180°) at vertex *b*."""
    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(math.degrees(radians))
    return 360.0 - angle if angle > 180.0 else angle


def pixel_coords(landmark, w: int, h: int) -> tuple:
    """Normalised landmark → (x, y) pixel coordinates."""
    return (landmark.x * w, landmark.y * h)


def _ema(prev: float | None, curr: float, alpha: float = EMA_ALPHA) -> float:
    """Exponential moving average; bootstraps on the first sample."""
    if prev is None:
        return curr
    return alpha * curr + (1.0 - alpha) * prev


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class DeadliftStateMachine:
    """
    Tracks deadlift phase transitions and counts completed reps.

    State graph (one full rep = one cycle):

        SETUP ──► CONCENTRIC ──► LOCKOUT ──► ECCENTRIC ──► SETUP
          ▲                                                  │
          └──────────────── rep_count += 1 ◄─────────────────┘

    Transitions:
        SETUP → CONCENTRIC
            Hip angle rises above HIP_FLEXED_THRESH (athlete begins standing up).

        CONCENTRIC → LOCKOUT
            Both hip > HIP_EXTENDED_THRESH and knee > KNEE_EXTENDED_THRESH
            (near-full extension reached).

        CONCENTRIC → SETUP  (aborted rep)
            Angles fall back to the flexed zone without reaching lockout.

        LOCKOUT → ECCENTRIC
            After being held for LOCKOUT_MIN_FRAMES, hip drops below
            HIP_EXTENDED_THRESH (athlete begins lowering the bar).

        ECCENTRIC → SETUP  (rep complete)
            Both hip < HIP_FLEXED_THRESH and knee < KNEE_FLEXED_THRESH
            (bar returned to the floor). rep_count increments.
    """

    def __init__(self):
        self.phase: Phase = Phase.SETUP
        self.rep_count: int = 0
        self.lockout_frames: int = 0
        self._smooth_hip: float | None = None
        self._smooth_knee: float | None = None

    def update(self, raw_hip: float, raw_knee: float) -> Phase:
        """Feed raw joint angles for the current frame; returns the phase."""
        self._smooth_hip = _ema(self._smooth_hip, raw_hip)
        self._smooth_knee = _ema(self._smooth_knee, raw_knee)
        hip = self._smooth_hip
        knee = self._smooth_knee

        if self.phase == Phase.SETUP:
            if hip > HIP_FLEXED_THRESH:
                self.phase = Phase.CONCENTRIC
                self.lockout_frames = 0

        elif self.phase == Phase.CONCENTRIC:
            if hip > HIP_EXTENDED_THRESH and knee > KNEE_EXTENDED_THRESH:
                self.phase = Phase.LOCKOUT
                self.lockout_frames = 0
            elif hip < HIP_FLEXED_THRESH and knee < KNEE_FLEXED_THRESH:
                self.phase = Phase.SETUP

        elif self.phase == Phase.LOCKOUT:
            self.lockout_frames += 1
            if (self.lockout_frames >= LOCKOUT_MIN_FRAMES
                    and hip < HIP_EXTENDED_THRESH):
                self.phase = Phase.ECCENTRIC

        elif self.phase == Phase.ECCENTRIC:
            if hip < HIP_FLEXED_THRESH and knee < KNEE_FLEXED_THRESH:
                self.rep_count += 1
                self.phase = Phase.SETUP

        return self.phase

    @property
    def hip_angle(self) -> float:
        return self._smooth_hip if self._smooth_hip is not None else 0.0

    @property
    def knee_angle(self) -> float:
        return self._smooth_knee if self._smooth_knee is not None else 0.0


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
def draw_hud(frame, sm: DeadliftStateMachine) -> None:
    """Overlay phase badge, rep counter, and smoothed angle readouts."""
    h, w = frame.shape[:2]
    colour = PHASE_COLOURS[sm.phase]

    # Phase badge — top left
    cv2.putText(
        frame, sm.phase.value, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3,
    )

    # Rep counter — top right
    rep_text = f"Reps: {sm.rep_count}"
    (tw, _), _ = cv2.getTextSize(
        rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2,
    )
    cv2.putText(
        frame, rep_text, (w - tw - 20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    )

    # Smoothed angles — below the phase badge
    cv2.putText(
        frame,
        f"Hip: {sm.hip_angle:.0f}°   Knee: {sm.knee_angle:.0f}°",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        sys.exit(
            f"[ERROR] Model not found: {MODEL_PATH}\n"
            "Download pose_landmarker_heavy.task and place it in models/."
        )

    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video source: {VIDEO_SOURCE!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0

    # Video files: pace playback at native FPS.
    # Live camera: minimal 1 ms delay so the loop doesn't block.
    is_live = isinstance(VIDEO_SOURCE, int)
    wait_ms = 1 if is_live else max(1, int(1000 / fps))

    print(f"[INFO] Source: {VIDEO_SOURCE!r}  |  FPS: {fps:.1f}"
          f"  |  Mode: {'Live camera' if is_live else 'Video file'}")
    print("[INFO] Press Q to quit.\n")

    sm = DeadliftStateMachine()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                if not is_live:
                    print(f"\n[INFO] Video ended. Total reps: {sm.rep_count}")
                break

            frame_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_index * 1000) / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                h, w, _ = frame.shape

                # Draw the full 33-landmark skeleton
                drawing_utils.draw_landmarks(
                    frame, lm, pose_connections,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=4,
                    ),
                    connection_drawing_spec=drawing_utils.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2,
                    ),
                )

                shoulder = pixel_coords(lm[RIGHT_SHOULDER], w, h)
                hip = pixel_coords(lm[RIGHT_HIP], w, h)
                knee = pixel_coords(lm[RIGHT_KNEE], w, h)
                ankle = pixel_coords(lm[RIGHT_ANKLE], w, h)

                raw_hip = calculate_angle(shoulder, hip, knee)
                raw_knee = calculate_angle(hip, knee, ankle)

                # Update the state machine only when valid landmarks exist
                sm.update(raw_hip, raw_knee)

                # Raw angle labels next to the skeleton joints
                cv2.putText(
                    frame, f"{int(raw_hip)}°",
                    (int(hip[0]) - 50, int(hip[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame, f"{int(raw_knee)}°",
                    (int(knee[0]) - 50, int(knee[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            # HUD is drawn every frame so phase/rep info stays visible
            # even during brief landmark dropouts
            draw_hud(frame, sm)

            cv2.imshow("MyoMotion — Deadlift Tracker", frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
