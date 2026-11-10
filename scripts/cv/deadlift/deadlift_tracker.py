"""
Deadlift Tracker — MyoMotion FYP
=================================
Tracks hip and knee joint angles during a deadlift using MediaPipe Pose,
segments each repetition into biomechanical phases with a time-based,
hysteresis-gated state machine, and counts completed repetitions.

Phases (one rep = one cycle):

    SETUP ──► CONCENTRIC ──► ISOMETRIC HOLD ──► ECCENTRIC ──► SETUP
    (STANDBY is the "not in a lift" state: start-up, tracking loss, stalled rep.)

Signal chain (per joint)
    raw 2-D angle → median-3 + jump gate → One-Euro low-pass filter
    → least-squares angular velocity over a short sliding time window
    → state machine (angle zones + velocity + displacement + dwell time).

Everything that depends on time uses *seconds from frame timestamps*, never
frame counts, so behaviour is identical at 15, 30 or 60 FPS and survives
dropped frames.

Input mode: toggle the VIDEO_SOURCE lines below (live camera vs video file).
Camera orientation: person facing LEFT (right side toward camera).
Press Q to quit.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

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
# scripts/cv/deadlift/ → scripts/cv/ → scripts/ → project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "pose_landmarker_heavy.task")

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose — 33 landmarks, right side)
# ---------------------------------------------------------------------------
RIGHT_SHOULDER = 12
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

# A joint angle is only used if all four landmarks are at least this visible
# and lie inside the image. Below this, MediaPipe is extrapolating.
MIN_LANDMARK_VISIBILITY = 0.5

# ---------------------------------------------------------------------------
# Signal conditioning
# ---------------------------------------------------------------------------
# One-Euro filter (Casiez, Roussel & Vogel, CHI 2012): the cut-off rises with
# signal speed, so it smooths hard while the lifter is still (kills jitter)
# and tracks closely while the lifter is moving (low lag).
FILTER_MIN_CUTOFF_HZ = 0.5     # cut-off at zero speed
FILTER_BETA = 0.01             # cut-off gain per deg/s of speed
FILTER_D_CUTOFF_HZ = 1.0       # cut-off for the internal speed estimate

# Physiological outlier gate. Landmark swaps / mis-detections produce
# single-frame jumps far beyond any real joint motion (peak hip angular
# velocity in a maximal pull is a few hundred deg/s).
MAX_ANGULAR_SPEED_DEG_S = 600.0
JUMP_SLACK_DEG = 8.0           # allowance on top of speed * dt

# Angular velocity = least-squares slope over this window. Noise std of the
# slope falls with ~1/T^1.5, at the price of T/2 latency. With 1.5° landmark
# noise at 30 FPS this gives ≈1.5 deg/s std on a stationary joint.
VELOCITY_WINDOW_S = 0.40

# ---------------------------------------------------------------------------
# Phase-detection thresholds
# ---------------------------------------------------------------------------
# Hip angle = shoulder–hip–knee, knee angle = hip–knee–ankle (interior
# angles, 180° = fully extended). The hip is the primary driver because the
# deadlift is a hip hinge and it is the only joint that is flexed at the
# bottom of *every* variant (conventional, sumo, Romanian, stiff-leg); the
# knee is used only to confirm lockout.
#
# Typical values: bottom hip ≈ 60–110°, lockout hip/knee ≈ 165–180°.
#
# Position zones use *separate enter/exit thresholds* (hysteresis). The
# SETUP and LOCKOUT zones are ≥ 10° apart from each other and from the
# mid-range, so a lifter can never be in both at once and noise on the
# boundary cannot make the label flicker.
HIP_SETUP_ENTER_DEG = 130.0    # hip must be at/below this to be "at the bottom"
HIP_SETUP_EXIT_DEG = 140.0     # hip above this = definitely off the floor
HIP_LOCKOUT_ENTER_DEG = 160.0  # hip at/above this (and knee) = lockout zone
HIP_LOCKOUT_EXIT_DEG = 150.0   # hip below this = lockout lost
KNEE_LOCKOUT_DEG = 160.0       # knee must be this extended for lockout

# Hip angular-velocity threshold (deg/s, + = extension). |v| below this is
# "stationary", at/above it is "moving". ≈ 4-5 sigma of the residual
# velocity noise, yet well below real lifting speeds (tens of deg/s), so a
# slow grinding pull is still detected. Hysteresis comes from the
# displacement and dwell conditions below rather than a second threshold.
VELOCITY_MOVE_DEG_S = 8.0

# Angular displacement from the last turning point that must accompany a
# velocity trigger. Velocity alone is noise-prone; requiring the angle to
# actually have travelled makes direction changes physically meaningful.
ONSET_DISPLACEMENT_DEG = 4.0

# Dwell times: a condition must hold continuously for this long (seconds).
DWELL_ONSET_S = 0.10           # start of a concentric / eccentric movement
DWELL_SETTLE_S = 0.15          # movement has stopped at the bottom
DWELL_HOLD_S = 0.10            # stationary in lockout zone → isometric hold
DWELL_SETUP_S = 0.30          # standing → stable bottom position

# Robustness timeouts (seconds).
STALL_TIMEOUT_S = 5.0          # stationary mid-range → abandon the rep
TRACKING_LOSS_TIMEOUT_S = 1.0  # no valid pose for this long → STANDBY
MAX_SAMPLE_GAP_S = 0.5         # larger gap invalidates dwell timers/velocity


# ---------------------------------------------------------------------------
# Phase enum and display colours
# ---------------------------------------------------------------------------
class Phase(Enum):
    STANDBY = "Standby"
    SETUP = "Setup"
    CONCENTRIC = "Concentric"
    ISOMETRIC_HOLD = "Isometric Hold"
    ECCENTRIC = "Eccentric"


# BGR colours for the on-screen phase badge
PHASE_COLOURS = {
    Phase.STANDBY:        (120, 120, 120),   # dark grey — not in a lift
    Phase.SETUP:          (200, 200, 200),   # light grey — pre-pull / bottom
    Phase.CONCENTRIC:     (0, 200, 0),       # green  — ascending
    Phase.ISOMETRIC_HOLD: (0, 180, 255),     # amber  — holding at the top
    Phase.ECCENTRIC:      (0, 100, 255),     # red-orange — descending
}


# ---------------------------------------------------------------------------
# Geometry / landmark utilities
# ---------------------------------------------------------------------------
def calculate_angle(a: Tuple[float, float], b: Tuple[float, float],
                    c: Tuple[float, float]) -> float:
    """Interior joint angle (0–180°) at vertex *b*; NaN if degenerate.

    Uses atan2(|cross|, dot), which is numerically stable over the whole
    0–180° range (acos loses precision near 0° and 180°, exactly where the
    lockout lives).
    """
    v1x, v1y = a[0] - b[0], a[1] - b[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    if math.hypot(v1x, v1y) < 1e-6 or math.hypot(v2x, v2y) < 1e-6:
        return float("nan")
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    return math.degrees(math.atan2(abs(cross), dot))


def pixel_coords(landmark, w: int, h: int) -> Tuple[float, float]:
    """Normalised landmark → (x, y) pixel coordinates.

    Scaling x by width and y by height (rather than using normalised units)
    keeps angles correct on non-square frames.
    """
    return (landmark.x * w, landmark.y * h)


def landmark_reliable(landmark) -> bool:
    """True if MediaPipe is confident the landmark is visible and in-frame."""
    for attr in ("visibility", "presence"):
        value = getattr(landmark, attr, None)
        if value is not None and value < MIN_LANDMARK_VISIBILITY:
            return False
    return 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0


def extract_joint_angles(landmarks, w: int, h: int):
    """Return (hip_deg, knee_deg, hip_xy, knee_xy) or None if unreliable."""
    needed = (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    if any(not landmark_reliable(landmarks[i]) for i in needed):
        return None
    shoulder = pixel_coords(landmarks[RIGHT_SHOULDER], w, h)
    hip = pixel_coords(landmarks[RIGHT_HIP], w, h)
    knee = pixel_coords(landmarks[RIGHT_KNEE], w, h)
    ankle = pixel_coords(landmarks[RIGHT_ANKLE], w, h)
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    if not (math.isfinite(hip_angle) and math.isfinite(knee_angle)):
        return None
    return hip_angle, knee_angle, hip, knee


# ---------------------------------------------------------------------------
# Signal-conditioning building blocks (pure Python, timestamp-driven)
# ---------------------------------------------------------------------------
def _alpha(dt: float, cutoff_hz: float) -> float:
    """Smoothing factor of a 1st-order low-pass for sample period *dt*."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro adaptive low-pass filter driven by real timestamps."""

    def __init__(self, min_cutoff: float = FILTER_MIN_CUTOFF_HZ,
                 beta: float = FILTER_BETA,
                 d_cutoff: float = FILTER_D_CUTOFF_HZ):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._t: Optional[float] = None
        self._x_raw = 0.0
        self._x_hat = 0.0
        self._dx_hat = 0.0

    def __call__(self, t: float, x: float) -> float:
        if self._t is None:
            self._t, self._x_raw, self._x_hat = t, x, x
            return x
        dt = t - self._t
        if dt <= 0.0:
            return self._x_hat
        dx = (x - self._x_raw) / dt
        a_d = _alpha(dt, self._d_cutoff)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        cutoff = self._min_cutoff + self._beta * abs(self._dx_hat)
        a = _alpha(dt, cutoff)
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        self._t, self._x_raw = t, x
        return self._x_hat


class AngleChannel:
    """Outlier rejection + One-Euro filter for one joint angle.

    raw → 3-point median (removes any isolated single-frame glitch,
    whatever its size) → physiological jump gate (rejects multi-frame
    glitches) → One-Euro low-pass.
    """

    def __init__(self):
        self._filter = OneEuroFilter()
        self._recent: Deque[float] = deque(maxlen=3)
        self._last_ok: Optional[float] = None
        self._last_t: Optional[float] = None

    def update(self, t: float, raw: float) -> Optional[float]:
        """Filtered angle, or None if the sample is rejected as an outlier."""
        if not math.isfinite(raw):
            return None
        self._recent.append(raw)
        value = sorted(self._recent)[len(self._recent) // 2]
        if self._last_ok is not None and self._last_t is not None:
            dt = max(t - self._last_t, 1e-3)
            limit = MAX_ANGULAR_SPEED_DEG_S * dt + JUMP_SLACK_DEG
            # The limit grows with the time since the last *accepted* sample,
            # so a genuine sustained change is re-acquired within ~0.2 s.
            if abs(value - self._last_ok) > limit:
                return None
        self._last_ok, self._last_t = value, t
        return self._filter(t, value)


class SlidingSlope:
    """Least-squares slope (units/s) of samples inside a trailing time window.

    Works with irregular timestamps (variable frame rate, dropped frames).
    """

    def __init__(self, window_s: float = VELOCITY_WINDOW_S, min_samples: int = 4):
        self._window = window_s
        self._min_samples = min_samples
        self._buf: Deque[Tuple[float, float]] = deque()

    def clear(self) -> None:
        self._buf.clear()

    def add(self, t: float, x: float) -> None:
        self._buf.append((t, x))
        cutoff = t - self._window
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def slope(self) -> Optional[float]:
        n = len(self._buf)
        if n < self._min_samples:
            return None
        if self._buf[-1][0] - self._buf[0][0] < 0.6 * self._window:
            return None
        mt = sum(t for t, _ in self._buf) / n
        mx = sum(x for _, x in self._buf) / n
        den = sum((t - mt) ** 2 for t, _ in self._buf)
        if den <= 0.0:
            return None
        return sum((t - mt) * (x - mx) for t, x in self._buf) / den


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class DeadliftStateMachine:
    """
    Phase segmentation and rep counting from hip/knee angle time series.

    Zones (hip angle, with hysteresis):

        0° ──── SETUP zone ────130|140──── mid-range ────150|160──── LOCKOUT zone ──── 180°
                (enter ≤130, exit >140)             (enter ≥160 & knee, exit <150)

    Transitions (every one needs its condition to hold for a dwell time):

        STANDBY → SETUP
            hip ≤ 130° and not ascending, for DWELL_SETUP_S. Standing upright
            can therefore never be labelled Setup.
        SETUP → CONCENTRIC
            hip velocity ≥ +8°/s AND hip has risen ≥ 4° above the bottom
            turning point; or hip leaves the setup zone (> 140°).
        CONCENTRIC → ISOMETRIC_HOLD
            lockout zone (hip ≥ 160°, knee ≥ 160°) AND |v| < 8°/s.
            The hold can only be reached through a concentric phase.
        ISOMETRIC_HOLD → ECCENTRIC
            hip velocity ≤ −8°/s AND hip ≥ 4° below the top turning point;
            or hip drops below 150° (lockout lost).
        CONCENTRIC → ECCENTRIC
            same descent test as above (fast reversal or failed pull).
        ECCENTRIC → SETUP  (rep complete)
            hip ≤ 130° and no longer descending (v > −8°/s). rep_count
            increments only if the rep reached lockout.
        ECCENTRIC → CONCENTRIC
            mid-range reversal (hip > 130°): upward velocity + 4° rise from
            the lowest point reached. Does not count a rep.
        any moving phase → STANDBY
            stationary mid-range for STALL_TIMEOUT_S, or tracking lost for
            TRACKING_LOSS_TIMEOUT_S.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.phase: Phase = Phase.STANDBY
        self.rep_count: int = 0
        self.tracking_ok: bool = False
        self.transitions: List[Tuple[float, Phase, Phase]] = []

        self._hip_ch = AngleChannel()
        self._knee_ch = AngleChannel()
        self._slope = SlidingSlope()
        self._hip: Optional[float] = None
        self._knee: Optional[float] = None
        self._velocity: Optional[float] = None

        self._last_t: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._since: Dict[str, float] = {}
        self._turning_point = 0.0   # running min (SETUP/ECC) or max (CONC/HOLD)
        self._reached_lockout = False
        self._phase_start_t = 0.0

    # -- public API ---------------------------------------------------------
    def update(self, t: float, raw_hip: float, raw_knee: float) -> Phase:
        """Feed raw joint angles sampled at time *t* (seconds)."""
        if self._last_t is not None:
            if t <= self._last_t:
                return self.phase            # non-monotonic timestamp: ignore
            if t - self._last_t > MAX_SAMPLE_GAP_S:
                self._since.clear()          # stale dwell timers / velocity
                self._slope.clear()
        self._last_t = t

        hip = self._hip_ch.update(t, raw_hip)
        knee = self._knee_ch.update(t, raw_knee)
        if hip is None or knee is None:
            self.report_missing(t)           # rejected outlier == no data
            return self.phase

        self._last_valid_t = t
        self.tracking_ok = True
        self._hip, self._knee = hip, knee
        self._slope.add(t, hip)
        velocity = self._slope.slope()
        if velocity is None:                 # velocity window still warming up
            return self.phase
        self._velocity = velocity

        handler = {
            Phase.STANDBY: self._step_standby,
            Phase.SETUP: self._step_setup,
            Phase.CONCENTRIC: self._step_concentric,
            Phase.ISOMETRIC_HOLD: self._step_hold,
            Phase.ECCENTRIC: self._step_eccentric,
        }[self.phase]
        handler(t, hip, knee, velocity)
        return self.phase

    def report_missing(self, t: float) -> None:
        """Call when a frame yields no usable pose (dropout / low visibility)."""
        if self._last_valid_t is None:
            return
        missing = t - self._last_valid_t
        self.tracking_ok = missing <= 0.3
        if missing > TRACKING_LOSS_TIMEOUT_S and self.phase != Phase.STANDBY:
            self._reset_tracking(t)

    @property
    def hip_angle(self) -> float:
        return self._hip if self._hip is not None else 0.0

    @property
    def knee_angle(self) -> float:
        return self._knee if self._knee is not None else 0.0

    @property
    def hip_velocity(self) -> float:
        return self._velocity if self._velocity is not None else 0.0

    # -- helpers ------------------------------------------------------------
    def _sustained(self, key: str, cond: bool, t: float, dwell: float) -> bool:
        """True once *cond* has held continuously for *dwell* seconds."""
        if not cond:
            self._since.pop(key, None)
            return False
        return t - self._since.setdefault(key, t) >= dwell

    def _enter(self, new: Phase, t: float, hip: float) -> None:
        old = self.phase
        self.phase = new
        self._since.clear()
        self._turning_point = hip
        self._phase_start_t = t
        self.transitions.append((t, old, new))
        if self.verbose:
            print(f"[PHASE] t={t:7.2f}s  {old.value:>14} -> {new.value:<14}"
                  f" hip={hip:5.1f}  reps={self.rep_count}")

    def _reset_tracking(self, t: float) -> None:
        self._hip_ch = AngleChannel()
        self._knee_ch = AngleChannel()
        self._slope.clear()
        self._reached_lockout = False
        self.tracking_ok = False
        self._enter(Phase.STANDBY, t, self.hip_angle)

    # -- per-phase logic ----------------------------------------------------
    def _step_standby(self, t, hip, knee, v):
        at_bottom = hip <= HIP_SETUP_ENTER_DEG and v < VELOCITY_MOVE_DEG_S
        if self._sustained("setup", at_bottom, t, DWELL_SETUP_S):
            self._enter(Phase.SETUP, t, hip)

    def _step_setup(self, t, hip, knee, v):
        self._turning_point = min(self._turning_point, hip)   # bottom turning point
        rising = (v >= VELOCITY_MOVE_DEG_S
                  and hip - self._turning_point >= ONSET_DISPLACEMENT_DEG)
        left_zone = hip > HIP_SETUP_EXIT_DEG
        if left_zone or self._sustained("onset", rising, t, DWELL_ONSET_S):
            self._enter(Phase.CONCENTRIC, t, hip)

    def _in_lockout_zone(self, hip, knee) -> bool:
        return hip >= HIP_LOCKOUT_ENTER_DEG and knee >= KNEE_LOCKOUT_DEG

    def _descent_confirmed(self, t, hip, v) -> bool:
        descending = (v <= -VELOCITY_MOVE_DEG_S
                      and self._turning_point - hip >= ONSET_DISPLACEMENT_DEG)
        return self._sustained("descent", descending, t, DWELL_ONSET_S)

    def _step_concentric(self, t, hip, knee, v):
        self._turning_point = max(self._turning_point, hip)   # top turning point
        in_zone = self._in_lockout_zone(hip, knee)
        if self._sustained("lock_zone", in_zone, t, DWELL_ONSET_S):
            self._reached_lockout = True
        holding = in_zone and abs(v) < VELOCITY_MOVE_DEG_S
        if self._sustained("hold", holding, t, DWELL_HOLD_S):
            self._reached_lockout = True
            self._enter(Phase.ISOMETRIC_HOLD, t, hip)
        elif self._descent_confirmed(t, hip, v):
            self._enter(Phase.ECCENTRIC, t, hip)
        elif self._sustained("stall", abs(v) < VELOCITY_MOVE_DEG_S and not in_zone,
                             t, STALL_TIMEOUT_S):
            self._enter(Phase.STANDBY, t, hip)

    def _step_hold(self, t, hip, knee, v):
        self._turning_point = max(self._turning_point, hip)
        lost = self._sustained("lost", hip < HIP_LOCKOUT_EXIT_DEG, t, DWELL_ONSET_S)
        if lost or self._descent_confirmed(t, hip, v):
            self._enter(Phase.ECCENTRIC, t, hip)

    def _step_eccentric(self, t, hip, knee, v):
        self._turning_point = min(self._turning_point, hip)   # lowest point so far
        settled = hip <= HIP_SETUP_ENTER_DEG and v > -VELOCITY_MOVE_DEG_S
        if self._sustained("settle", settled, t, DWELL_SETTLE_S):
            if self._reached_lockout:
                self.rep_count += 1
                self._reached_lockout = False
            self._enter(Phase.SETUP, t, hip)
            return
        reversing = (hip > HIP_SETUP_ENTER_DEG and v >= VELOCITY_MOVE_DEG_S
                     and hip - self._turning_point >= ONSET_DISPLACEMENT_DEG)
        if self._sustained("reverse", reversing, t, DWELL_ONSET_S):
            self._enter(Phase.CONCENTRIC, t, hip)          # no rep counted
        elif self._sustained("stall", abs(v) < VELOCITY_MOVE_DEG_S, t, STALL_TIMEOUT_S):
            self._enter(Phase.STANDBY, t, hip)


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------
# NOTE: OpenCV's Hershey fonts are ASCII-only; a "°" glyph renders as "??",
# so angles are written with "deg".
def draw_hud(frame, sm: DeadliftStateMachine) -> None:
    """Overlay phase badge, rep counter, and smoothed angle readouts."""
    _, w = frame.shape[:2]
    colour = PHASE_COLOURS[sm.phase]

    cv2.putText(frame, sm.phase.value, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

    rep_text = f"Reps: {sm.rep_count}"
    (tw, _), _ = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, rep_text, (w - tw - 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(
        frame,
        f"Hip: {sm.hip_angle:.0f} deg ({sm.hip_velocity:+.0f} deg/s)"
        f"   Knee: {sm.knee_angle:.0f} deg",
        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )
    if not sm.tracking_ok:
        cv2.putText(frame, "Tracking lost - keep right side in view", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


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
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video source: {VIDEO_SOURCE!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    is_live = isinstance(VIDEO_SOURCE, int)
    frame_period_ms = 1000.0 / fps

    print(f"[INFO] Source: {VIDEO_SOURCE!r}  |  Nominal FPS: {fps:.1f}"
          f"  |  Mode: {'Live camera' if is_live else 'Video file'}")
    print("[INFO] Press Q to quit.\n")

    sm = DeadliftStateMachine()
    landmark_style = drawing_utils.DrawingSpec(
        color=(245, 117, 66), thickness=2, circle_radius=4)
    connection_style = drawing_utils.DrawingSpec(
        color=(245, 66, 230), thickness=2, circle_radius=2)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        last_ts_ms = -1
        t_start = time.monotonic()

        while True:
            loop_start = time.perf_counter()
            success, frame = cap.read()
            if not success:
                if not is_live:
                    print(f"\n[INFO] Video ended. Total reps: {sm.rep_count}")
                break

            # Frame time in seconds. Live: wall clock, so a camera that
            # delivers fewer frames than advertised is still timed correctly.
            # File: frame index / nominal FPS (assumes constant frame rate).
            t_s = (time.monotonic() - t_start) if is_live else frame_index / fps
            frame_index += 1
            # MediaPipe requires strictly increasing integer-ms timestamps.
            timestamp_ms = max(int(t_s * 1000.0), last_ts_ms + 1)
            last_ts_ms = timestamp_ms

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            joints = None
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                drawing_utils.draw_landmarks(
                    frame, lm, pose_connections,
                    landmark_drawing_spec=landmark_style,
                    connection_drawing_spec=connection_style,
                )
                joints = extract_joint_angles(lm, w, h)

            if joints is not None:
                hip_deg, knee_deg, hip_xy, knee_xy = joints
                sm.update(t_s, hip_deg, knee_deg)
                # Label the *filtered* angles next to the joints
                cv2.putText(
                    frame, f"{sm.hip_angle:.0f}",
                    (int(hip_xy[0]) - 50, int(hip_xy[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{sm.knee_angle:.0f}",
                    (int(knee_xy[0]) - 50, int(knee_xy[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                sm.report_missing(t_s)

            draw_hud(frame, sm)
            cv2.imshow("MyoMotion - Deadlift Tracker", frame)

            # File playback is paced to native FPS, net of processing time.
            if is_live:
                wait_ms = 1
            else:
                elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
                wait_ms = max(1, int(frame_period_ms - elapsed_ms))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
