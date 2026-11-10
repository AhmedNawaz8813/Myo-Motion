"""
Synthetic-signal tests for the deadlift phase state machine.

No camera, model or video is needed: joint-angle time series are generated
with known ground-truth phase boundaries, corrupted with realistic landmark
noise, and fed through DeadliftStateMachine.

Run:  python -m pytest tests/ -q
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "cv", "deadlift"))

import deadlift_tracker as dt  # noqa: E402
from deadlift_tracker import DeadliftStateMachine, Phase  # noqa: E402

P = Phase


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def smoothstep(x):
    """Minimum-jerk profile 0→1."""
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def rep_profile(bottom=80.0, top=175.0, knee_bottom=115.0, knee_top=178.0,
                pre=1.0, up=2.0, hold=0.8, down=2.0, post=1.0, stand_first=0.0):
    """Return (duration, f(t) -> (hip, knee), truth) for one rep.

    truth: list of (t_start, Phase) ground-truth segments.
    """
    t_up0 = stand_first + pre
    t_hold0 = t_up0 + up
    t_down0 = t_hold0 + hold
    t_end_down = t_down0 + down
    total = t_end_down + post

    def f(t):
        if t < stand_first:
            return top, knee_top
        if t < t_up0:
            return bottom, knee_bottom
        if t < t_hold0:
            s = smoothstep((t - t_up0) / up)
        elif t < t_down0:
            s = 1.0
        elif t < t_end_down:
            s = 1.0 - smoothstep((t - t_down0) / down)
        else:
            s = 0.0
        return (bottom + s * (top - bottom), knee_bottom + s * (knee_top - knee_bottom))

    truth = []
    if stand_first:
        truth.append((0.0, P.STANDBY))
        # lifter walks to bar: treated as standby until in setup posture
    truth += [(stand_first, P.SETUP), (t_up0, P.CONCENTRIC), (t_hold0, P.ISOMETRIC_HOLD),
              (t_down0, P.ECCENTRIC), (t_end_down, P.SETUP)]
    return total, f, truth


def run(f, total, fps=30.0, noise=1.5, seed=0, spikes=0.0, dropout=None,
        jitter=0.0, start=0.0):
    """Feed f(t) through a state machine; return (sm, timeline)."""
    rng = random.Random(seed)
    sm = DeadliftStateMachine(verbose=False)
    timeline = []
    n = int(total * fps)
    for i in range(n):
        t = start + i / fps + (rng.uniform(-jitter, jitter) / fps if i else 0.0)
        if dropout and dropout[0] <= t - start < dropout[1]:
            sm.report_missing(t)
            timeline.append((t, sm.phase))
            continue
        hip, knee = f(t - start)
        hip += rng.gauss(0, noise)
        knee += rng.gauss(0, noise)
        if spikes and rng.random() < spikes:
            hip += rng.choice([-1, 1]) * rng.uniform(40, 90)   # landmark glitch
        sm.update(t, hip, knee)
        timeline.append((t, sm.phase))
    return sm, timeline


def phase_sequence(sm):
    return [new for _, _, new in sm.transitions]


EXPECTED_ONE_REP = [P.SETUP, P.CONCENTRIC, P.ISOMETRIC_HOLD, P.ECCENTRIC, P.SETUP]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_calculate_angle_cardinal_cases():
    assert math.isclose(dt.calculate_angle((1, 0), (0, 0), (0, 1)), 90.0)
    assert math.isclose(dt.calculate_angle((1, 0), (0, 0), (-1, 0)), 180.0)
    assert math.isclose(dt.calculate_angle((1, 0), (0, 0), (1, 0)), 0.0)
    assert math.isnan(dt.calculate_angle((0, 0), (0, 0), (1, 0)))


def test_label_is_isometric_hold_not_setup():
    assert P.ISOMETRIC_HOLD.value == "Isometric Hold"
    assert P.SETUP.value == "Setup"
    assert not hasattr(P, "LOCKOUT")


def test_single_rep_exact_phase_order_and_count():
    total, f, _ = rep_profile()
    sm, _ = run(f, total)
    assert phase_sequence(sm) == EXPECTED_ONE_REP
    assert sm.rep_count == 1


def test_no_flicker_across_frame_rates_and_noise():
    """Exactly 5 transitions (no extra flips) for every fps / noise / seed."""
    total, f, _ = rep_profile()
    for fps in (15, 24, 30, 60):
        for noise in (0.5, 1.5, 3.0):
            for seed in range(5):
                sm, _ = run(f, total, fps=fps, noise=noise, seed=seed, jitter=0.3)
                assert phase_sequence(sm) == EXPECTED_ONE_REP, (fps, noise, seed)
                assert sm.rep_count == 1


def test_phase_boundaries_close_to_ground_truth():
    total, f, truth = rep_profile()
    sm, _ = run(f, total, fps=30, noise=1.5)
    # skip the leading SETUP entry (STANDBY dwell), compare the rest
    for (t_true, ph), (t_det, _, new) in zip(truth[1:], sm.transitions[1:]):
        assert new == ph
        # Detection is causal (velocity window + dwell); lag must stay bounded.
        assert -0.1 <= t_det - t_true <= 0.9, (ph, t_true, t_det)


def test_standing_start_is_standby_never_setup_or_hold():
    """Upright for the whole clip must never be labelled Setup / Isometric Hold."""
    sm, timeline = run(lambda t: (175.0, 178.0), 10.0, noise=2.0)
    assert {p for _, p in timeline} == {P.STANDBY}
    assert sm.transitions == [] and sm.rep_count == 0


def test_lifter_stands_then_goes_to_bar():
    """Standing → walks down to a bent-over setup (smoothly) → one full rep."""
    total_rep, f_rep, _ = rep_profile(pre=1.5)

    def f(t):
        if t < 3.0:
            return 175.0, 178.0
        if t < 4.5:                                   # hinge down to the bar
            s = smoothstep((t - 3.0) / 1.5)
            return 175.0 + s * (80.0 - 175.0), 178.0 + s * (115.0 - 178.0)
        return f_rep(t - 4.5)

    sm, timeline = run(f, 4.5 + total_rep)
    assert {p for t, p in timeline if t < 3.0} == {P.STANDBY}
    assert sm.rep_count == 1
    assert P.ISOMETRIC_HOLD in phase_sequence(sm)


def test_setup_and_hold_are_never_confused():
    """Phase must match the angle zone at every frame of many noisy reps."""
    total, f, _ = rep_profile()
    for seed in range(20):
        sm = DeadliftStateMachine(verbose=False)
        rng = random.Random(seed)
        fps = 30.0
        for i in range(int(total * fps)):
            t = i / fps
            hip, knee = f(t)
            sm.update(t, hip + rng.gauss(0, 2.0), knee + rng.gauss(0, 2.0))
            if sm.phase == P.SETUP:
                assert sm.hip_angle <= dt.HIP_SETUP_EXIT_DEG + 1.0
            if sm.phase == P.ISOMETRIC_HOLD:
                assert sm.hip_angle >= dt.HIP_LOCKOUT_EXIT_DEG - 1.0


def test_multiple_reps_counted():
    total1, f1, _ = rep_profile()
    n = 4

    def f(t):
        return f1(t % total1)

    sm, _ = run(f, total1 * n)
    assert sm.rep_count >= n - 1
    seq = phase_sequence(sm)
    assert seq[:5] == EXPECTED_ONE_REP


def test_all_reps_counted_when_ending_in_setup():
    total1, f1, _ = rep_profile()
    n = 3

    def f(t):
        return f1(t % total1)

    sm, _ = run(f, total1 * n + 0.5)
    assert sm.rep_count == n


def test_outlier_spikes_do_not_cause_transitions():
    total, f, _ = rep_profile()
    for seed in range(10):
        sm, _ = run(f, total, spikes=0.03, seed=seed)
        assert phase_sequence(sm) == EXPECTED_ONE_REP, seed
        assert sm.rep_count == 1


def test_short_dropout_during_hold_is_ignored():
    total, f, _ = rep_profile(hold=1.5)
    # hold spans t=4..5.5; drop 0.4 s of frames in the middle
    sm, _ = run(f, total, dropout=(4.5, 4.9))
    assert phase_sequence(sm) == EXPECTED_ONE_REP
    assert sm.rep_count == 1


def test_long_dropout_resets_to_standby_and_recovers():
    total, f, _ = rep_profile(up=2.0, hold=3.0)
    sm, _ = run(f, total, dropout=(3.5, 5.5))   # 2 s loss mid-concentric / hold
    assert P.STANDBY in phase_sequence(sm)


def test_failed_rep_is_not_counted():
    """Pull to mid-range, give up and lower: no lockout → no rep."""
    def f(t):
        if t < 1.0:
            return 80.0, 115.0
        if t < 2.5:
            return 80 + 45 * smoothstep((t - 1.0) / 1.5), 130.0   # up to 125°
        if t < 3.0:
            return 125.0, 130.0
        if t < 4.5:
            return 125 - 45 * smoothstep((t - 3.0) / 1.5), 130.0
        return 80.0, 115.0
    sm, _ = run(f, 6.0)
    assert sm.rep_count == 0


def test_romanian_deadlift_knee_stays_extended():
    """Stiff-knee variant: knee never goes below ~160° but hip hinges deeply."""
    total, f, _ = rep_profile(bottom=85.0, knee_bottom=162.0, knee_top=178.0)
    sm, _ = run(f, total)
    assert phase_sequence(sm) == EXPECTED_ONE_REP
    assert sm.rep_count == 1


def test_fast_reversal_without_pause_still_counts():
    total, f, _ = rep_profile(hold=0.0)
    sm, _ = run(f, total)
    assert sm.rep_count == 1
    assert P.CONCENTRIC in phase_sequence(sm) and P.ECCENTRIC in phase_sequence(sm)


def test_non_monotonic_timestamps_ignored():
    sm = DeadliftStateMachine(verbose=False)
    sm.update(1.0, 80.0, 115.0)
    before = sm.phase
    sm.update(0.9, 170.0, 178.0)
    assert sm.phase == before


def test_velocity_sign_convention():
    sm = DeadliftStateMachine(verbose=False)
    for i in range(30):
        t = i / 30.0
        sm.update(t, 80.0 + 30.0 * t, 115.0)     # +30 deg/s extension
    assert 20.0 < sm.hip_velocity < 40.0
