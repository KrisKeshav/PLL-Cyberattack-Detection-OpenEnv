"""
validate_gains.py — KP/KI Gain Validation for PLL Attack Detection
====================================================================
Runs the full PLL simulation + heuristic agent across multiple KP/KI
values and all attack types. Reports settling time, detection accuracy,
and false-positive rate for each gain combination.

Usage:
    python validate_gains.py
"""

import sys, os, math, json
import numpy as np
from collections import deque

# ---- Inline PLL (parameterized KP/KI) ----

V_NOM = 1.0
F0 = 50.0
OMEGA0 = 2.0 * math.pi * F0
DT = 1e-3
WINDOW_SIZE = 20


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PLL:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.reset()

    def reset(self):
        self.t = 0.0
        self.theta_true = 0.0
        self.theta_hat = 0.0
        self.omega_hat = OMEGA0
        self.vq_integral = 0.0
        self.vd = 0.0
        self.vq = 0.0

    def step(self, attack_signal=0.0):
        va = V_NOM * math.sin(self.theta_true)
        vb = V_NOM * math.sin(self.theta_true - 2.0 * math.pi / 3.0)
        vc = V_NOM * math.sin(self.theta_true + 2.0 * math.pi / 3.0)
        va_m = va + attack_signal
        vb_m = vb
        v_alpha = va_m
        v_beta = (va_m + 2.0 * vb_m) / math.sqrt(3.0)
        cos_th = math.cos(self.theta_hat)
        sin_th = math.sin(self.theta_hat)
        vd = v_alpha * cos_th + v_beta * sin_th
        vq = -v_alpha * sin_th + v_beta * cos_th
        self.vq_integral += vq * DT
        omega_hat = OMEGA0 + self.kp * vq + self.ki * self.vq_integral
        self.theta_hat += omega_hat * DT
        self.theta_true += OMEGA0 * DT
        theta_err = wrap_angle(self.theta_hat - self.theta_true)
        self.t += DT
        self.vd = vd
        self.vq = vq
        self.omega_hat = omega_hat
        return {"vq": vq, "vd": vd, "omega_hat": omega_hat, "theta_err": theta_err}


# ---- Attack generators (from attacks.py) ----

def make_sinusoidal(amp, freq, phase):
    def gen(step, t):
        return amp * math.sin(2 * math.pi * freq * t + phase)
    return gen

def make_ramp(rate):
    def gen(step, t):
        return rate * step
    return gen

def make_pulse(mag, dur):
    def gen(step, t):
        return mag if step < dur else 0.0
    return gen

def make_stealthy(amp, drift_rate):
    state = {"delta": 0.0}
    def gen(step, t):
        state["delta"] += drift_rate * DT
        return amp * math.sin(2 * math.pi * F0 * t + state["delta"])
    return gen


# ---- Heuristic agent (simplified from inference.py) ----

class HeuristicDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vq_history = []
        self.omega_dev_history = []
        self.attack_detected = False
        self.settled_baseline = None
        self.peak_vq = 0.0

    def decide(self, vq_window, omega_dev_window, step, task_id):
        vq_abs = [abs(v) for v in vq_window]
        vq_mean = sum(vq_abs) / len(vq_abs)
        vq_max = max(vq_abs)
        omega_dev_abs = [abs(v) for v in omega_dev_window]
        omega_dev_mean = sum(omega_dev_abs) / len(omega_dev_abs)
        self.vq_history.append(vq_mean)
        self.omega_dev_history.append(omega_dev_mean)
        self.peak_vq = max(self.peak_vq, vq_mean)
        if step == 50:
            self.settled_baseline = omega_dev_mean

        # Task 0/1: vq-based detection
        if task_id in (0, 1):
            if step < 25:
                detected = False
            else:
                detected = vq_mean > 0.01 or vq_max > 0.025
            if detected:
                self.attack_detected = True
            return self.attack_detected

        # Task 2: stealthy — omega deviation drift
        if task_id == 2:
            if step > 50 and self.settled_baseline is not None:
                baseline = self.settled_baseline
                ratio = omega_dev_mean / baseline if baseline > 0.01 else omega_dev_mean * 100
                if len(self.omega_dev_history) > 10:
                    recent_10 = self.omega_dev_history[-10:]
                    old_10 = self.omega_dev_history[-20:-10] if len(self.omega_dev_history) > 20 else self.omega_dev_history[:10]
                    recent_avg = sum(recent_10) / len(recent_10)
                    old_avg = sum(old_10) / len(old_10)
                    rising = recent_avg > old_avg * 1.1
                else:
                    rising = False
                if ratio > 2.0 or (ratio > 1.3 and rising) or (rising and vq_mean > 0.1) or vq_mean > 0.2:
                    self.attack_detected = True
            return self.attack_detected

        return False


# ---- Validation runner ----

def measure_settling(pll, threshold=0.001, max_steps=200):
    """Run PLL with no attack and find when vq settles below threshold."""
    pll.reset()
    for s in range(max_steps):
        out = pll.step(0.0)
        if s > 5 and abs(out["vq"]) < threshold:
            # Check it stays settled for 10 more steps
            settled = True
            for _ in range(10):
                o2 = pll.step(0.0)
                if abs(o2["vq"]) >= threshold:
                    settled = False
                    break
            if settled:
                return s
    return max_steps


def run_episode(kp, ki, attack_fn, attack_start, task_id, max_steps=500):
    """
    Run one full episode and return detection metrics.
    Returns: (first_detect_step, n_correct_detect, n_missed, n_false_alarm, n_steps)
    """
    pll = PLL(kp, ki)
    det = HeuristicDetector()
    vq_win = deque(maxlen=WINDOW_SIZE)
    vd_win = deque(maxlen=WINDOW_SIZE)
    omega_dev_win = deque(maxlen=WINDOW_SIZE)

    # Warm-start
    for _ in range(WINDOW_SIZE):
        out = pll.step(0.0)
        vq_win.append(out["vq"])
        vd_win.append(out["vd"])
        omega_dev_win.append(out["omega_hat"] - OMEGA0)

    first_detect = None
    n_tp = 0  # true positive steps
    n_miss = 0  # missed detection steps
    n_fp = 0  # false positive steps
    n_tn = 0  # true negative steps

    for step in range(max_steps):
        attack_active = (step >= attack_start)
        if attack_active:
            steps_since = step - attack_start
            sig = attack_fn(steps_since, pll.t)
        else:
            sig = 0.0

        out = pll.step(sig)
        vq_win.append(out["vq"])
        vd_win.append(out["vd"])
        omega_dev_win.append(out["omega_hat"] - OMEGA0)

        detected = det.decide(list(vq_win), list(omega_dev_win), step, task_id)

        if attack_active:
            if detected:
                n_tp += 1
                if first_detect is None:
                    first_detect = step
            else:
                n_miss += 1
        else:
            if detected:
                n_fp += 1
            else:
                n_tn += 1

    return first_detect, n_tp, n_miss, n_fp, n_tn


def validate_gains(kp, ki, n_trials=20):
    """Run multiple trials for each attack type and report metrics."""
    rng = np.random.default_rng(42)
    results = {}

    attack_configs = [
        ("sinusoidal", 0, lambda r: make_sinusoidal(
            r.uniform(0.05, 0.20), r.uniform(5.0, 20.0), r.uniform(0, 2*math.pi))),
        ("ramp", 1, lambda r: make_ramp(r.uniform(0.0002, 0.001))),
        ("pulse", 1, lambda r: make_pulse(r.uniform(0.1, 0.3), int(r.integers(20, 81)))),
        ("stealthy", 2, lambda r: make_stealthy(0.03, r.uniform(0.05, 0.2))),
    ]

    for atk_name, task_id, atk_factory in attack_configs:
        detect_rates = []
        false_alarm_rates = []
        detect_delays = []

        for _ in range(n_trials):
            attack_start = int(rng.integers(30, 81))
            atk_fn = atk_factory(rng)
            first_det, n_tp, n_miss, n_fp, n_tn = run_episode(
                kp, ki, atk_fn, attack_start, task_id)

            total_attack_steps = n_tp + n_miss
            total_safe_steps = n_fp + n_tn
            dr = n_tp / max(1, total_attack_steps)
            far = n_fp / max(1, total_safe_steps)
            detect_rates.append(dr)
            false_alarm_rates.append(far)
            if first_det is not None:
                detect_delays.append(first_det - attack_start)

        avg_dr = sum(detect_rates) / len(detect_rates)
        avg_far = sum(false_alarm_rates) / len(false_alarm_rates)
        avg_delay = sum(detect_delays) / len(detect_delays) if detect_delays else float('inf')
        pct_detected = len(detect_delays) / n_trials * 100

        results[atk_name] = {
            "detect_rate": avg_dr,
            "false_alarm_rate": avg_far,
            "avg_delay_steps": avg_delay,
            "pct_episodes_detected": pct_detected,
        }

    return results


def main():
    # Gain combinations to test
    gain_sets = [
        (50.0,  1500.0, "Current (zeta=0.645, t_s=160ms)"),
        (266.0, 17689.0, "Ultra Fast (zeta=1.0, t_s=30ms)"),
        (500.0, 62500.0, "Instant (zeta=1.0, t_s=16ms)"),
    ]

    print("=" * 90)
    print("PLL KP/KI Gain Validation - Attack Detection Accuracy")
    print("=" * 90)

    for kp, ki, label in gain_sets:
        pll = PLL(kp, ki)
        settling = measure_settling(pll)
        omega_n = math.sqrt(ki)
        zeta = kp / (2 * omega_n)

        print(f"\n{'-' * 90}")
        print(f"  KP={kp:.0f}  KI={ki:.0f}  |  omega_n={omega_n:.1f} rad/s  zeta={zeta:.3f}  "
              f"settling={settling} steps  |  {label}")
        print(f"{'-' * 90}")

        results = validate_gains(kp, ki, n_trials=30)

        print(f"  {'Attack Type':<14} {'Detect Rate':>12} {'FA Rate':>10} {'Avg Delay':>12} {'Episodes Det':>14}")
        for atk, m in results.items():
            det_str = f"{m['detect_rate']*100:.1f}%"
            fa_str = f"{m['false_alarm_rate']*100:.1f}%"
            delay_str = f"{m['avg_delay_steps']:.0f} steps" if m['avg_delay_steps'] != float('inf') else "never"
            ep_str = f"{m['pct_episodes_detected']:.0f}%"
            print(f"  {atk:<14} {det_str:>12} {fa_str:>10} {delay_str:>12} {ep_str:>14}")

    # Summary recommendation
    print(f"\n{'=' * 90}")
    print("RECOMMENDATION")
    print("=" * 90)
    print("Choose gains that maximize detection rate + episodes detected")
    print("while keeping false alarm rate at 0% and settling time < 30 steps.")
    print("=" * 90)


if __name__ == "__main__":
    main()
