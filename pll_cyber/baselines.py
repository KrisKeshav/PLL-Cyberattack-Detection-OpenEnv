"""
Reference policies.

`PhysicsHeuristicAgent` is a transparent, physics-informed baseline that a
protection engineer could write by hand. It works on plain observation
dicts, so the same code drives in-process, HTTP and WebSocket rollouts.

  * Single-phase injection (tasks 0/1). A genuine grid has no DC component
    and its only zero-sequence content comes from single-phase faults, which
    sit at the 50 Hz fundamental. The agent fits v0 = (va+vb+vc)/3 with
    DC + 50 Hz terms:
        DC                   → bias attack (ramp if growing, pulse if not)
        residual off 50 Hz   → sinusoidal interharmonic injection
        50 Hz only           → single-phase sag (benign)
  * Phase spoofing (task 2). The local frequency is compared with the PMU
    frequency at the *same* instant (latency-compensated using the agent's
    own history of local frequency). The mismatch is integrated into an
    estimate of the accumulated angle error; holdover is engaged before it
    approaches the 10° limit.

Also included: trivial policies used by tests to prove the graders cannot
be gamed.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

WAVE_DT_S = 0.002          # exported waveform resolution
F0 = 50.0


def _v0_decompose(obs: Dict[str, Any]) -> Dict[str, float]:
    w = obs["waveforms"]
    v0 = (np.asarray(w["va"]) + np.asarray(w["vb"]) + np.asarray(w["vc"])) / 3.0
    t = np.arange(v0.size) * WAVE_DT_S
    X = np.column_stack([np.ones_like(t), np.cos(2 * np.pi * F0 * t), np.sin(2 * np.pi * F0 * t)])
    coef, *_ = np.linalg.lstsq(X, v0, rcond=None)
    resid = v0 - X @ coef
    return {"dc": float(coef[0]), "amp50": float(math.hypot(coef[1], coef[2])),
            "resid_rms": float(np.sqrt(np.mean(resid ** 2)))}


class PhysicsHeuristicAgent:
    # Thresholds (pu unless noted); noise floor of v0 residual is ≈0.003 pu
    RESID_THR = 0.0075
    DC_THR = 0.012
    SPOOF_ANGLE_ALARM_DEG = 2.5
    SPOOF_ANGLE_HOLDOVER_DEG = 3.5
    SPOOF_RESID_HZ = 0.025
    CONFIRM = 2

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.latched = False
        self.latched_type = 0
        self.streak = 0
        self.dc_hist: List[float] = []
        self.resid_hist: List[float] = []
        self.pulse_ended = False
        self.f_hist: List[tuple] = []        # (t_center_s, f_local)
        self.last_pmu: Optional[tuple] = None
        self.angle_est_deg = 0.0
        self.spoof_streak = 0

    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if obs.get("step", 0) == 0:
            self.reset()
        task = obs["task_id"]
        if task == 2:
            return self._act_spoofing(obs)
        return self._act_injection(obs, task)

    # ------------------------------------------------------------------
    def _act_injection(self, obs: Dict[str, Any], task: int) -> Dict[str, Any]:
        d = _v0_decompose(obs)
        thr_dc = self.DC_THR / 3.0                  # v0 carries one third of a single-phase offset
        bias_now = abs(d["dc"]) > thr_dc
        resid_now = d["resid_rms"] > self.RESID_THR
        hit = bias_now or resid_now

        self.streak = self.streak + 1 if hit else 0
        if hit or self.latched:
            self.dc_hist.append(d["dc"])
            self.resid_hist.append(d["resid_rms"])
        if not self.latched and self.streak >= self.CONFIRM:
            self.latched = True
        if self.latched and task == 1:
            self.latched_type = self._classify(bias_now)

        if not self.latched:
            conf = 0.9 if self.streak == 0 else 0.6
            return {"attack_detected": False, "attack_type": 0, "confidence": conf, "protective_action": 0}
        atype = 1 if task == 0 else self.latched_type
        return {"attack_detected": True, "attack_type": atype, "confidence": 0.9, "protective_action": 1}

    def _classify(self, bias_now: bool) -> int:
        """sinusoidal ↔ bias (ramp/pulse) from the v0 history since the alarm started."""
        if self.latched_type == 3 and self.pulse_ended:
            return 3
        thr_dc = self.DC_THR / 3.0
        active = [x for x in self.dc_hist if abs(x) > thr_dc]
        signs = {np.sign(x) for x in active}
        if len(signs) > 1:
            return 1                                 # offset changes sign → oscillation, not a bias
        if np.median(self.resid_hist) > self.RESID_THR:
            return 1
        if not bias_now and len(active) >= 1:
            self.pulse_ended = True                  # a bias that disappears was a pulse
            return 3
        mags = [abs(x) for x in active]
        if len(mags) >= 2:
            growth = mags[-1] - mags[0]
            if growth > 0.25 * max(mags):
                return 2
            if self.latched_type == 2 and len(mags) > 4:
                return 2                             # ramp that has reached its saturation cap
            return 3
        return self.latched_type or 2

    # ------------------------------------------------------------------
    def _act_spoofing(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        f = obs["features"]
        pmu = obs["pmu"]
        t_now = obs["time_s"]
        interval = obs.get("decision_interval_ms", 50.0) / 1000.0
        self.f_hist.append((t_now - interval / 2.0, f["f_local_hz"]))
        self.f_hist = self.f_hist[-20:]

        # Only learn from a *new* PMU report
        t_rep = t_now - pmu["age_ms"] / 1000.0 - 0.05     # report = average over its 100 ms interval
        new_report = self.last_pmu is None or abs(t_rep - self.last_pmu[0]) > 1e-6
        if new_report and len(self.f_hist) >= 3 and obs["inverter_mode"] == "normal":
            ts = np.array([x[0] for x in self.f_hist])
            fs = np.array([x[1] for x in self.f_hist])
            if ts[0] <= t_rep:
                # average local frequency over the same 100 ms interval the PMU measured
                grid = np.linspace(t_rep - 0.05, t_rep + 0.05, 5)
                f_loc = float(np.interp(grid, ts, fs).mean())
                resid = f_loc - pmu["f_hz"]
                dt = 0.1 if self.last_pmu is None else max(0.05, t_rep - self.last_pmu[0])
                self.angle_est_deg = 0.9 * self.angle_est_deg + 360.0 * resid * dt
                same_sign = np.sign(resid) == np.sign(self.angle_est_deg)
                if abs(resid) > self.SPOOF_RESID_HZ and same_sign:
                    self.spoof_streak += 1
                else:
                    self.spoof_streak = 0
        if new_report:
            self.last_pmu = (t_rep, pmu["f_hz"])

        angle = abs(self.angle_est_deg)
        if not self.latched and (angle > self.SPOOF_ANGLE_ALARM_DEG or self.spoof_streak >= 2):
            self.latched = True
        if not self.latched:
            return {"attack_detected": False, "attack_type": 0, "confidence": 0.8, "protective_action": 0}
        holdover = angle > self.SPOOF_ANGLE_HOLDOVER_DEG or self.spoof_streak >= 2 or obs["inverter_mode"] == "holdover"
        return {"attack_detected": True, "attack_type": 4, "confidence": 0.85,
                "protective_action": 2 if holdover else 1}


# ---------------------------------------------------------------------------
# Trivial policies (exploit checks)
# ---------------------------------------------------------------------------

class ConstantAgent:
    def __init__(self, detected: bool, attack_type: int = 0, protective_action: int = 0):
        self.a = {"attack_detected": detected, "attack_type": attack_type,
                  "confidence": 1.0, "protective_action": protective_action}

    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self.a)


class RandomAgent:
    def __init__(self, seed: int = 0, p_detect: float = 0.5):
        self.rng = random.Random(seed)
        self.p = p_detect

    def reset(self) -> None:
        pass

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        d = self.rng.random() < self.p
        return {"attack_detected": d, "attack_type": self.rng.randint(0, 4) if d else 0,
                "confidence": self.rng.random(), "protective_action": self.rng.choice([0, 0, 1, 2])}
