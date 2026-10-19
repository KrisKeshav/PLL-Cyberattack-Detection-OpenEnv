"""
Task definitions and seeded scenario sampling.

Every episode is fully determined by (task_id, seed): attack type/onset and
parameters, benign grid events, harmonics, PMU latency and all noise streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .attacks import AttackSpec
from .grid import GridConfig, GridEvent

# ---------------------------------------------------------------------------
# Timing (shared by all tasks)
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 5000.0
DT = 1.0 / SAMPLE_RATE_HZ
DECISION_INTERVAL_S = 0.050          # 2.5 fundamental cycles per agent decision
SAMPLES_PER_DECISION = int(round(DECISION_INTERVAL_S * SAMPLE_RATE_HZ))
WARMUP_S = 0.200                     # PLL settles before the agent's first observation
MAX_STEPS = 40                       # 2.0 s monitored horizon

PMU_REPORT_INTERVAL_S = 0.100
PMU_F_NOISE_HZ = 0.01
PMU_V_NOISE_PU = 0.003
SENSOR_NOISE_PU = 0.005
ADC_LSB_PU = 4.0 / 4096.0            # 12-bit ADC over ±2 pu


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    difficulty: str
    attack_probability: float
    brief: str


TASKS: Dict[int, TaskSpec] = {
    0: TaskSpec(
        0, "sinusoidal_fdi_detection", "easy", 0.80,
        "Monitor the inverter's PLL. A sinusoidal false-data-injection attack may start on one phase "
        "(some episodes are attack-free). Raise attack_detected=True as soon as the attack is present "
        "and keep it raised; never raise it before. Grid background harmonics and occasional mild sags "
        "are normal operation.",
    ),
    1: TaskSpec(
        1, "attack_classification_under_disturbances", "medium", 0.80,
        "A sinusoidal (1), ramp-bias (2) or pulse-bias (3) injection may start on one phase, while genuine "
        "grid events (phase jumps, voltage sags, single-phase faults, harmonic steps, frequency events) also "
        "occur. Detect the attack quickly, report its attack_type, and do NOT alarm on natural grid events. "
        "Once an attack has started the system counts as compromised for the rest of the episode.",
    ),
    2: TaskSpec(
        2, "stealthy_phase_spoofing_mitigation", "hard", 0.80,
        "A coordinated phase-spoofing attack (4) may slowly rotate the measured voltage angle. The PLL "
        "follows it with almost no residual, and the local frequency drifts like a natural grid frequency "
        "event. Use the trusted but delayed PMU stream to tell them apart. Detect it and engage "
        "protective_action=2 (PMU-aided holdover) before the inverter drifts more than 10 deg out of sync "
        "with the real grid. Holdover or disconnect (3) on a healthy grid is costly and can itself cause "
        "desynchronisation.",
    ),
}


@dataclass
class Scenario:
    task_id: int
    seed: int
    grid: GridConfig
    attack: AttackSpec
    onset_step: Optional[int]          # decision index at which the attack begins, None if attack-free
    pmu_latency_s: float
    theta0: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": TASKS[self.task_id].name,
            "seed": self.seed,
            "attack": self.attack.to_dict(),
            "onset_step": self.onset_step,
            "pmu_latency_ms": round(self.pmu_latency_s * 1000.0, 1),
            "grid": self.grid.to_dict(),
        }


def step_start_time(step: int) -> float:
    """Absolute sim time at which decision window `step` begins (window 0 = last warm-up window)."""
    return WARMUP_S + (step - 1) * DECISION_INTERVAL_S


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sign(rng: np.random.Generator) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def _benign_event(rng: np.random.Generator, kind: str, t: float, strong: bool) -> GridEvent:
    if kind == "phase_jump":
        return GridEvent(kind, t, {"angle_deg": _sign(rng) * _u(rng, 5.0, 25.0 if strong else 12.0)})
    if kind == "voltage_sag":
        return GridEvent(kind, t, {"depth_pu": _u(rng, 0.5 if strong else 0.8, 0.9), "duration_s": _u(rng, 0.06, 0.25)})
    if kind == "single_phase_sag":
        return GridEvent(kind, t, {"phase": int(rng.integers(0, 3)), "depth_pu": _u(rng, 0.3, 0.8), "duration_s": _u(rng, 0.06, 0.20)})
    if kind == "harmonic_step":
        return GridEvent(kind, t, {"gain": _u(rng, 1.5, 2.5)})
    if kind == "frequency_event":
        return GridEvent(kind, t, {"rocof_hz_s": _u(rng, 0.1, 0.6), "delta_f_hz": _sign(rng) * _u(rng, 0.05, 0.25)})
    raise ValueError(kind)


def _injection_attack(rng: np.random.Generator, kind: str, t: float, easy: bool) -> AttackSpec:
    phase = int(rng.integers(0, 3))
    if kind == "sinusoidal":
        lo, hi = (0.05, 0.15) if easy else (0.03, 0.10)
        p = {"amplitude_pu": _u(rng, lo, hi), "freq_hz": _u(rng, 5.0, 25.0),
             "phase": _u(rng, 0.0, 2 * math.pi), "phase_index": phase}
    elif kind == "ramp":
        p = {"rate_pu_s": _sign(rng) * _u(rng, 0.08, 0.30), "cap_pu": 0.30, "phase_index": phase}
    elif kind == "pulse":
        p = {"magnitude_pu": _sign(rng) * _u(rng, 0.08, 0.25), "duration_s": _u(rng, 0.10, 0.30), "phase_index": phase}
    else:
        raise ValueError(kind)
    return AttackSpec(kind, t, p)


def sample_scenario(task_id: int, seed: int) -> Scenario:
    if task_id not in TASKS:
        raise ValueError(f"task_id must be one of {sorted(TASKS)}")
    task = TASKS[task_id]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1000 + task_id]))

    grid = GridConfig(
        f_drift_amp_hz=_u(rng, 0.005, 0.02 if task_id < 2 else 0.03),
        f_drift_period_s=_u(rng, 4.0, 10.0),
        f_drift_phase=_u(rng, 0.0, 2 * math.pi),
        h5_pu=_u(rng, 0.01, 0.04),
        h7_pu=_u(rng, 0.005, 0.025),
        h5_phase=_u(rng, 0.0, 2 * math.pi),
        h7_phase=_u(rng, 0.0, 2 * math.pi),
        phase_unbalance=[_u(rng, 0.99, 1.01) for _ in range(3)],
    )
    attacked = rng.random() < task.attack_probability
    pmu_latency = _u(rng, 0.04, 0.10)
    t_end = step_start_time(MAX_STEPS)

    if task_id == 0:
        onset = int(rng.integers(6, 21)) if attacked else None
        if rng.random() < 0.3:
            grid.events.append(_benign_event(rng, "voltage_sag", step_start_time(int(rng.integers(2, MAX_STEPS - 4))), strong=False))
        attack = _injection_attack(rng, "sinusoidal", step_start_time(onset), easy=True) if attacked else AttackSpec()

    elif task_id == 1:
        onset = int(rng.integers(10, 25)) if attacked else None
        kinds = ["phase_jump", "voltage_sag", "single_phase_sag", "harmonic_step", "frequency_event"]
        n_events = int(rng.integers(1, 3)) + (0 if attacked else 1)
        # Guarantee a benign event in the clean part of the episode, so false-alarm
        # resistance is always tested.
        last_clean = (onset - 1) if attacked else MAX_STEPS - 4
        times = [step_start_time(int(rng.integers(2, max(3, last_clean))))]
        times += [step_start_time(int(rng.integers(2, MAX_STEPS - 4))) for _ in range(n_events - 1)]
        for t in times:
            grid.events.append(_benign_event(rng, str(rng.choice(kinds)), t + _u(rng, 0.0, DECISION_INTERVAL_S), strong=True))
        if attacked:
            kind = ["sinusoidal", "ramp", "pulse"][int(rng.integers(0, 3))]
            attack = _injection_attack(rng, kind, step_start_time(onset), easy=False)
        else:
            attack = AttackSpec()

    else:
        onset = int(rng.integers(6, 19)) if attacked else None
        # Natural frequency events are the main confounder for phase spoofing.
        n_freq = 1 if attacked else int(rng.integers(1, 3))
        for _ in range(n_freq):
            k = int(rng.integers(2, (onset + 4) if attacked else MAX_STEPS - 8))
            grid.events.append(_benign_event(rng, "frequency_event", step_start_time(k) + _u(rng, 0.0, DECISION_INTERVAL_S), strong=True))
        if rng.random() < 0.6:
            kind = ["voltage_sag", "single_phase_sag", "harmonic_step"][int(rng.integers(0, 3))]
            grid.events.append(_benign_event(rng, kind, step_start_time(int(rng.integers(2, MAX_STEPS - 4))), strong=False))
        if attacked:
            attack = AttackSpec("phase_spoofing", step_start_time(onset), {
                "rocof_hz_s": _u(rng, 0.12, 0.45),
                "max_offset_hz": _u(rng, 0.20, 0.40),
                "direction": _sign(rng),
            })
        else:
            attack = AttackSpec()

    grid.events.sort(key=lambda e: e.t_start)
    grid.events = [e for e in grid.events if e.t_start < t_end]
    return Scenario(task_id=task_id, seed=seed, grid=grid, attack=attack,
                    onset_step=onset, pmu_latency_s=pmu_latency)
