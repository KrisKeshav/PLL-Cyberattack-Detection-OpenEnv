"""
True (physical) grid voltage at the inverter's point of common coupling.

The grid is not a perfect 50 Hz source. Besides slow frequency wander and
background 5th/7th harmonics, an episode may contain *benign* power-system
events that disturb the PLL exactly like an attack would at first glance:

    frequency_event     — generation/load imbalance: RoCoF ramp to a new frequency
    phase_jump          — fault clearing / line switching: step in voltage angle
    voltage_sag         — balanced sag (e.g. remote three-phase fault, motor start)
    single_phase_sag    — single-line-to-ground fault: one phase dips (creates
                          zero/negative sequence, like a single-phase FDI attack)
    harmonic_step       — non-linear load switching: harmonic distortion rises

These are what make cyber-attack detection on PLLs hard in practice: an alarm
on every grid event is useless to an operator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .pll import F_NOMINAL, TWO_PI

PHASE_SHIFTS = np.array([0.0, -TWO_PI / 3.0, TWO_PI / 3.0])  # a, b, c


@dataclass
class GridEvent:
    kind: str
    t_start: float
    params: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GridConfig:
    v_nominal: float = 1.0
    f_drift_amp_hz: float = 0.01        # slow frequency wander amplitude
    f_drift_period_s: float = 6.0
    f_drift_phase: float = 0.0
    h5_pu: float = 0.02                 # 5th harmonic (negative sequence)
    h7_pu: float = 0.01                 # 7th harmonic (positive sequence)
    h5_phase: float = 0.0
    h7_phase: float = 0.0
    phase_unbalance: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    events: List[GridEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["events"] = [e.to_dict() for e in self.events]
        return d


class GridModel:
    """Generates true three-phase voltages block-by-block."""

    def __init__(self, cfg: GridConfig, theta0: float = 0.0):
        self.cfg = cfg
        self.theta = theta0            # unwrapped true angle at the end of the last block
        self.t = 0.0                   # time at the end of the last block

    # ------------------------------------------------------------------
    def frequency(self, t: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        f = F_NOMINAL + cfg.f_drift_amp_hz * np.sin(TWO_PI * t / cfg.f_drift_period_s + cfg.f_drift_phase)
        for ev in cfg.events:
            if ev.kind != "frequency_event":
                continue
            tau = np.clip(t - ev.t_start, 0.0, None)
            rocof, target = ev.params["rocof_hz_s"], ev.params["delta_f_hz"]
            ramp = np.minimum(np.abs(rocof) * tau, abs(target)) * math.copysign(1.0, target)
            f = f + ramp
        return f

    def _phase_jumps(self, t: np.ndarray) -> np.ndarray:
        off = np.zeros_like(t)
        for ev in self.cfg.events:
            if ev.kind == "phase_jump":
                off += np.where(t >= ev.t_start, math.radians(ev.params["angle_deg"]), 0.0)
        return off

    def _amplitudes(self, t: np.ndarray) -> np.ndarray:
        """Per-phase fundamental amplitude, shape (3, n)."""
        cfg = self.cfg
        amp = np.tile(np.asarray(cfg.phase_unbalance, dtype=float)[:, None] * cfg.v_nominal, (1, t.size))
        for ev in cfg.events:
            if ev.kind == "voltage_sag":
                active = (t >= ev.t_start) & (t < ev.t_start + ev.params["duration_s"])
                amp[:, active] *= ev.params["depth_pu"]
            elif ev.kind == "single_phase_sag":
                active = (t >= ev.t_start) & (t < ev.t_start + ev.params["duration_s"])
                amp[int(ev.params["phase"]), active] *= ev.params["depth_pu"]
        return amp

    def _harmonic_gain(self, t: np.ndarray) -> np.ndarray:
        g = np.ones_like(t)
        for ev in self.cfg.events:
            if ev.kind == "harmonic_step":
                g = np.where(t >= ev.t_start, g * ev.params["gain"], g)
        return g

    # ------------------------------------------------------------------
    def generate(self, n: int, dt: float, angle_offset: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Advance the grid by `n` samples.

        `angle_offset` (rad, shape (n,)) is added to the *reported* angle only —
        it is how a coordinated phase-spoofing FDI attack is synthesised. The
        returned `theta` is always the true, physical grid angle.
        """
        t = self.t + dt * np.arange(1, n + 1)
        f = self.frequency(t)
        theta_smooth = self.theta + TWO_PI * np.cumsum(f) * dt
        theta = theta_smooth + self._phase_jumps(t)
        theta_rep = theta if angle_offset is None else theta + angle_offset

        amp = self._amplitudes(t)
        hg = self._harmonic_gain(t)
        cfg = self.cfg
        ph = PHASE_SHIFTS[:, None]
        th = theta_rep[None, :]
        v = (
            amp * np.cos(th + ph)
            + (cfg.h5_pu * hg)[None, :] * np.cos(5.0 * (th + ph) + cfg.h5_phase)
            + (cfg.h7_pu * hg)[None, :] * np.cos(7.0 * (th + ph) + cfg.h7_phase)
        )

        self.theta = float(theta_smooth[-1])
        self.t = float(t[-1])
        # True positive-sequence magnitude (what a PMU would report)
        v_pos = amp.mean(axis=0)
        return {"t": t, "v": v, "theta": theta, "f": f, "v_pos": v_pos}
