"""
False Data Injection (FDI) attacks on the PLL's voltage-measurement channel.

The attacker sits between the voltage transducers / ADC and the controller
(compromised sensor firmware, a man-in-the-middle on the sampled-values bus,
or malware on the control board) and rewrites the samples the PLL sees.

    1  sinusoidal      — interharmonic injection on one phase: excites a
                         persistent oscillation in the PLL frequency estimate
    2  ramp            — slowly growing DC bias on one phase (sensor-offset
                         style): a 50 Hz ripple in vq that grows over time
    3  pulse           — step bias on one phase for a short burst
    4  phase_spoofing  — coordinated, balanced manipulation of all three
                         phases that rotates the measured voltage vector by a
                         slowly accelerating angle δ(t). The PLL tracks the
                         spoofed angle perfectly (vq stays ≈ 0) so classic
                         residual detectors are blind, while the inverter
                         drifts out of synchronism with the real grid. Its
                         only local footprint is a small frequency deviation
                         that is indistinguishable from a natural grid
                         frequency event — unless it is cross-checked against
                         a trusted reference (the PMU stream).

Types 1–3 are "loud but harmless-ish"; type 4 is quiet and destructive.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .pll import TWO_PI

ATTACK_TYPES: Dict[int, str] = {
    0: "none",
    1: "sinusoidal",
    2: "ramp",
    3: "pulse",
    4: "phase_spoofing",
}
ATTACK_IDS: Dict[str, int] = {v: k for k, v in ATTACK_TYPES.items()}


@dataclass
class AttackSpec:
    kind: str = "none"
    t_start: float = math.inf
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def type_id(self) -> int:
        return ATTACK_IDS[self.kind]

    @property
    def present(self) -> bool:
        return self.kind != "none"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type_id"] = self.type_id
        if not math.isfinite(d["t_start"]):
            d["t_start"] = None
        return d


class Attack:
    """Evaluates an AttackSpec on a block of sample times."""

    def __init__(self, spec: AttackSpec):
        self.spec = spec

    def angle_offset(self, t: np.ndarray) -> Optional[np.ndarray]:
        """Spoofed angle δ(t) in rad added to all three measured phases, or None."""
        s = self.spec
        if s.kind != "phase_spoofing":
            return None
        tau = np.clip(t - s.t_start, 0.0, None)
        r, df_max, sign = s.params["rocof_hz_s"], s.params["max_offset_hz"], s.params["direction"]
        t_sat = df_max / r
        # Frequency offset ramps at r until df_max, then holds; δ = 2π ∫ Δf dτ
        ramp_part = 0.5 * r * np.minimum(tau, t_sat) ** 2
        hold_part = df_max * np.clip(tau - t_sat, 0.0, None)
        return sign * TWO_PI * (ramp_part + hold_part)

    def additive(self, t: np.ndarray) -> Optional[np.ndarray]:
        """Additive injection, shape (3, n), or None."""
        s = self.spec
        if s.kind in ("none", "phase_spoofing"):
            return None
        tau = t - s.t_start
        active = tau >= 0.0
        if not active.any():
            return None
        sig = np.zeros_like(t)
        p = s.params
        if s.kind == "sinusoidal":
            sig = np.where(active, p["amplitude_pu"] * np.sin(TWO_PI * p["freq_hz"] * tau + p["phase"]), 0.0)
        elif s.kind == "ramp":
            sig = np.where(active, np.clip(p["rate_pu_s"] * tau, -p["cap_pu"], p["cap_pu"]), 0.0)
        elif s.kind == "pulse":
            sig = np.where(active & (tau < p["duration_s"]), p["magnitude_pu"], 0.0)
        out = np.zeros((3, t.size))
        out[int(p["phase_index"])] = sig
        return out
