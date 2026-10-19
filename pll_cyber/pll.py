"""
Synchronous Reference Frame Phase-Locked Loop (SRF-PLL).

Conventions (cosine-referenced, amplitude-invariant):

    va = V cos(θ),  vb = V cos(θ − 2π/3),  vc = V cos(θ + 2π/3)

    Clarke:  vα = (2va − vb − vc) / 3        vβ = (vb − vc) / √3
             → vα = V cos θ,  vβ = V sin θ

    Park:    vd =  vα cos θ̂ + vβ sin θ̂  = V cos(θ − θ̂)
             vq = −vα sin θ̂ + vβ cos θ̂  = V sin(θ − θ̂)

So when locked (θ̂ ≈ θ): vd → V and vq → 0, and for small errors vq ≈ V·(θ − θ̂),
which is the phase-detector characteristic the PI loop filter drives to zero.

Loop design (amplitude-normalised, so dynamics do not change during voltage sags):

    θ̂/θ = (Kp s + Ki) / (s² + Kp s + Ki),   ωn = √Ki,   ζ = Kp / (2√Ki)

Defaults give ωn = 100 rad/s, ζ = 0.707 (≈ 25 Hz closed-loop bandwidth), a
typical industrial tuning trade-off between tracking speed and harmonic rejection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

TWO_PI = 2.0 * math.pi
SQRT3 = math.sqrt(3.0)

F_NOMINAL = 50.0
OMEGA_NOMINAL = TWO_PI * F_NOMINAL


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [−π, π)."""
    return (angle + math.pi) % TWO_PI - math.pi


def clarke(va: float, vb: float, vc: float) -> tuple[float, float]:
    """Amplitude-invariant Clarke transform (abc → αβ), using all three phases."""
    return (2.0 * va - vb - vc) / 3.0, (vb - vc) / SQRT3


def park(v_alpha: float, v_beta: float, theta: float) -> tuple[float, float]:
    """Park transform (αβ → dq) on the estimated angle θ̂."""
    c, s = math.cos(theta), math.sin(theta)
    return v_alpha * c + v_beta * s, -v_alpha * s + v_beta * c


@dataclass
class PLLGains:
    omega_n: float = 100.0      # natural frequency (rad/s)
    zeta: float = 0.7071        # damping ratio
    freq_filter_hz: float = 10.0  # LPF on the reported frequency
    max_dev_hz: float = 5.0      # anti-windup / frequency clamp

    @property
    def kp(self) -> float:
        return 2.0 * self.zeta * self.omega_n

    @property
    def ki(self) -> float:
        return self.omega_n ** 2


@dataclass
class PLLWindowOutput:
    """Per-sample PLL signals for one simulation window."""
    vd: List[float] = field(default_factory=list)
    vq: List[float] = field(default_factory=list)
    f_hat: List[float] = field(default_factory=list)       # filtered frequency estimate (Hz)
    theta_hat: List[float] = field(default_factory=list)


class SRFPLL:
    """Discrete-time SRF-PLL with an optional PMU-aided holdover mode.

    Modes:
        normal   — closed-loop tracking of the measured voltages
        holdover — loop opened; θ̂ free-runs at an externally supplied trusted
                   frequency (e.g. a PMU stream). Measurements are still
                   transformed so vd/vq remain observable.
    """

    def __init__(self, dt: float, gains: PLLGains | None = None, theta0: float = 0.0):
        self.dt = dt
        self.gains = gains or PLLGains()
        self.reset(theta0)

    def reset(self, theta0: float = 0.0) -> None:
        self.theta_hat = wrap_angle(theta0)
        self.integrator = 0.0              # ∫ normalised vq dt (rad/s units after Ki)
        self.omega_hat = OMEGA_NOMINAL
        self.f_filtered = F_NOMINAL
        self.vd_filtered = 1.0
        self.vd = 1.0
        self.vq = 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "theta_hat": self.theta_hat,
            "integrator": self.integrator,
            "omega_hat": self.omega_hat,
            "f_filtered": self.f_filtered,
            "vd_filtered": self.vd_filtered,
        }

    def run(
        self,
        va: Sequence[float],
        vb: Sequence[float],
        vc: Sequence[float],
        holdover: bool = False,
        holdover_freq_hz: float = F_NOMINAL,
        holdover_rocof_hz_s: float = 0.0,
        record: bool = True,
    ) -> PLLWindowOutput:
        """Process a block of samples. Returns per-sample signals if `record`."""
        g = self.gains
        dt = self.dt
        kp, ki = g.kp, g.ki
        max_dev = TWO_PI * g.max_dev_hz
        a_f = dt / (dt + 1.0 / (TWO_PI * g.freq_filter_hz))      # 1st-order LPF coefficient
        a_v = dt / (dt + 1.0 / (TWO_PI * 20.0))                   # amplitude estimator LPF
        omega_hold = TWO_PI * holdover_freq_hz
        d_omega_hold = TWO_PI * holdover_rocof_hz_s * dt

        th = self.theta_hat
        integ = self.integrator
        f_filt = self.f_filtered
        vd_filt = self.vd_filtered
        omega = self.omega_hat
        vd = self.vd
        vq = self.vq

        out = PLLWindowOutput()
        cos, sin = math.cos, math.sin
        two_pi, pi = TWO_PI, math.pi

        for a, b, c in zip(va, vb, vc):
            v_alpha = (2.0 * a - b - c) / 3.0
            v_beta = (b - c) / SQRT3
            ct, st = cos(th), sin(th)
            vd = v_alpha * ct + v_beta * st
            vq = -v_alpha * st + v_beta * ct

            vd_filt += a_v * (vd - vd_filt)
            if holdover:
                omega = omega_hold
                omega_hold += d_omega_hold
            else:
                vq_n = vq / max(vd_filt, 0.2)          # amplitude normalisation
                integ += ki * vq_n * dt
                if integ > max_dev:
                    integ = max_dev
                elif integ < -max_dev:
                    integ = -max_dev
                dev = kp * vq_n + integ
                if dev > max_dev:
                    dev = max_dev
                elif dev < -max_dev:
                    dev = -max_dev
                omega = OMEGA_NOMINAL + dev

            f_filt += a_f * (omega / two_pi - f_filt)
            if record:
                # θ̂ recorded is the angle used for this sample's Park transform,
                # i.e. the controller's estimate of θ at this sampling instant.
                out.vd.append(vd)
                out.vq.append(vq)
                out.f_hat.append(f_filt)
                out.theta_hat.append(th)
            th = (th + omega * dt + pi) % two_pi - pi

        self.theta_hat = th
        self.integrator = integ
        self.f_filtered = f_filt
        self.vd_filtered = vd_filt
        self.omega_hat = omega
        self.vd = vd
        self.vq = vq
        return out
