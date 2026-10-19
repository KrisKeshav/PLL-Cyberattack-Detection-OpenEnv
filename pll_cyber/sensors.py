"""
Measurement chain: local voltage sensors (noise + ADC quantisation) and the
trusted, out-of-band PMU reference stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from .pll import F_NOMINAL, TWO_PI


def measure(v: np.ndarray, rng: np.random.Generator, noise_pu: float, lsb_pu: float) -> np.ndarray:
    """Additive Gaussian sensor noise followed by uniform ADC quantisation."""
    noisy = v + rng.normal(0.0, noise_pu, v.shape)
    return np.round(noisy / lsb_pu) * lsb_pu


@dataclass
class PMUReport:
    t_end: float          # end of the measurement interval (s)
    f_hz: float
    v_pos_pu: float
    theta: float          # synchrophasor angle at t_end (rad, GPS-referenced)


class PMUStream:
    """IEEE C37.118-style µPMU at the substation (GPS time-stamped, not spoofable).

    Each report gives the average frequency over the last reporting interval,
    computed from the true phase-angle advance — so it also shows the
    frequency spike that accompanies a real phase jump — plus Gaussian noise.
    Reports reach the inverter after a fixed communication latency.

    The synchrophasor angle is only consumed by the inverter's holdover
    controller; the monitoring agent subscribes to the frequency and magnitude
    channels (a low-bandwidth SCADA status feed).
    """

    def __init__(self, interval_s: float, latency_s: float, f_noise_hz: float,
                 v_noise_pu: float, rng: np.random.Generator, angle_noise_rad: float = 0.0035):
        self.interval = interval_s
        self.latency = latency_s
        self.f_noise = f_noise_hz
        self.v_noise = v_noise_pu
        self.angle_noise = angle_noise_rad
        self.rng = rng
        self.reports: Deque[PMUReport] = deque(maxlen=64)
        self._last_theta: Optional[Tuple[float, float]] = None   # (t, θ) at last report boundary
        self._v_acc = 0.0
        self._v_n = 0
        self._next_t = None

    def ingest(self, t: np.ndarray, theta: np.ndarray, v_pos: np.ndarray) -> None:
        """Feed a block of true grid samples; emits reports at interval boundaries."""
        if self._next_t is None:
            self._next_t = (np.floor(t[0] / self.interval) + 1) * self.interval
            self._last_theta = (float(t[0]), float(theta[0]))
        start = 0
        while True:
            idx = int(np.searchsorted(t, self._next_t - 1e-9))
            if idx >= t.size:
                break
            seg = v_pos[start:idx + 1]
            self._v_acc += float(seg.sum())
            self._v_n += seg.size
            t0, th0 = self._last_theta
            t1, th1 = float(t[idx]), float(theta[idx])
            f = (th1 - th0) / (TWO_PI * (t1 - t0)) if t1 > t0 else F_NOMINAL
            v = self._v_acc / max(self._v_n, 1)
            self.reports.append(PMUReport(
                t_end=t1,
                f_hz=float(f + self.rng.normal(0.0, self.f_noise)),
                v_pos_pu=float(v + self.rng.normal(0.0, self.v_noise)),
                theta=float(th1 + self.rng.normal(0.0, self.angle_noise)),
            ))
            self._last_theta = (t1, th1)
            self._v_acc, self._v_n = 0.0, 0
            self._next_t += self.interval
            start = idx + 1
        seg = v_pos[start:]
        self._v_acc += float(seg.sum())
        self._v_n += seg.size

    def arrived(self, since: float, now: float) -> List[PMUReport]:
        """Reports that arrived in (since, now]."""
        return [r for r in self.reports if since < r.t_end + self.latency <= now + 1e-9]

    def latest(self, now: float) -> Optional[PMUReport]:
        """Most recent report that has arrived by time `now`."""
        for rep in reversed(self.reports):
            if rep.t_end + self.latency <= now + 1e-9:
                return rep
        return None


class DisciplinedHoldover:
    """PMU/GPS-disciplined oscillator used while the PLL loop is open.

    On every PMU report that has arrived, it re-anchors θ̂ towards the angle
    *predicted for the present instant* from the synchrophasor angle, the
    reported frequency and a RoCoF estimated by a linear fit over the last
    few reports (latency compensation). Between reports θ̂ free-runs with the
    extrapolated frequency ramp.
    """

    PHASE_GAIN = 0.7
    FIT_REPORTS = 4
    MAX_ROCOF = 1.0          # Hz/s, beyond natural grid behaviour

    def __init__(self):
        self.active = False
        self.f_hold = 50.0
        self.rocof = 0.0
        self.last_arrival_check = -1.0

    def engage(self, now: float) -> None:
        self.active = True
        self.last_arrival_check = -1.0

    def release(self) -> None:
        self.active = False

    def update(self, pll, pmu: "PMUStream", now: float) -> None:
        """Steer the oscillator using reports that arrived since the last call."""
        arrived = [r for r in pmu.reports if r.t_end + pmu.latency <= now + 1e-9]
        if not arrived:
            return
        newest = arrived[-1]
        if newest.t_end + pmu.latency <= self.last_arrival_check:
            return                                    # nothing new
        T = pmu.interval
        fit = arrived[-self.FIT_REPORTS:]
        tm = np.array([r.t_end - T / 2 for r in fit])      # report freq = interval average
        fv = np.array([r.f_hz for r in fit])
        if len(fit) >= 3:
            coef = np.polyfit(tm, fv, 1)
            self.rocof = float(np.clip(coef[0], -self.MAX_ROCOF, self.MAX_ROCOF))
            f_at_t = float(np.polyval(coef, newest.t_end))
        else:
            self.rocof = 0.0
            f_at_t = float(np.median(fv))
        dt = now - newest.t_end
        theta_pred = newest.theta + TWO_PI * (f_at_t * dt + 0.5 * self.rocof * dt * dt)
        err = (theta_pred - pll.theta_hat + np.pi) % TWO_PI - np.pi
        pll.theta_hat = float((pll.theta_hat + self.PHASE_GAIN * err + np.pi) % TWO_PI - np.pi)
        self.f_hold = f_at_t + self.rocof * dt
        self.last_arrival_check = now
