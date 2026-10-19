"""
On-board signal processing: turns raw per-sample signals of a decision window
into the compact, physically meaningful features exposed to the agent.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np

from .pll import F_NOMINAL

TREND_LEN = 10
FFT_BUFFER_S = 0.200
DECIMATE = 10          # 5 kHz → 500 Hz for the exported waveforms (25 points / window)


def _r(x: float, nd: int = 5) -> float:
    return float(round(float(x), nd))


def _rl(xs, nd: int = 4) -> List[float]:
    return [float(round(float(x), nd)) for x in xs]


class FeatureExtractor:
    def __init__(self, sample_rate_hz: float, samples_per_window: int):
        self.fs = sample_rate_hz
        self.n = samples_per_window
        fft_len = int(FFT_BUFFER_S * sample_rate_hz) // 5          # vq decimated ×5 → 1 kHz
        self.vq_buf: Deque[float] = deque(maxlen=fft_len)
        self.f_means: Deque[float] = deque(maxlen=TREND_LEN)
        self.pmu_f: Deque[float] = deque(maxlen=TREND_LEN)
        self.vq_rms: Deque[float] = deque(maxlen=TREND_LEN)
        self.v0_rms: Deque[float] = deque(maxlen=TREND_LEN)
        self.vpos: Deque[float] = deque(maxlen=TREND_LEN)

    def update(self, v_meas: np.ndarray, vd: List[float], vq: List[float], f_hat: List[float],
               pmu_f: float) -> Dict[str, object]:
        vd_a, vq_a, f_a = np.asarray(vd), np.asarray(vq), np.asarray(f_hat)

        self.vq_buf.extend(vq_a[::5].tolist())
        vq_rms = float(np.sqrt(np.mean(vq_a ** 2)))
        v0 = v_meas.sum(axis=0) / 3.0
        v0_rms = float(np.sqrt(np.mean(v0 ** 2)))
        f_mean = float(f_a.mean())
        vpos = float(vd_a.mean())

        self.f_means.append(f_mean)
        self.pmu_f.append(pmu_f)
        self.vq_rms.append(vq_rms)
        self.v0_rms.append(v0_rms)
        self.vpos.append(vpos)

        # RoCoF from the slope of window-mean frequency over the last 4 windows
        fm = np.asarray(self.f_means)[-4:]
        if fm.size >= 2:
            rocof = float(np.polyfit(np.arange(fm.size) * (self.n / self.fs), fm, 1)[0])
        else:
            rocof = 0.0

        # Dominant vq oscillation (Hann-windowed FFT over the last 200 ms at 1 kHz)
        buf = np.asarray(self.vq_buf)
        osc_f, osc_a = 0.0, 0.0
        if buf.size >= 32:
            x = (buf - buf.mean()) * np.hanning(buf.size)
            spec = np.abs(np.fft.rfft(x)) * 2.0 / (np.hanning(buf.size).sum())
            freqs = np.fft.rfftfreq(buf.size, d=5.0 / self.fs)
            band = (freqs >= 3.0) & (freqs <= 400.0)
            if band.any():
                i = int(np.argmax(np.where(band, spec, 0.0)))
                osc_f, osc_a = float(freqs[i]), float(spec[i])

        two_cycles = int(round(2.0 * self.fs / F_NOMINAL))
        dc = v_meas[:, -two_cycles:].mean(axis=1)

        features = {
            "v_pos_seq_pu": _r(vpos),
            "vq_mean_pu": _r(vq_a.mean()),
            "vq_rms_pu": _r(vq_rms),
            "f_local_hz": _r(f_mean),
            "f_local_std_hz": _r(f_a.std()),
            "rocof_local_hz_s": _r(rocof, 4),
            "v_zero_seq_rms_pu": _r(v0_rms),
            "phase_rms_pu": _rl(np.sqrt(np.mean(v_meas ** 2, axis=1)), 5),
            "phase_dc_offset_pu": _rl(dc, 5),
            "vq_osc_freq_hz": _r(osc_f, 1),
            "vq_osc_amp_pu": _r(osc_a),
            "f_local_minus_pmu_hz": _r(f_mean - pmu_f),
        }
        waveforms = {
            "va": _rl(v_meas[0, DECIMATE - 1::DECIMATE]),
            "vb": _rl(v_meas[1, DECIMATE - 1::DECIMATE]),
            "vc": _rl(v_meas[2, DECIMATE - 1::DECIMATE]),
            "vd": _rl(vd_a[DECIMATE - 1::DECIMATE]),
            "vq": _rl(vq_a[DECIMATE - 1::DECIMATE]),
            "f_hz": _rl(f_a[DECIMATE - 1::DECIMATE], 4),
        }
        trend = {
            "f_local_hz": _rl(self.f_means, 4),
            "pmu_f_hz": _rl(self.pmu_f, 4),
            "vq_rms_pu": _rl(self.vq_rms, 5),
            "v_zero_seq_rms_pu": _rl(self.v0_rms, 5),
            "v_pos_seq_pu": _rl(self.vpos, 4),
        }
        return {"features": features, "waveforms": waveforms, "trend": trend}
