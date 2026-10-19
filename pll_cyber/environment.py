"""
PLL Cyber-Guard environment (OpenEnv `Environment` implementation).

One episode = 40 decisions × 50 ms of a grid-connected inverter. Each step:

    1. the agent's action a_t is scored against window t (what it observed)
    2. the protective action is applied (normal / holdover / disconnect)
    3. the next 50 ms window is simulated at 5 kHz:
         grid → FDI attack → sensors+ADC → SRF-PLL → features
       alongside a "shadow" PLL that never mitigates, which gives the
       counterfactual time at which the unprotected inverter would have
       lost synchronism with the real grid
    4. physical harm (sustained >10° desynchronisation) is charged
"""

from __future__ import annotations

import math
import secrets
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from . import __version__
from .attacks import Attack, ATTACK_TYPES
from .features import FeatureExtractor
from .graders import grade_episode
from .grid import GridModel
from .models import (EpisodeResult, PLLAction, PLLObservation, PLLState, PMUReading,
                     RewardBreakdown, SignalFeatures, Trend, Waveforms)
from .pll import F_NOMINAL, SRFPLL
from .rewards import StepOutcome, compute_reward
from .scenarios import (ADC_LSB_PU, DECISION_INTERVAL_S, DT, MAX_STEPS, PMU_F_NOISE_HZ,
                        PMU_REPORT_INTERVAL_S, PMU_V_NOISE_PU, SAMPLE_RATE_HZ,
                        SAMPLES_PER_DECISION, SENSOR_NOISE_PU, TASKS, WARMUP_S,
                        Scenario, sample_scenario)
from .sensors import DisciplinedHoldover, PMUStream, measure

DESYNC_THRESHOLD_DEG = 10.0
DESYNC_SUSTAIN_S = 0.030
_DESYNC_RAD = math.radians(DESYNC_THRESHOLD_DEG)
_DESYNC_SAMPLES = int(round(DESYNC_SUSTAIN_S * SAMPLE_RATE_HZ))
_INTERVAL_MS = DECISION_INTERVAL_S * 1000.0


class _DesyncTracker:
    """Detects sustained loss of synchronism: |θ̂ − θ| > 10° for ≥ 30 ms."""

    def __init__(self):
        self.run = 0
        self.first_step: Optional[int] = None
        self.max_err_deg = 0.0

    def update(self, theta_hat: List[float], theta_true: np.ndarray, step: int, active: bool = True) -> bool:
        err = np.abs(np.angle(np.exp(1j * (np.asarray(theta_hat) - theta_true))))
        if not active:
            self.run = 0
            return False
        self.max_err_deg = max(self.max_err_deg, float(np.degrees(err.max())))
        hit = False
        run = self.run
        for above in (err > _DESYNC_RAD):
            run = run + 1 if above else 0
            if run >= _DESYNC_SAMPLES:
                hit = True
        self.run = run
        if hit and self.first_step is None:
            self.first_step = step
        return hit


class PLLCyberEnvironment(Environment[PLLAction, PLLObservation, PLLState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._sc: Optional[Scenario] = None
        self._state = PLLState(episode_id=None, step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[int] = None, task: Optional[str] = None, **kwargs: Any) -> PLLObservation:
        tid = self._resolve_task(task_id, task)
        if seed is None:
            seed = secrets.randbelow(2 ** 31)
        seed = int(seed)

        sc = sample_scenario(tid, seed)
        self._sc = sc
        ss = np.random.SeedSequence([seed, 2000 + tid])
        rng_sensor, rng_pmu = (np.random.default_rng(s) for s in ss.spawn(2))
        self._rng_sensor = rng_sensor

        self._grid = GridModel(sc.grid, theta0=sc.theta0)
        self._attack = Attack(sc.attack)
        self._pll = SRFPLL(DT, theta0=sc.theta0)
        self._shadow = SRFPLL(DT, theta0=sc.theta0)
        self._pmu = PMUStream(PMU_REPORT_INTERVAL_S, sc.pmu_latency_s, PMU_F_NOISE_HZ,
                              PMU_V_NOISE_PU, rng_pmu)
        self._fx = FeatureExtractor(SAMPLE_RATE_HZ, SAMPLES_PER_DECISION)
        self._desync = _DesyncTracker()
        self._desync_shadow = _DesyncTracker()
        self._holdover = DisciplinedHoldover()
        self._theta_hist: deque = deque(maxlen=6)     # (t0, θ̂ samples) of recent windows

        self._mode = "normal"
        self._step = 0
        self._done = False
        self._had_tp = False
        self._log: List[Dict[str, Any]] = []
        self._cum_reward = 0.0
        self._result: Optional[EpisodeResult] = None

        # Warm-up: let the PLL settle; the last warm-up window is observation 0.
        n_warm = int(round(WARMUP_S / DECISION_INTERVAL_S))
        for k in range(n_warm):
            window = self._simulate_window(step=k - n_warm + 1, warmup=True)

        self._state = PLLState(
            episode_id=episode_id or str(uuid.uuid4()), step_count=0, task_id=tid,
            task_name=TASKS[tid].name, seed=seed, done=False, time_s=0.0,
            inverter_mode="normal", cumulative_reward=0.0,
        )
        obs = self._observation(window, reward=None, breakdown=None)
        obs.task_brief = TASKS[tid].brief
        obs.message = f"Episode started: task {tid} ({TASKS[tid].difficulty}), seed {seed}."
        return obs

    def step(self, action: PLLAction, timeout_s: Optional[float] = None, **kwargs: Any) -> PLLObservation:
        if self._sc is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            obs = self._observation(self._last_window, reward=0.0, breakdown=RewardBreakdown())
            obs.message = "Episode already finished — call reset() to start a new one."
            return obs

        sc = self._sc
        t = self._step
        onset = sc.onset_step
        label = onset is not None and t >= onset
        detected = bool(action.attack_detected)
        first_tp = label and detected and not self._had_tp
        self._had_tp = self._had_tp or (label and detected)

        tripped_now = action.protective_action == 3 and self._mode != "disconnected"
        if self._mode != "disconnected":
            self._mode = {3: "disconnected", 2: "holdover"}.get(action.protective_action, "normal")

        window = self._simulate_window(step=t + 1)

        outcome = StepOutcome(
            label_attack=label, true_type=sc.attack.type_id,
            steps_since_onset=(t - onset) if label else 0,
            detected=detected, reported_type=int(action.attack_type),
            confidence=float(action.confidence), first_true_positive=first_tp,
            classify=sc.task_id >= 1, tripped_now=tripped_now, mode_next=self._mode,
            desync_in_next_window=window["desync"], first_desync=window["first_desync"],
            decision_interval_ms=_INTERVAL_MS,
        )
        breakdown = compute_reward(outcome)
        self._cum_reward += breakdown.total

        self._log.append({
            "step": t, "label": label, "detected": detected,
            "attack_type": int(action.attack_type), "confidence": float(action.confidence),
            "protective_action": int(action.protective_action), "reward": breakdown.total,
        })
        self._step += 1
        self._done = self._step >= MAX_STEPS
        if self._done:
            self._finish()

        self._state.step_count = self._step
        self._state.done = self._done
        self._state.time_s = round(self._step * DECISION_INTERVAL_S, 3)
        self._state.inverter_mode = self._mode
        self._state.cumulative_reward = round(self._cum_reward, 6)

        obs = self._observation(window, reward=breakdown.total, breakdown=breakdown)
        if self._done:
            obs.message = f"Episode finished. grader_score={self._result.grader_score:.3f}"
        return obs

    @property
    def state(self) -> PLLState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="pll-cyberattack-detection",
            description=("Cyber-guard for a grid-connected inverter: detect, classify and mitigate false-data-"
                         "injection attacks on an SRF-PLL, without alarming on natural grid disturbances."),
            version=__version__,
            author="Kris Keshav",
            documentation_url="https://github.com/KrisKeshav/PLL-Cyberattack-Detection-OpenEnv",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_task(task_id: Optional[int], task: Optional[str]) -> int:
        if task is not None:
            for tid, spec in TASKS.items():
                if task in (spec.name, spec.difficulty, str(tid)):
                    return tid
            raise ValueError(f"Unknown task '{task}'. Options: {[s.name for s in TASKS.values()]}")
        tid = 0 if task_id is None else int(task_id)
        if tid not in TASKS:
            raise ValueError(f"task_id must be one of {sorted(TASKS)}")
        return tid

    def _simulate_window(self, step: int, warmup: bool = False) -> Dict[str, Any]:
        n = SAMPLES_PER_DECISION
        grid, attack = self._grid, self._attack
        t_next = grid.t + DT * np.arange(1, n + 1)
        out = grid.generate(n, DT, attack.angle_offset(t_next))
        v = out["v"]
        add = attack.additive(out["t"])
        if add is not None:
            v = v + add
        vm = measure(v, self._rng_sensor, SENSOR_NOISE_PU, ADC_LSB_PU)

        self._pmu.ingest(out["t"], out["theta"], out["v_pos"])
        now = float(out["t"][-1])
        rep = self._pmu.latest(now)

        holdover = self._mode == "holdover"
        if holdover:
            # θ̂ held by the PLL is the angle for the next sample instant, out["t"][0]
            self._update_holdover(t_next=float(out["t"][0]))
        elif self._holdover.active:
            self._holdover.release()
        r = self._pll.run(vm[0], vm[1], vm[2], holdover=holdover,
                          holdover_freq_hz=self._holdover.f_hold,
                          holdover_rocof_hz_s=self._holdover.rocof)
        if holdover:      # carry the extrapolated frequency ramp into the next window
            self._holdover.f_hold += self._holdover.rocof * n * DT
        rs = self._shadow.run(vm[0], vm[1], vm[2])
        self._theta_hist.append((float(out["t"][0]), r.theta_hat))

        connected = self._mode != "disconnected"
        had_desync = self._desync.first_step is not None
        desync = False
        if not warmup:
            desync = self._desync.update(r.theta_hat, out["theta"], step, active=connected)
            self._desync_shadow.update(rs.theta_hat, out["theta"], step)

        pmu_f = rep.f_hz if rep is not None else F_NOMINAL
        fx = self._fx.update(vm, r.vd, r.vq, r.f_hat, pmu_f)
        window = {
            **fx,
            "pmu": {
                "f_hz": round(pmu_f, 4),
                "v_pos_pu": round(rep.v_pos_pu if rep is not None else 1.0, 4),
                "age_ms": round((now - rep.t_end) * 1000.0, 1) if rep is not None else 999.0,
            },
            "desync": desync,
            "first_desync": desync and not had_desync,
        }
        self._last_window = window
        return window

    def _theta_hat_at(self, t: float) -> Optional[float]:
        for t0, th in reversed(self._theta_hist):
            i = int(round((t - t0) / DT))
            if 0 <= i < len(th):
                return th[i]
        return None

    def _update_holdover(self, t_next: float) -> None:
        """Engage / steer the PMU-disciplined oscillator before simulating a window."""
        if not self._holdover.active:
            self._holdover.engage(t_next)
        self._holdover.update(self._pll, self._pmu, t_next)

    def _finish(self) -> None:
        sc = self._sc
        graded = grade_episode(
            sc.task_id, self._log, sc.onset_step, _INTERVAL_MS,
            true_type=sc.attack.type_id,
            unmitigated_desync_step=self._desync_shadow.first_step,
            actual_desync_step=self._desync.first_step,
        )
        scenario = sc.to_dict()
        scenario["unmitigated_desync_step"] = self._desync_shadow.first_step
        scenario["actual_desync_step"] = self._desync.first_step
        scenario["max_sync_error_deg"] = round(self._desync.max_err_deg, 3)
        scenario["unmitigated_max_sync_error_deg"] = round(self._desync_shadow.max_err_deg, 3)
        scenario["attack_type_name"] = ATTACK_TYPES[sc.attack.type_id]
        self._result = EpisodeResult(
            grader_score=graded["score"], score_breakdown=graded,
            total_reward=round(self._cum_reward, 6), scenario=scenario,
        )
        self._state.grader_score = graded["score"]
        self._state.scenario = scenario

    def _observation(self, window: Dict[str, Any], reward: Optional[float],
                     breakdown: Optional[RewardBreakdown]) -> PLLObservation:
        sc = self._sc
        return PLLObservation(
            done=self._done,
            reward=None if reward is None else round(reward, 6),
            task_id=sc.task_id,
            task_name=TASKS[sc.task_id].name,
            step=self._step,
            max_steps=MAX_STEPS,
            time_s=round(self._step * DECISION_INTERVAL_S, 3),
            decision_interval_ms=_INTERVAL_MS,
            inverter_mode=self._mode,
            waveforms=Waveforms(**window["waveforms"]),
            features=SignalFeatures(**window["features"]),
            pmu=PMUReading(**window["pmu"]),
            trend=Trend(**window["trend"]),
            reward_breakdown=breakdown,
            episode_result=self._result if self._done else None,
        )
