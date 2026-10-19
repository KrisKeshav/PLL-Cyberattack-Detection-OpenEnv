"""
Typed OpenEnv models: Action, Observation, Reward breakdown and State.

All models subclass the openenv-core base types so they plug straight into
`create_app` (HTTP + WebSocket) and `EnvClient`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

InverterMode = Literal["normal", "holdover", "disconnected"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PLLAction(Action):
    """The cyber-guard's decision for the window it has just observed."""

    attack_detected: bool = Field(..., description="True if a cyber-attack is believed to be in progress.")
    attack_type: int = Field(
        0, ge=0, le=4,
        description="0=none, 1=sinusoidal injection, 2=ramp bias, 3=pulse bias, 4=phase spoofing.",
    )
    confidence: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Probability that the reported attack_detected value is correct (scored with a Brier rule).",
    )
    protective_action: int = Field(
        0, ge=0, le=3,
        description=(
            "0=none, 1=alert operator (no physical effect), "
            "2=PMU-aided holdover: open the PLL loop and free-run on the trusted PMU frequency, "
            "3=disconnect inverter (latched for the rest of the episode)."
        ),
    )


# ---------------------------------------------------------------------------
# Observation pieces
# ---------------------------------------------------------------------------

class Waveforms(_Strict):
    """Last decision window (50 ms) decimated to 2 ms resolution (25 points each)."""
    va: List[float]
    vb: List[float]
    vc: List[float]
    vd: List[float]
    vq: List[float]
    f_hz: List[float] = Field(description="PLL frequency estimate (low-pass filtered), Hz.")


class SignalFeatures(_Strict):
    """Standard signal-processing features a protection relay would compute on-board."""
    v_pos_seq_pu: float = Field(description="Mean d-axis voltage ≈ positive-sequence magnitude.")
    vq_mean_pu: float
    vq_rms_pu: float = Field(description="RMS of the q-axis (phase-detector) error.")
    f_local_hz: float = Field(description="Mean local PLL frequency over the window.")
    f_local_std_hz: float
    rocof_local_hz_s: float = Field(description="Rate of change of local frequency (from the last 4 windows).")
    v_zero_seq_rms_pu: float = Field(description="RMS of (va+vb+vc)/3; non-zero for unbalanced faults or single-phase injection.")
    phase_rms_pu: List[float] = Field(description="Per-phase RMS [a, b, c].")
    phase_dc_offset_pu: List[float] = Field(description="Per-phase mean over the last two fundamental cycles.")
    vq_osc_freq_hz: float = Field(description="Dominant oscillation frequency in vq over the last 200 ms.")
    vq_osc_amp_pu: float
    f_local_minus_pmu_hz: float = Field(description="Local frequency minus the latest PMU frequency (note PMU latency).")


class PMUReading(_Strict):
    """Trusted, GPS-synchronised phasor-measurement stream from the substation.

    It is out-of-band (cannot be spoofed by the attacker), but low-rate
    (10 reports/s), noisy and delayed — so it cannot be used naively.
    """
    f_hz: float
    v_pos_pu: float
    age_ms: float = Field(description="Age of the measurement interval end-time at the moment of observation.")


class Trend(_Strict):
    """History of the last 10 decision windows (oldest first)."""
    f_local_hz: List[float]
    pmu_f_hz: List[float]
    vq_rms_pu: List[float]
    v_zero_seq_rms_pu: List[float]
    v_pos_seq_pu: List[float]


class RewardBreakdown(_Strict):
    """Typed Reward model: every component of the dense per-step reward."""
    detection: float = 0.0
    early_detection_bonus: float = 0.0
    calibration: float = 0.0
    classification: float = 0.0
    mitigation_cost: float = 0.0
    harm_penalty: float = 0.0
    total: float = 0.0


class EpisodeResult(_Strict):
    """Revealed only once the episode is done."""
    grader_score: float
    score_breakdown: Dict[str, Any]
    total_reward: float
    scenario: Dict[str, Any]


class PLLObservation(Observation):
    task_id: int = Field(0, ge=0, le=2)
    task_name: str = ""
    task_brief: str = Field("", description="Task instructions (sent on reset only).")
    step: int = 0
    max_steps: int = 0
    time_s: float = Field(0.0, description="Time since the start of the monitored episode, s.")
    decision_interval_ms: float = 50.0
    inverter_mode: InverterMode = "normal"
    waveforms: Optional[Waveforms] = None
    features: Optional[SignalFeatures] = None
    pmu: Optional[PMUReading] = None
    trend: Optional[Trend] = None
    reward_breakdown: Optional[RewardBreakdown] = None
    episode_result: Optional[EpisodeResult] = None
    message: str = ""


class PLLState(State):
    """Server-side episode state. Ground truth stays hidden until `done`."""
    task_id: int = 0
    task_name: str = ""
    seed: Optional[int] = None
    done: bool = False
    time_s: float = 0.0
    inverter_mode: InverterMode = "normal"
    cumulative_reward: float = 0.0
    grader_score: Optional[float] = None
    scenario: Optional[Dict[str, Any]] = Field(
        None, description="Attack / disturbance ground truth — null while the episode is running."
    )
