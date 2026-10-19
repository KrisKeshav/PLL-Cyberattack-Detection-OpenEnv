"""
Dense per-step reward.

Action a_t is judged against the window t the agent actually observed
(label y_t = 1 once the attack has started — the system is then considered
compromised until the end of the episode). Physical consequences of the
chosen protective action (costs, desynchronisation harm) are charged on the
window simulated right after the action.

The costs are asymmetric on purpose: false alarms and false trips are what
make real intrusion-detection systems unusable, so "always alarm" and
"always trip" are strongly negative-return strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import RewardBreakdown

TRUE_POSITIVE = 0.10
TRUE_NEGATIVE = 0.02
MISSED_DETECTION = -0.10
FALSE_ALARM = -0.25

EARLY_BONUS_MAX = 0.50
EARLY_BONUS_HORIZON_MS = 500.0

CALIBRATION_SCALE = 0.04            # 0.04 * (1 - 4*Brier) ∈ [-0.12, +0.04]

CLASS_CORRECT = 0.05
CLASS_WRONG = -0.02

HOLDOVER_COST = -0.02               # per window spent in holdover (degraded mode)
DISCONNECTED_COST = -0.04           # per window of lost generation
TRIP_COST_JUSTIFIED = -0.30         # one-off, attack in progress
TRIP_COST_FALSE = -1.50             # one-off, healthy grid ("destructive action")

DESYNC_WINDOW_PENALTY = -0.30       # each window with sustained >10° error while connected
DESYNC_EVENT_PENALTY = -1.00        # one-off, first loss of synchronism

REWARD_MAX = TRUE_POSITIVE + EARLY_BONUS_MAX + CALIBRATION_SCALE + CLASS_CORRECT
REWARD_MIN = (FALSE_ALARM - 3 * CALIBRATION_SCALE + TRIP_COST_FALSE + DISCONNECTED_COST
              + DESYNC_WINDOW_PENALTY + DESYNC_EVENT_PENALTY)


@dataclass
class StepOutcome:
    """Everything the reward needs to know about one transition."""
    label_attack: bool
    true_type: int
    steps_since_onset: int              # valid when label_attack
    detected: bool
    reported_type: int
    confidence: float
    first_true_positive: bool
    classify: bool                      # task scores the attack type
    tripped_now: bool
    mode_next: str                      # inverter mode during the next window
    desync_in_next_window: bool
    first_desync: bool
    decision_interval_ms: float


def compute_reward(o: StepOutcome) -> RewardBreakdown:
    r = RewardBreakdown()

    if o.label_attack:
        r.detection = TRUE_POSITIVE if o.detected else MISSED_DETECTION
        if o.first_true_positive:
            delay_ms = o.steps_since_onset * o.decision_interval_ms
            r.early_detection_bonus = EARLY_BONUS_MAX * max(0.0, 1.0 - delay_ms / EARLY_BONUS_HORIZON_MS)
        if o.classify and o.detected:
            r.classification = CLASS_CORRECT if o.reported_type == o.true_type else CLASS_WRONG
    else:
        r.detection = FALSE_ALARM if o.detected else TRUE_NEGATIVE

    # Brier score on the implied probability of "attack in progress"
    p_attack = o.confidence if o.detected else 1.0 - o.confidence
    brier = (p_attack - (1.0 if o.label_attack else 0.0)) ** 2
    r.calibration = CALIBRATION_SCALE * (1.0 - 4.0 * brier)

    if o.tripped_now:
        r.mitigation_cost += TRIP_COST_JUSTIFIED if o.label_attack else TRIP_COST_FALSE
    if o.mode_next == "holdover":
        r.mitigation_cost += HOLDOVER_COST
    elif o.mode_next == "disconnected":
        r.mitigation_cost += DISCONNECTED_COST

    if o.desync_in_next_window:
        r.harm_penalty += DESYNC_WINDOW_PENALTY
    if o.first_desync:
        r.harm_penalty += DESYNC_EVENT_PENALTY

    r.total = (r.detection + r.early_detection_bonus + r.calibration + r.classification
               + r.mitigation_cost + r.harm_penalty)
    return r
