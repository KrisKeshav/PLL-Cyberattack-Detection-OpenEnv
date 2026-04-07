"""
Main environment class for the PLL Cyberattack Detection OpenEnv.

Implements step(), reset(), get_state(), and compute_reward().
Manages the PLL simulation, attack injection, observation windowing,
episode history, and grading.
"""

import uuid
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from collections import deque

from src.models import Observation, Action, Reward, State
from src.pll_sim import SRFPLLSimulator, OMEGA0
from src.attacks import (
    AttackGenerator,
    sample_sinusoidal_params,
    sample_ramp_params,
    sample_pulse_params,
    sample_stealthy_params,
    sample_attack_start,
    get_attack_type_id,
)
from src.graders import grade_task_easy, grade_task_medium, grade_task_hard
from src.detector import AdaptiveDetector


WINDOW_SIZE = 20
MAX_STEPS = 500
LOCK_LOSS_THRESHOLD = 0.0873  # 5 degrees in radians

DETECTION_THRESHOLD = 2.0
EARLY_DETECTION_WINDOW = 100
FALSE_ALARM_PENALTY = -0.2
TRUE_POSITIVE_REWARD = 0.1
TRUE_NEGATIVE_REWARD = 0.05
MISSED_DETECTION_PENALTY = -0.05
CLASSIFICATION_BONUS = 0.05
LOCK_LOSS_PENALTY = -2.0


class PLLAttackEnv:
    """OpenEnv-compliant PLL cyberattack detection environment."""

    def __init__(self):
        self.pll = SRFPLLSimulator()
        self.rng: Optional[np.random.Generator] = None
        self.task_id = 0
        self.step_count = 0
        self.episode_id = ""
        self.done = False

        # Attack state
        self.attack_generator: Optional[AttackGenerator] = None
        self.attack_active = False
        self.attack_type = 0
        self.attack_params: Dict[str, Any] = {}
        self.attack_start_step = 0
        self.true_attack_type = 0

        # Detection tracking
        self.first_detection_recorded = False
        self.first_detection_step = 0

        # Lock loss tracking (Task 2 / hard)
        self.lock_lost = False
        self.lock_loss_step: Optional[int] = None
        self.lock_loss_penalized = False

        # Observation windows
        self.vq_window: deque = deque(maxlen=WINDOW_SIZE)
        self.vd_window: deque = deque(maxlen=WINDOW_SIZE)
        self.omega_window: deque = deque(maxlen=WINDOW_SIZE)
        self.omega_deviation_window: deque = deque(maxlen=WINDOW_SIZE)  # Fix 8

        # Detector
        self.detector = AdaptiveDetector()

        # Episode history for grading
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: int = 0, seed: Optional[int] = None) -> Observation:
        """
        Reset the environment for a new episode.
        Args:
            task_id: 0=easy (sinusoidal), 1=medium (multi-type),
                     2=hard (stealthy).
            seed:    Optional RNG seed for reproducibility.
        Returns:
            Initial Observation with non-zero raw_voltages.
        """
        self.rng = np.random.default_rng(seed)  # seed=None → random

        self.task_id = task_id
        self.step_count = 0
        self.episode_id = str(uuid.uuid4())
        self.done = False

        # Reset PLL simulator
        self.pll.reset()

        # Reset detection tracking
        self.first_detection_recorded = False
        self.first_detection_step = 0

        # Reset lock-loss tracking
        self.lock_lost = False
        self.lock_loss_step = None
        self.lock_loss_penalized = False

        # Reset history
        self.history = []

        # Reset observation windows (Fix 6: no theta_err_window)
        self.vq_window = deque(maxlen=WINDOW_SIZE)
        self.vd_window = deque(maxlen=WINDOW_SIZE)
        self.omega_window = deque(maxlen=WINDOW_SIZE)
        self.omega_deviation_window = deque(maxlen=WINDOW_SIZE)

        # Reset detector
        self.detector = AdaptiveDetector()

        # Sample attack for this episode
        self._setup_attack()
      
        for _ in range(WINDOW_SIZE):
            pll_out = self.pll.step(0.0)  # no attack during warm-up
            omega_norm = (pll_out["omega_hat"] - OMEGA0) / OMEGA0
            omega_dev  = pll_out["omega_hat"] - OMEGA0
            self.vq_window.append(pll_out["vq"])
            self.vd_window.append(pll_out["vd"])
            self.omega_window.append(omega_norm)
            self.omega_deviation_window.append(omega_dev)
        # step_count stays at 0 — warm-up steps are invisible to the agent

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.
        Args:
            action: Agent's Action for this step.
        Returns:
            (observation, reward, done, info)
        """
        if self.done:
            return (
                self._get_observation(),
                Reward(
                    total=0.0, detection_reward=0.0, classification_bonus=0.0,
                    early_detection_bonus=0.0, false_alarm_penalty=0.0,
                    lock_loss_penalty=0.0,
                ),
                True,
                {"message": "Episode already done. Call /reset to start a new episode."},
            )

        # --- Attack signal ------------------------------------------------
        # attack_active uses is_active() (step-based). It does NOT depend on the instantaneous
        # signal value, because the attack signal can cross zero even while the attack is active.
        attack_signal = self.attack_generator.get_signal(self.step_count, self.pll.t)
        self.attack_active = self.attack_generator.is_active(self.step_count)

        # --- Advance PLL --------------------------------------------------
        pll_out = self.pll.step(attack_signal)

        # --- Updating observation windows -----------------------------------
        omega_norm = (pll_out["omega_hat"] - OMEGA0) / OMEGA0
        omega_dev  = pll_out["omega_hat"] - OMEGA0  # raw deviation (rad/s)
        self.vq_window.append(pll_out["vq"])
        self.vd_window.append(pll_out["vd"])
        self.omega_window.append(omega_norm)
        self.omega_deviation_window.append(omega_dev)

        # --- Lock-loss check (Task 2) -------------------------
        PLL_CONVERGENCE_STEPS = 60  # PLL transient settles by ~step 50, using 60 for margin
        if (
            self.task_id == 2
            and not self.lock_lost
            and self.step_count > self.attack_start_step
            and self.step_count > PLL_CONVERGENCE_STEPS   # guard against startup transient
        ):
            if abs(pll_out["theta_err"]) > LOCK_LOSS_THRESHOLD:
                self.lock_lost = True
                self.lock_loss_step = self.step_count

        # --- Reward -------------------------------------------------------
        reward = self.compute_reward(action)

        # --- Record history entry for graders ----------------------------
        self.history.append({
            "step":              self.step_count,
            "attack_active":     self.attack_active,
            "attack_detected":   action.attack_detected,
            "true_attack_type":  self.true_attack_type,
            "agent_attack_type": action.attack_type,
            "theta_err":         pll_out["theta_err"],
        })

        # --- Advance step counter ----------------------------------------
        self.step_count += 1

        # --- Episode termination -----------------------------------------
        # Fix 4: Task 2 terminates early on lock-loss, not just at MAX_STEPS
        if self.step_count >= MAX_STEPS:
            self.done = True
        elif self.task_id == 2 and self.lock_lost:
            self.done = True  # early termination — no point continuing

        # --- Physics-informed detector (evaluation/debug only) ------------
        detector_output = self.detector.detect(self._get_observation())

        # --- Build info --------------------------------------------------
        info: Dict[str, Any] = {
            "detector": detector_output,
            "detector_features": {"step": self.step_count, "raw_score": detector_output.get("score")}
        }
        if self.done:
            info["grader_score"] = self._compute_grader_score()
            info["episode_id"]   = self.episode_id
            info["total_steps"]  = self.step_count
            info["lock_lost"]    = self.lock_lost

        return self._get_observation(), reward, self.done, info

    def compute_reward(self, action: Action) -> Reward:
        """
        Computes the dense reward signal for the current step.
        Reward components:
          detection_reward:     +0.10 true positive (per step)
                                +0.05 true negative (per step)
                                -0.05 missed detection (per step)
          false_alarm_penalty:  -0.20 per false-positive step
          classification_bonus: +0.05 per step correct type (task 1 only)
          early_detection_bonus: one-time sparse, scaled by detection speed
          lock_loss_penalty:    -2.00 one-time on lock loss (task 2 only)
        """
        detection_reward      = 0.0
        false_alarm_penalty   = 0.0
        classification_bonus  = 0.0
        early_detection_bonus = 0.0
        lock_loss_penalty     = 0.0

        if self.attack_active:
            if action.attack_detected:
                detection_reward = TRUE_POSITIVE_REWARD
                # One-time early detection bonus on first correct detection
                if not self.first_detection_recorded:
                    self.first_detection_step    = self.step_count
                    self.first_detection_recorded = True
                    # Relative steps since attack started
                    t = self.first_detection_step - self.attack_start_step
                    early_detection_bonus = max(0.0, 1.0 - t / EARLY_DETECTION_WINDOW)
            else:
                detection_reward = MISSED_DETECTION_PENALTY
        else:
            if action.attack_detected:
                false_alarm_penalty = FALSE_ALARM_PENALTY
            else:
                detection_reward = TRUE_NEGATIVE_REWARD

        # Task 1 (medium): per-step classification bonus
        if self.task_id == 1 and self.attack_active:
            if action.attack_type == self.true_attack_type:
                classification_bonus = CLASSIFICATION_BONUS

        # Task 2 (hard): one-time lock-loss penalty
        if self.task_id == 2 and self.lock_lost and not self.lock_loss_penalized:
            lock_loss_penalty        = LOCK_LOSS_PENALTY
            self.lock_loss_penalized = True

        total = (
            detection_reward
            + false_alarm_penalty
            + classification_bonus
            + early_detection_bonus
            + lock_loss_penalty
        )

        return Reward(
            total=total,
            detection_reward=detection_reward,
            classification_bonus=classification_bonus,
            early_detection_bonus=early_detection_bonus,
            false_alarm_penalty=false_alarm_penalty,
            lock_loss_penalty=lock_loss_penalty,
        )

    def get_state(self) -> State:
        """Returning full internal state for debugging / GET /state endpoint."""
        return State(
            theta_true=self.pll.theta_true,
            theta_hat=self.pll.theta_hat,
            omega_hat=self.pll.omega_hat,
            vq_integral=self.pll.vq_integral,
            attack_active=self.attack_active,
            attack_type=self.attack_type,
            attack_params=self.attack_params,
            attack_start_step=self.attack_start_step,
            lock_lost=self.lock_lost,
            step=self.step_count,
            episode_id=self.episode_id,
            task_id=self.task_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_attack(self) -> None:
        """Sample attack type and parameters based on current task_id."""
        self.attack_start_step = sample_attack_start(self.rng)

        if self.task_id == 0:
            # Easy: sinusoidal FDI only
            self.attack_params    = sample_sinusoidal_params(self.rng)
            self.true_attack_type = 1

        elif self.task_id == 1:
            # Medium: random choice of sinusoidal / ramp / pulse
            choice = int(self.rng.integers(0, 3))
            if choice == 0:
                self.attack_params    = sample_sinusoidal_params(self.rng)
                self.true_attack_type = 1
            elif choice == 1:
                self.attack_params    = sample_ramp_params(self.rng)
                self.true_attack_type = 2
            else:
                self.attack_params    = sample_pulse_params(self.rng)
                self.true_attack_type = 3

        elif self.task_id == 2:
            # Hard: stealthy low-and-slow
            self.attack_params    = sample_stealthy_params(self.rng)
            self.true_attack_type = 4

        self.attack_type      = get_attack_type_id(self.attack_params.get("type", "none"))
        self.attack_generator = AttackGenerator(self.attack_params, self.attack_start_step)

    def _get_observation(self) -> Observation:
        """
        Building the current Observation from internal windows.
        """
        return Observation(
            vq_window=list(self.vq_window),
            vd_window=list(self.vd_window),
            omega_window=list(self.omega_window),
            omega_deviation_window=list(self.omega_deviation_window),  # Fix 5
            raw_voltages=[self.pll.va_m, self.pll.vb_m, self.pll.vc_m],
            task_id=self.task_id,
            step=self.step_count,
        )

    def _compute_grader_score(self) -> float:
        """Running the appropriate grader at episode end."""
        if self.task_id == 0:
            return grade_task_easy(self.history, self.attack_start_step)
        elif self.task_id == 1:
            return grade_task_medium(self.history, self.attack_start_step)
        elif self.task_id == 2:
            return grade_task_hard(
                self.history,
                self.lock_loss_step,
                self.attack_start_step,
            )
        return 0.0
