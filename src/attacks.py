"""
Attack injection logic for the PLL Cyberattack Detection OpenEnv.

Implements four attack types:
  1. Sinusoidal FDI (Easy)
  2. Ramp injection (Medium)
  3. Pulse/step bias (Medium)
  4. Stealthy low-and-slow phase drift (Hard)
"""

import math
import numpy as np
from typing import Dict, Any


def sample_sinusoidal_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample parameters for a sinusoidal FDI attack."""
    return {
        "type": "sinusoidal",
        "amplitude": float(rng.uniform(0.05, 0.20)),
        "freq": float(rng.uniform(5.0, 20.0)),
        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
    }


def sample_ramp_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample parameters for a ramp injection attack."""
    return {
        "type": "ramp",
        "rate": float(rng.uniform(0.0002, 0.001)),
    }


def sample_pulse_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample parameters for a pulse/step bias attack."""
    return {
        "type": "pulse",
        "magnitude": float(rng.uniform(0.1, 0.3)),
        "duration": int(rng.integers(20, 81)),  # 20 to 80 steps inclusive
    }


def sample_stealthy_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample parameters for a stealthy low-and-slow attack."""
    return {
        "type": "stealthy",
        "amplitude": 0.03,
        "drift_rate": float(rng.uniform(0.05, 0.2)),
    }


def sample_attack_start(rng: np.random.Generator) -> int:
    """Sample a random attack start step between 30 and 80 inclusive."""
    return int(rng.integers(30, 81))


class AttackGenerator:
    """Generates attack signals given parameters and current simulation state."""

    def __init__(self, attack_params: Dict[str, Any], attack_start_step: int):
        self.params = attack_params
        self.attack_start_step = attack_start_step
        self.attack_type_str = attack_params.get("type", "none")

        # For stealthy attack: track cumulative phase drift
        self.delta = 0.0

    def get_signal(self, current_step: int, sim_time: float) -> float:
        """
        Compute the attack signal value at the given step.
        Args:
            current_step: Current environment step (0-indexed).
            sim_time: Current simulation time in seconds.
        Returns:
            Attack signal value (pu). Returns 0.0 if attack not yet started.
        """
        if current_step < self.attack_start_step:
            return 0.0

        steps_since_start = current_step - self.attack_start_step
        dt = 1e-3  # time step

        if self.attack_type_str == "sinusoidal":
            A = self.params["amplitude"]
            fa = self.params["freq"]
            phi = self.params["phase"]
            return A * math.sin(2.0 * math.pi * fa * sim_time + phi)

        elif self.attack_type_str == "ramp":
            rate = self.params["rate"]
            return rate * steps_since_start

        elif self.attack_type_str == "pulse":
            mag = self.params["magnitude"]
            dur = self.params["duration"]
            if steps_since_start < dur:
                return mag
            else:
                return 0.0

        elif self.attack_type_str == "stealthy":
            A_s = self.params["amplitude"]
            drift_rate = self.params["drift_rate"]
            # δ(t) = δ(t-1) + drift_rate * Δt — accumulated each call
            self.delta += drift_rate * dt
            f0 = 50.0
            return A_s * math.sin(2.0 * math.pi * f0 * sim_time + self.delta)

        return 0.0

    def is_active(self, current_step: int) -> bool:
        """Checking if the attack is currently active at this step."""
        if current_step < self.attack_start_step:
            return False

        # Pulse attacks end after duration
        if self.attack_type_str == "pulse":
            steps_since_start = current_step - self.attack_start_step
            dur = self.params["duration"]
            return steps_since_start < dur

        return True


def get_attack_type_id(attack_type_str: str) -> int:
    """Mapping attack type string to integer ID."""
    mapping = {
        "none": 0,
        "sinusoidal": 1,
        "ramp": 2,
        "pulse": 3,
        "stealthy": 4,
    }
    return mapping.get(attack_type_str, 0)
