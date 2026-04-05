"""
SRF-PLL Discrete-Time Simulation.

Implements the Synchronous Reference Frame Phase-Locked Loop used in
grid-connected inverters. Discrete time step Δt = 1 ms.

Steps:
  1. Generate true 3-phase grid voltages (50 Hz, 1.0 pu)
  2. Apply attack injection on va
  3. Clarke transform (αβ)
  4. Park transform (dq) using estimated angle θ̂
  5. PI controller to update ω̂ and θ̂
  6. Compute phase error
"""

import numpy as np
import math

# Constants
V_NOM = 1.0           # Nominal voltage (pu)
F0 = 50.0             # Grid frequency (Hz)
OMEGA0 = 2.0 * math.pi * F0  # Nominal angular freq (rad/s)
DT = 1e-3             # Time step (1 ms)
KP = 50.0             # PI proportional gain
KI = 1500.0           # PI integral gain


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SRFPLLSimulator:
    """Discrete-time SRF-PLL simulator."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset PLL state to initial conditions."""
        self.t = 0.0              # Simulation time (s)
        self.theta_true = 0.0     # True grid angle (rad)
        self.theta_hat = 0.0      # Estimated angle (rad)
        self.omega_hat = OMEGA0   # Estimated angular freq (rad/s)
        self.vq_integral = 0.0    # Integral of vq for PI controller

        # Current signal values
        self.vd = 0.0
        self.vq = 0.0
        self.va_m = 0.0
        self.vb_m = 0.0
        self.vc_m = 0.0
        self.theta_err = 0.0

    def step(self, attack_signal: float = 0.0):
        """
        Advance the PLL by one time step.
        Args:
            attack_signal: Attack injection added to va (pu).

        Returns:
            dict with vd, vq, omega_hat, theta_err, va_m, vb_m, vc_m, theta_true, theta_hat
        """
        # Step 1 — True three-phase grid voltages
        va = V_NOM * math.sin(self.theta_true)
        vb = V_NOM * math.sin(self.theta_true - 2.0 * math.pi / 3.0)
        vc = V_NOM * math.sin(self.theta_true + 2.0 * math.pi / 3.0)

        # Step 2 — Apply attack injection on va
        va_m = va + attack_signal
        vb_m = vb
        vc_m = vc

        # Step 3 — Clarke Transform (αβ)
        v_alpha = va_m
        v_beta = (va_m + 2.0 * vb_m) / math.sqrt(3.0)

        # Step 4 — Park Transform (dq) using estimated angle θ̂
        cos_th = math.cos(self.theta_hat)
        sin_th = math.sin(self.theta_hat)
        vd = v_alpha * cos_th + v_beta * sin_th
        vq = -v_alpha * sin_th + v_beta * cos_th

        # Step 5 — PI Controller
        self.vq_integral += vq * DT
        omega_hat = OMEGA0 + KP * vq + KI * self.vq_integral
        self.theta_hat += omega_hat * DT

        # Advance true angle
        self.theta_true += OMEGA0 * DT

        # Step 6 — Phase error wrapped to [-π, π]
        theta_err = wrap_angle(self.theta_hat - self.theta_true)

        # Update time
        self.t += DT

        # Storing current values
        self.vd = vd
        self.vq = vq
        self.omega_hat = omega_hat
        self.va_m = va_m
        self.vb_m = vb_m
        self.vc_m = vc_m
        self.theta_err = theta_err

        return {
            "vd": vd,
            "vq": vq,
            "omega_hat": omega_hat,
            "theta_err": theta_err,
            "va_m": va_m,
            "vb_m": vb_m,
            "vc_m": vc_m,
            "theta_true": self.theta_true,
            "theta_hat": self.theta_hat,
        }
