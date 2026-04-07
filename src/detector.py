"""
Adaptive Physics-informed cyberattack detector for the PLL OpenEnv.

Uses residual-based and pattern-based features derived from the
observation windows to detect, classify, and recommend protective
actions. The detector builds a baseline from the first 20 observations
(warmup window) of the episode to normalize the features individually.
"""

import numpy as np
from typing import Dict, Any

from src.models import Observation


class AdaptiveDetector:
    def __init__(self):
        # Baseline collections
        self.r1_history = []
        self.r3_history = []
        self.r4_history = []
        self.r5_history = []

        # Calibrated statistics
        self.mean_R1 = 0.0
        self.std_R1 = 1e-6
        self.mean_R3 = 0.0
        self.std_R3 = 1e-6
        self.mean_R4 = 0.0
        self.std_R4 = 1e-6
        self.mean_R5 = 0.0
        self.std_R5 = 1e-6

        self.is_calibrated = False

    def detect(self, observation) -> Dict[str, Any]:
        """
        Run physics-informed anomaly detection on the current observation.
        """
        vq = np.array(observation.vq_window, dtype=np.float64)
        vd = np.array(observation.vd_window, dtype=np.float64)
        omega = np.array(observation.omega_window, dtype=np.float64)
        omega_dev = np.array(observation.omega_deviation_window, dtype=np.float64)
        va, vb, vc = observation.raw_voltages

        # ---- Step 1: Feature Extraction --------------------------------
        vq_mean = float(np.mean(np.abs(vq)))
        vd_mean = float(np.mean(np.abs(vd)))
        vq_ratio = vq_mean / (vd_mean + 1e-6)

        omega_var = float(np.var(omega))
        omega_dev_var = float(np.var(omega_dev))
        vd_var = float(np.var(vd))
        
        abs_v_sum = abs(va) + abs(vb) + abs(vc) + 1e-6
        symmetry_ratio = float(abs(va + vb + vc) / abs_v_sum)

        vq_diff = np.diff(vq) if len(vq) > 1 else np.array([0.0])
        vq_trend = float(np.mean(vq_diff))
        vq_spike = float(np.max(np.abs(vq)))
        vq_drift = float(np.sum(vq))

        step = observation.step

        # ---- Step 2: Baseline Calibration ------------------------------
        if step < 20:
            self.r1_history.append(vq_ratio)
            self.r3_history.append(omega_var)
            self.r4_history.append(vd_var)
            self.r5_history.append(symmetry_ratio)
            
            return {
                "attack_detected": False,
                "attack_type": 0,
                "confidence": 0.0,
                "protective_action": 0,
                "score": 0.0,
                "baseline_score": 0.0
            }
        
        if not self.is_calibrated:
            self.mean_R1 = float(np.mean(self.r1_history))
            self.std_R1 = max(float(np.std(self.r1_history)), 1e-6)
            
            self.mean_R3 = float(np.mean(self.r3_history))
            self.std_R3 = max(float(np.std(self.r3_history)), 1e-6)
            
            self.mean_R4 = float(np.mean(self.r4_history))
            self.std_R4 = max(float(np.std(self.r4_history)), 1e-6)
            
            self.mean_R5 = float(np.mean(self.r5_history))
            self.std_R5 = max(float(np.std(self.r5_history)), 1e-6)
            
            self.is_calibrated = True

        # ---- Step 3: Normalized Features ------------------------------
        R1 = (vq_ratio - self.mean_R1) / self.std_R1
        R3 = (omega_var - self.mean_R3) / self.std_R3
        R4 = (vd_var - self.mean_R4) / self.std_R4
        R5 = (symmetry_ratio - self.mean_R5) / self.std_R5

        # ---- Step 4: Score --------------------------------------------
        score = 0.4 * R1 + 0.2 * R3 + 0.2 * R5 + 0.2 * R4

        # ---- Step 5: Detection ----------------------------------------
        attack_detected = score > 5.0
        confidence = min(1.0, score / 5.0) if attack_detected else 0.0

        # ---- Step 6: Classification -----------------------------------
        if not attack_detected:
            attack_type = 0
        else:
            if R3 > 2:
                attack_type = 1  # sinusoidal
            elif abs(vq_trend) > 0.01:
                attack_type = 2  # ramp
            elif vq_spike > 0.1:
                attack_type = 3  # pulse
            else:
                attack_type = 4  # stealthy

        # ---- Step 7: Protective Action --------------------------------
        if score > 6:
            protective_action = 3
        elif score > 3:
            protective_action = 2
        else:
            protective_action = 1
            if not attack_detected:
                protective_action = 0

        return {
            "attack_detected": bool(attack_detected),
            "attack_type": int(attack_type),
            "confidence": float(confidence),
            "protective_action": int(protective_action),
            "score": float(score),
            "baseline_score": 0.0
        }
