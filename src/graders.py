"""
Per-task deterministic graders for the PLL Cyberattack Detection OpenEnv.

Each grader takes an episode history and returns a score in [0.0, 1.0].
Graders are deterministic given the same episode data.
"""

from typing import List, Dict, Any, Optional


def grade_task_easy(history: List[Dict[str, Any]], attack_start_step: int) -> float:
    """
    Task 1 — Sinusoidal FDI Detection (Easy).

    Grader logic (relative to attack onset):
      delay = first_correct_detection_step - attack_start_step
      if delay <= 20:   score = 1.0
      elif delay <= 100: score = linear decay from 1.0 to 0.5
      elif delay <= 420: score = 0.2
      else (never detected): score = 0.0
    """
    first_correct_detection_step = None

    for entry in history:
        step = entry["step"]
        attack_active = entry["attack_active"]
        attack_detected = entry["attack_detected"]

        if attack_active and attack_detected:
            first_correct_detection_step = step
            break

    if first_correct_detection_step is None:
        return 0.0

    delay = first_correct_detection_step - attack_start_step

    if delay <= 20:
        return 1.0
    elif delay <= 100:
        # Linear decay from 1.0 at delay=20 to 0.5 at delay=100
        return 1.0 - 0.5 * (delay - 20) / 80.0
    elif delay <= 420:
        return 0.2
    else:
        return 0.0


def grade_task_medium(history: List[Dict[str, Any]], attack_start_step: int) -> float:
    """
    Task 2 — Multi-Attack Classification (Medium).

    Grader logic:
      base_score = fraction of steps (after attack_start) where attack_type is correctly classified
      early_bonus = 0.4 * max(0, 1 - first_correct_classification_step / 100)
      score = min(1.0, base_score * 0.6 + early_bonus)
    """
    steps_after_attack = 0
    correct_classifications = 0
    first_correct_classification_step = None

    for entry in history:
        step = entry["step"]
        if step < attack_start_step:
            continue

        steps_after_attack += 1
        true_type = entry["true_attack_type"]
        agent_type = entry["agent_attack_type"]

        if agent_type == true_type:
            correct_classifications += 1
            if first_correct_classification_step is None:
                first_correct_classification_step = step

    if steps_after_attack == 0:
        return 0.0

    base_score = correct_classifications / steps_after_attack

    if first_correct_classification_step is not None:
        early_bonus = 0.4 * max(0.0, 1.0 - first_correct_classification_step / 100.0)
    else:
        early_bonus = 0.0

    score = min(1.0, base_score * 0.6 + early_bonus)
    return max(0.0, score)


def grade_task_hard(
    history: List[Dict[str, Any]],
    loss_of_lock_step: Optional[int],
    attack_start_step: int,
) -> float:
    """
    Task 3 — Stealthy Low-and-Slow Attack (Hard).

    Grader logic:
      if detected before loss_of_lock_step:
          score = 1.0 * (1 - first_detection_step / loss_of_lock_step)
      elif detected after loss_of_lock but before episode end:
          score = 0.3
      else (never detected):
          score = 0.0
      false_alarm_penalty = 0.2 per false alarm before attack starts
      (capped at reducing score to 0.0 minimum)
    """
    first_detection_step = None
    false_alarm_count = 0

    for entry in history:
        step = entry["step"]
        attack_active = entry["attack_active"]
        attack_detected = entry["attack_detected"]

        # Only count false alarms before the attack starts
        if attack_detected and not attack_active and step < attack_start_step:
            false_alarm_count += 1

        if attack_detected and attack_active and first_detection_step is None:
            first_detection_step = step

    # Compute base score
    if first_detection_step is None:
        score = 0.0
    elif loss_of_lock_step is not None and first_detection_step < loss_of_lock_step:
        score = 1.0 * (1.0 - first_detection_step / loss_of_lock_step)
    elif loss_of_lock_step is not None and first_detection_step >= loss_of_lock_step:
        score = 0.3
    else:
        # No loss of lock occurred but attack was detected
        score = 0.3

    # Apply false alarm penalty
    penalty = 0.2 * false_alarm_count
    score = max(0.0, score - penalty)

    return min(1.0, score)