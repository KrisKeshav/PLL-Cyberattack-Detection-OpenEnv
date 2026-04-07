"""
Inference Script — PLL Cyberattack Detection OpenEnv
=====================================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM
  MODEL_NAME     The model identifier to use
  HF_TOKEN       Your Hugging Face / API key

Uses a HYBRID approach:
  - A fast rule-based heuristic agent runs by default (no LLM needed)
  - The heuristic analyzes vq/omega_deviation windows to detect attacks
  - Set USE_LLM=1 env var to use the LLM instead (slower, may fail)

Must be named inference.py and placed at the project root.
Uses OpenAI client for LLM calls when enabled.
"""

import os
import json
from typing import List, Optional
import time
import math
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "https://krishuggingface-cyberattack-pll.hf.space")
USE_LLM = os.environ.get("USE_LLM", "0") == "1"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an AI agent monitoring a power grid inverter's Phase-Locked Loop (PLL).
You receive time-windowed sensor readings each step and must detect cyberattacks.

vq_window: q-axis voltage error (should be ~0 when healthy)
vd_window: d-axis voltage
omega_window: estimated frequency (normalized, nominal=0)
omega_deviation_window: frequency deviation from nominal in rad/s (useful for detecting slow phase drift)
raw_voltages: [va, vb, vc] at current step
task_id: 0=detect only, 1=classify type, 2=detect stealthy attack

For task_id=0: Focus on detecting any attack (attack_detected=True/False).
For task_id=1: Also classify the attack type (1=sinusoidal, 2=ramp, 3=pulse).
For task_id=2: Detect very subtle attacks before the PLL loses lock. Look for slow drifts in omega_deviation and vq.

Analysis tips:
- In healthy state, vq values should be near 0 and stable.
- Sinusoidal attacks cause oscillating patterns in vq.
- Ramp attacks cause steadily increasing vq magnitude.
- Pulse attacks cause sudden step changes in vq.
- Stealthy attacks cause very slow, gradual drift in omega_deviation_window.
- Look at trends across the full window, not just the latest value.

Respond ONLY with valid JSON, no explanation:
{
  "attack_detected": <bool>,
  "attack_type": <int 0-4>,
  "confidence": <float 0.0-1.0>,
  "protective_action": <int 0-3>
}"""

TASK_NAMES = {
    0: "Sinusoidal FDI Detection (Easy)",
    1: "Multi-Attack Classification (Medium)",
    2: "Stealthy Attack Detection (Hard)",
}

DEFAULT_ACTION = {
    "attack_detected": False,
    "attack_type": 0,
    "confidence": 0.5,
    "protective_action": 0,
}


# =====================================================================
# Logging Helpers (OpenEnv compliance)
# =====================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error) -> None:
    action_str = json.dumps(action, separators=(',', ':'))
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# =====================================================================
# Detector-Based Agent
# =====================================================================

def detector_agent(prev_info: dict) -> Optional[dict]:
    """Reads the environment's adaptive detector output from the previous step."""
    det = prev_info.get("detector", {})
    if not det or "attack_detected" not in det:
        return None
        
    return {
        "attack_detected": det.get("attack_detected", False),
        "attack_type": det.get("attack_type", 0),
        "confidence": det.get("confidence", 0.5),
        "protective_action": det.get("protective_action", 0),
    }


# =====================================================================
# Rule-Based Heuristic Agent
# =====================================================================

class HeuristicState:
    """Tracks running state for the heuristic agent across steps."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.vq_history = []           # all vq_mean(abs) values
        self.omega_dev_history = []    # all omega_dev_mean(abs) values
        self.attack_detected = False   # latched detection flag
        self.predicted_type = 0        # latched classification
        self.settled_baseline = None   # omega_dev baseline when PLL settles
        self.peak_vq = 0.0            # highest vq_mean seen


_hstate = HeuristicState()


def heuristic_agent(obs: dict) -> dict:
    """
    Rule-based attack detector using cumulative state tracking.
    No LLM needed — runs instantly.

    The key insight is that the PLL's closed-loop response transforms
    attack signals, so we track statistics over time rather than
    trying to classify from a single 20-step vq window shape.
    """
    global _hstate
    vq = obs["vq_window"]
    omega_dev = obs["omega_deviation_window"]
    task_id = obs["task_id"]
    step = obs["step"]

    if step == 0:
        _hstate.reset()

    # --- Compute per-step features ---
    vq_abs = [abs(v) for v in vq]
    vq_mean = sum(vq_abs) / len(vq_abs)
    vq_max = max(vq_abs)
    vq_latest = abs(vq[-1])

    omega_dev_abs = [abs(v) for v in omega_dev]
    omega_dev_mean = sum(omega_dev_abs) / len(omega_dev_abs)

    # Track history
    _hstate.vq_history.append(vq_mean)
    _hstate.omega_dev_history.append(omega_dev_mean)
    _hstate.peak_vq = max(_hstate.peak_vq, vq_mean)

    # Record baseline around step 45-50 (PLL settled)
    if step == 50:
        _hstate.settled_baseline = omega_dev_mean

    # -----------------------------------------------------------------
    # Detection: is vq significantly elevated?
    # After PLL warm-start settles (~step 20-30), healthy vq < 0.005
    # -----------------------------------------------------------------
    if step < 25:
        # PLL still settling, don't detect
        detected = False
    else:
        detected = vq_mean > 0.01 or vq_max > 0.025

    # Latch detection on
    if detected:
        _hstate.attack_detected = True

    # -----------------------------------------------------------------
    # Task 0: Binary detection only
    # -----------------------------------------------------------------
    if task_id == 0:
        return {
            "attack_detected": _hstate.attack_detected,
            "attack_type": 1 if _hstate.attack_detected else 0,
            "confidence": min(1.0, vq_mean * 50) if _hstate.attack_detected else 0.8,
            "protective_action": 1 if _hstate.attack_detected else 0,
        }

    # -----------------------------------------------------------------
    # Task 1: Classification using cumulative patterns
    # -----------------------------------------------------------------
    if task_id == 1:
        if not _hstate.attack_detected:
            return {
                "attack_detected": False,
                "attack_type": 0,
                "confidence": 0.7,
                "protective_action": 0,
            }

        # Classify using cumulative vq_history
        # Only classify after enough attack data (10+ steps of elevated vq)
        n_elevated = sum(1 for v in _hstate.vq_history if v > 0.01)

        if n_elevated < 5:
            # Not enough data yet, use simple guess
            attack_type = 1
        else:
            # Get recent vq trend (last 10 elevated values)
            elevated = [v for v in _hstate.vq_history if v > 0.005]
            recent = elevated[-min(20, len(elevated)):]

            # Feature 1: Is vq currently high or has it decayed?
            current_vs_peak = vq_mean / _hstate.peak_vq if _hstate.peak_vq > 0 else 0

            # Feature 2: How many zero crossings in current window
            zero_crossings = sum(1 for i in range(1, len(vq)) if vq[i] * vq[i-1] < 0)

            # Feature 3: Is vq growing or shrinking over recent history
            if len(recent) >= 6:
                first_third = sum(recent[:len(recent)//3]) / (len(recent)//3)
                last_third = sum(recent[-len(recent)//3:]) / (len(recent)//3)
                growth = last_third / first_third if first_third > 0.001 else 1.0
            else:
                growth = 1.0

            # Classification logic:
            # Sinusoidal: persistent oscillation, zero crossings, stable amplitude
            # Ramp: growing vq over time (growth > 1)
            # Pulse: high initial vq that decays to near zero (current_vs_peak < 0.3)

            if current_vs_peak < 0.15 and _hstate.peak_vq > 0.05:
                # vq has decayed significantly from peak → pulse (ended)
                attack_type = 3
            elif current_vs_peak < 0.4 and n_elevated > 30:
                # vq decayed after a long time → pulse
                attack_type = 3
            elif zero_crossings >= 2 and growth < 1.5:
                # Active oscillation without growing → sinusoidal
                attack_type = 1
            elif growth > 1.3:
                # Growing signal → ramp
                attack_type = 2
            elif zero_crossings >= 1:
                # Some oscillation → sinusoidal
                attack_type = 1
            else:
                # Default: if mono-decrease, pulse; else sinusoidal
                vq_diffs = [vq[i] - vq[i-1] for i in range(1, len(vq))]
                neg = sum(1 for d in vq_diffs if d < 0)
                if neg > 14:  # 14/19 = 73% decreasing
                    attack_type = 3
                else:
                    attack_type = 1

            _hstate.predicted_type = attack_type

        return {
            "attack_detected": True,
            "attack_type": _hstate.predicted_type,
            "confidence": 0.8,
            "protective_action": 1,
        }

    # -----------------------------------------------------------------
    # Task 2: Stealthy attack — detect omega_dev rising above baseline
    # -----------------------------------------------------------------
    if task_id == 2:
        drift_detected = False
        confidence = 0.3

        if step > 50 and _hstate.settled_baseline is not None:
            baseline = _hstate.settled_baseline

            # Compare current to baseline
            ratio = omega_dev_mean / baseline if baseline > 0.01 else omega_dev_mean * 100

            # Check if omega_dev is rising relative to recent history
            if len(_hstate.omega_dev_history) > 10:
                recent_10 = _hstate.omega_dev_history[-10:]
                old_10 = _hstate.omega_dev_history[-20:-10] if len(_hstate.omega_dev_history) > 20 else _hstate.omega_dev_history[:10]
                recent_avg = sum(recent_10) / len(recent_10)
                old_avg = sum(old_10) / len(old_10)
                rising = recent_avg > old_avg * 1.1
            else:
                rising = False

            if ratio > 2.0:
                drift_detected = True
                confidence = 0.9
            elif ratio > 1.3 and rising:
                drift_detected = True
                confidence = 0.8
            elif rising and vq_mean > 0.1:
                drift_detected = True
                confidence = 0.6
            elif vq_mean > 0.2:
                drift_detected = True
                confidence = 0.5

        if drift_detected:
            _hstate.attack_detected = True

        return {
            "attack_detected": drift_detected,
            "attack_type": 4 if drift_detected else 0,
            "confidence": confidence,
            "protective_action": 2 if drift_detected else 0,
        }

    return DEFAULT_ACTION.copy()


# =====================================================================
# LLM Agent (optional, set USE_LLM=1)
# =====================================================================

def parse_llm_response(response_text: str) -> dict:
    """Parse LLM response JSON, returning default action on failure."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip().startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        parsed = json.loads(text)
        action = {
            "attack_detected": bool(parsed.get("attack_detected", False)),
            "attack_type": max(0, min(4, int(parsed.get("attack_type", 0)))),
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "protective_action": max(0, min(3, int(parsed.get("protective_action", 0)))),
        }
        return action
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return DEFAULT_ACTION.copy()


def format_observation(obs: dict) -> str:
    """Format observation dict into a concise string for the LLM."""
    parts = [
        f"Step: {obs['step']}",
        f"Task: {obs['task_id']}",
        f"vq_window (last 20): {[round(v, 6) for v in obs['vq_window']]}",
        f"vd_window (last 20): {[round(v, 6) for v in obs['vd_window']]}",
        f"omega_window (last 20): {[round(v, 6) for v in obs['omega_window']]}",
        f"omega_deviation_window (last 20): {[round(v, 6) for v in obs['omega_deviation_window']]}",
        f"raw_voltages: {[round(v, 6) for v in obs['raw_voltages']]}",
    ]
    return "\n".join(parts)


def llm_agent(obs: dict) -> dict:
    """Call the LLM to decide an action. Falls back to heuristic on error."""
    try:
        obs_text = format_observation(obs)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        llm_response = completion.choices[0].message.content
        return parse_llm_response(llm_response)
    except Exception as e:
        print(f"    LLM error ({type(e).__name__}: {e}), falling back to heuristic")
        return heuristic_agent(obs)


# =====================================================================
# Episode Runner
# =====================================================================

def run_episode(task_id: int) -> float:
    log_start(task=TASK_NAMES[task_id], env="pll-cyberattack-detection", model=MODEL_NAME if USE_LLM else "rule-based-heuristic")

    print(f"\n{'='*60}")
    print(f"Task {task_id}: {TASK_NAMES[task_id]}")
    print(f"Agent: {'LLM (' + MODEL_NAME + ')' if USE_LLM else 'Rule-Based Heuristic'}")
    print(f"{'='*60}")

    step_count = 0
    grader_score = 0.0
    rewards = []

    try:
        # Reset environment
        reset_response = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        reset_response.raise_for_status()
        obs = reset_response.json()

        done = False
        total_reward = 0.0
        prev_info = {}

        while not done:
            # Choose agent
            if USE_LLM:
                action = llm_agent(obs)
            else:
                if step_count == 0:
                    action = DEFAULT_ACTION.copy()
                else:
                    det_action = detector_agent(prev_info) if "detector" in prev_info else None
                    heur_action = heuristic_agent(obs)
                    
                    if not det_action:
                        action = heur_action
                    elif det_action["confidence"] < 0.5:
                        action = heur_action
                    else:
                        action = heur_action

            # Step environment
            step_response = requests.post(
                f"{ENV_URL}/step",
                json=action,
                timeout=30,
            )
            step_response.raise_for_status()
            result = step_response.json()

            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            info = result["info"]
            total_reward += reward["total"]
            rewards.append(reward["total"])
            log_step(step=step_count, action=action, reward=reward["total"], done=done, error=None)

            prev_info = info
            step_count += 1

            # Print progress every 50 steps
            if step_count % 50 == 0:
                print(f"  Step {step_count:3d} | Reward: {reward['total']:+.4f} | "
                      f"Cumulative: {total_reward:+.4f} | "
                      f"Detected: {action['attack_detected']} | "
                      f"Type: {action['attack_type']}")

        # Extract grader score
        grader_score = info.get("grader_score", 0.0)
        print(f"\n  Episode complete: {step_count} steps")
        print(f"  Total reward: {total_reward:+.4f}")
        print(f"  Grader score: {grader_score:.4f}")
    finally:
        log_end(success=grader_score > 0.0, steps=step_count, score=grader_score, rewards=rewards)

    return grader_score


if __name__ == "__main__":
    agent_name = f"LLM ({MODEL_NAME})" if USE_LLM else "Rule-Based Heuristic"
    print("PLL Cyberattack Detection — Agentic Inference")
    print(f"Agent: {agent_name}")
    print(f"Environment: {ENV_URL}")
    if not USE_LLM:
        print("(Set USE_LLM=1 to use LLM agent instead of heuristic)")

    start_time = time.time()
    scores = []

    for task_id in range(3):
        score = run_episode(task_id)
        print(f"Task {task_id} score: {score:.4f}")
        scores.append(score)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for i, score in enumerate(scores):
        print(f"  Task {i} ({TASK_NAMES[i]}): {score:.4f}")
    print(f"\n  Average score: {sum(scores)/len(scores):.4f}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
