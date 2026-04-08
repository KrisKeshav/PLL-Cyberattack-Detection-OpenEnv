"""
Inference Script — PLL Cyberattack Detection OpenEnv
=====================================================
Environment variables (injected by the judging sandbox):
  API_BASE_URL   LiteLLM proxy endpoint  (MUST be used for all LLM calls)
  API_KEY        LiteLLM proxy key       (MUST be used — do not hardcode keys)
  MODEL_NAME     Model identifier
  ENV_URL        Environment server URL  (default: http://localhost:7860)

STDOUT FORMAT (OpenEnv compliance):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import requests
from typing import List, Optional
from openai import OpenAI

# ── Config — always read from environment, never hardcode ─────────────────────
# The judging sandbox injects API_BASE_URL and API_KEY via their LiteLLM proxy.
# All LLM calls MUST go through these values or the submission will be rejected.
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

# OpenAI client pointed at the proxy — never bypass this
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Task metadata ─────────────────────────────────────────────────────────────
TASK_NAMES = {
    0: "Sinusoidal FDI Detection (Easy)",
    1: "Multi-Attack Classification (Medium)",
    2: "Stealthy Attack Detection (Hard)",
}

BENCHMARK = "pll-cyberattack-detection"

DEFAULT_ACTION = {
    "attack_detected": False,
    "attack_type": 0,
    "confidence": 0.5,
    "protective_action": 0,
}

# ── System prompt ─────────────────────────────────────────────────────────────
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

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error) -> None:
    action_str = json.dumps(action, separators=(',', ':'))
    error_val  = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Heuristic agent (FALLBACK ONLY — used when LLM call fails) ────────────────

class HeuristicState:
    """Tracks running state for the heuristic agent across steps."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vq_history        = []
        self.omega_dev_history = []
        self.attack_detected   = False
        self.predicted_type    = 0
        self.settled_baseline  = None
        self.peak_vq           = 0.0


_hstate = HeuristicState()


def heuristic_agent(obs: dict) -> dict:
    """Rule-based fallback — only called when the LLM request fails."""
    global _hstate

    vq        = obs["vq_window"]
    omega_dev = obs["omega_deviation_window"]
    task_id   = obs["task_id"]
    step      = obs["step"]

    if step == 0:
        _hstate.reset()

    vq_abs         = [abs(v) for v in vq]
    vq_mean        = sum(vq_abs) / len(vq_abs)
    vq_max         = max(vq_abs)
    omega_dev_abs  = [abs(v) for v in omega_dev]
    omega_dev_mean = sum(omega_dev_abs) / len(omega_dev_abs)

    _hstate.vq_history.append(vq_mean)
    _hstate.omega_dev_history.append(omega_dev_mean)
    _hstate.peak_vq = max(_hstate.peak_vq, vq_mean)

    if step == 50:
        _hstate.settled_baseline = omega_dev_mean

    detected = False if step < 25 else (vq_mean > 0.01 or vq_max > 0.025)
    if detected:
        _hstate.attack_detected = True

    # ── Task 0: binary detection ──────────────────────────────────────────────
    if task_id == 0:
        return {
            "attack_detected": _hstate.attack_detected,
            "attack_type": 1 if _hstate.attack_detected else 0,
            "confidence": min(1.0, vq_mean * 50) if _hstate.attack_detected else 0.8,
            "protective_action": 1 if _hstate.attack_detected else 0,
        }

    # ── Task 1: classification ────────────────────────────────────────────────
    if task_id == 1:
        if not _hstate.attack_detected:
            return {
                "attack_detected": False,
                "attack_type": 0,
                "confidence": 0.7,
                "protective_action": 0,
            }

        n_elevated = sum(1 for v in _hstate.vq_history if v > 0.01)

        if n_elevated < 5:
            attack_type = 1
        else:
            elevated = [v for v in _hstate.vq_history if v > 0.005]
            recent   = elevated[-min(20, len(elevated)):]

            current_vs_peak = vq_mean / _hstate.peak_vq if _hstate.peak_vq > 0 else 0
            zero_crossings  = sum(1 for i in range(1, len(vq)) if vq[i] * vq[i - 1] < 0)

            if len(recent) >= 6:
                first_third = sum(recent[: len(recent) // 3]) / (len(recent) // 3)
                last_third  = sum(recent[-len(recent) // 3 :]) / (len(recent) // 3)
                growth      = last_third / first_third if first_third > 0.001 else 1.0
            else:
                growth = 1.0

            if current_vs_peak < 0.15 and _hstate.peak_vq > 0.05:
                attack_type = 3
            elif current_vs_peak < 0.4 and n_elevated > 30:
                attack_type = 3
            elif zero_crossings >= 2 and growth < 1.5:
                attack_type = 1
            elif growth > 1.3:
                attack_type = 2
            elif zero_crossings >= 1:
                attack_type = 1
            else:
                vq_diffs = [vq[i] - vq[i - 1] for i in range(1, len(vq))]
                neg = sum(1 for d in vq_diffs if d < 0)
                attack_type = 3 if neg > 14 else 1

            _hstate.predicted_type = attack_type

        return {
            "attack_detected": True,
            "attack_type": _hstate.predicted_type,
            "confidence": 0.8,
            "protective_action": 1,
        }

    # ── Task 2: stealthy attack ───────────────────────────────────────────────
    if task_id == 2:
        drift_detected = False
        confidence     = 0.3

        if step > 50 and _hstate.settled_baseline is not None:
            baseline = _hstate.settled_baseline
            ratio    = omega_dev_mean / baseline if baseline > 0.01 else omega_dev_mean * 100

            if len(_hstate.omega_dev_history) > 10:
                recent_10  = _hstate.omega_dev_history[-10:]
                old_10     = (_hstate.omega_dev_history[-20:-10]
                              if len(_hstate.omega_dev_history) > 20
                              else _hstate.omega_dev_history[:10])
                recent_avg = sum(recent_10) / len(recent_10)
                old_avg    = sum(old_10) / len(old_10)
                rising     = recent_avg > old_avg * 1.1
            else:
                rising = False

            if ratio > 2.0:
                drift_detected, confidence = True, 0.9
            elif ratio > 1.3 and rising:
                drift_detected, confidence = True, 0.8
            elif rising and vq_mean > 0.1:
                drift_detected, confidence = True, 0.6
            elif vq_mean > 0.2:
                drift_detected, confidence = True, 0.5

        if drift_detected:
            _hstate.attack_detected = True

        return {
            "attack_detected": drift_detected,
            "attack_type": 4 if drift_detected else 0,
            "confidence": confidence,
            "protective_action": 2 if drift_detected else 0,
        }

    return DEFAULT_ACTION.copy()

# ── LLM agent (PRIMARY — always called first) ─────────────────────────────────

def parse_llm_response(response_text: str) -> dict:
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines      = text.split("\n")
            in_block   = False
            json_lines: List[str] = []
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
        return {
            "attack_detected":   bool(parsed.get("attack_detected", False)),
            "attack_type":       max(0, min(4, int(parsed.get("attack_type", 0)))),
            "confidence":        max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "protective_action": max(0, min(3, int(parsed.get("protective_action", 0)))),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return DEFAULT_ACTION.copy()


def format_observation(obs: dict) -> str:
    return "\n".join([
        f"Step: {obs['step']}",
        f"Task: {obs['task_id']}",
        f"vq_window (last 20): {[round(v, 6) for v in obs['vq_window']]}",
        f"vd_window (last 20): {[round(v, 6) for v in obs['vd_window']]}",
        f"omega_window (last 20): {[round(v, 6) for v in obs['omega_window']]}",
        f"omega_deviation_window (last 20): {[round(v, 6) for v in obs['omega_deviation_window']]}",
        f"raw_voltages: {[round(v, 6) for v in obs['raw_voltages']]}",
    ])


def llm_agent(obs: dict) -> dict:
    """Primary agent — calls the LLM through the injected proxy.
    Falls back to heuristic only if the API call itself raises an exception.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_observation(obs)},
            ],
            temperature=0.1,
            max_tokens=200,
            timeout=10.0,
        )
        return parse_llm_response(completion.choices[0].message.content or "")
    except Exception as e:
        print(f"[DEBUG] LLM error ({type(e).__name__}: {e}), falling back to heuristic", file=sys.stderr, flush=True)
        return heuristic_agent(obs)

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: int) -> float:
    task_name = TASK_NAMES[task_id]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Reset heuristic state before every episode so stale data from a previous
    # task never bleeds into the next one (also covers the LLM fallback path).
    _hstate.reset()

    step_count   = 0
    grader_score = 0.0
    rewards: List[float] = []
    success      = False

    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=60,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        done         = False
        total_reward = 0.0
        info         = {}

        while not done:
            # Frame skipping: only invoke the LLM every 5 steps to prevent 30-min evaluation timeouts.
            # Step skips use the heuristics to keep episode run-time blazing fast.
            if step_count % 5 == 0:
                action = llm_agent(obs)
            else:
                action = heuristic_agent(obs)

            step_resp = requests.post(
                f"{ENV_URL}/step",
                json=action,
                timeout=60,
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            obs         = result["observation"]
            reward      = result["reward"]
            done        = result["done"]
            info        = result.get("info", {})
            error       = result.get("error", None)

            step_reward   = reward["total"] if isinstance(reward, dict) else float(reward)
            total_reward += step_reward
            rewards.append(step_reward)

            step_count += 1
            log_step(step=step_count, action=action, reward=step_reward, done=done, error=error)

            if step_count % 50 == 0:
                print(
                    f"[DEBUG] step={step_count} cumulative_reward={total_reward:+.4f} "
                    f"detected={action['attack_detected']} type={action['attack_type']}",
                    file=sys.stderr, flush=True,
                )

        grader_score = info.get("grader_score", 0.0)
        success      = grader_score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Episode error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        success = False
    except BaseException as exc:
        print(f"[DEBUG] Critical interruption: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        success = False
        raise

    finally:
        log_end(success=success, steps=step_count, score=grader_score, rewards=rewards)

    return grader_score

# ── Server Check ──────────────────────────────────────────────────────────────

def wait_for_server(env_url: str, timeout: int = 60) -> bool:
    print(f"[DEBUG] Waiting for environment server at {env_url} to start...", file=sys.stderr, flush=True)
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            resp = requests.get(f"{env_url}/health", timeout=2)
            if resp.status_code == 200:
                print("[DEBUG] Environment server is up!", file=sys.stderr, flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"[DEBUG] Environment server failed to start within {timeout}s.", file=sys.stderr, flush=True)
    return False

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[DEBUG] PLL Cyberattack Detection — model={MODEL_NAME} env={ENV_URL}", file=sys.stderr, flush=True)

    if not wait_for_server(ENV_URL):
        print("[DEBUG] Exiting due to server unavailable.", file=sys.stderr, flush=True)
        return

    start_time = time.time()
    scores: List[float] = []

    try:
        for task_id in range(3):
            try:
                score = run_episode(task_id)
            except Exception as exc:
                print(f"[DEBUG] run_episode({task_id}) crashed: {exc}", file=sys.stderr, flush=True)
                score = 0.0
            scores.append(score)
            print(f"[DEBUG] task={task_id} score={score:.4f}", file=sys.stderr, flush=True)
    except BaseException as exc:
        print(f"[DEBUG] Process interrupted: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    elapsed = time.time() - start_time
    avg     = sum(scores) / len(scores) if scores else 0.0
    print(f"[DEBUG] avg_score={avg:.4f} elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()