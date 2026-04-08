import os
import json
import time
import logging
import traceback
import threading
from typing import Optional, Dict, Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------
# 1. SETUP LOGGING
# ---------------------------------------------------------------------
# Ensure logs look like: [TIMESTAMP] [STAGE] message
class StageFormatter(logging.Formatter):
    def format(self, record):
        # We manually use the prefix if provided in extra
        stage = getattr(record, 'stage', 'SYSTEM')
        self._style._fmt = f"[%(asctime)s] [{stage}] %(message)s"
        # Ensure fast formatting matching standard requirements
        return super().format(record)

logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(StageFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

logger.info("Initializing Agent Scripts", extra={"stage": "APP STARTUP"})

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860")
USE_LLM = os.environ.get("USE_LLM", "0") == "1"

logger.info("Environment variables loaded.", extra={"stage": "APP STARTUP"})

client: Optional[OpenAI] = None
if API_BASE_URL and API_KEY:
    try:
        logger.info("Initializing OpenAI Client", extra={"stage": "MODEL LOADING"})
        _start_time = time.time()
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        _end_time = time.time()
        logger.info(f"Client Initialized. Duration: {_end_time - _start_time:.4f}s", extra={"stage": "MODEL LOADING"})
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}\n{traceback.format_exc()}", extra={"stage": "APP STARTUP"})
        client = None

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


def log_start(task: str, env: str, model: str) -> None:
    logger.info(f"task={task} env={env} model={model}", extra={"stage": "EPISODE START"})


def log_step(step: int, action: dict, reward: float, done: bool, error) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    error_val = error if error else "null"
    logger.debug(
        f"step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        extra={"stage": "EPISODE STEP"}
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    logger.info(
        f"success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        extra={"stage": "EPISODE END"}
    )


def safe_action(action: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {
            "attack_detected": bool(action.get("attack_detected", False)),
            "attack_type": max(0, min(4, int(action.get("attack_type", 0)))),
            "confidence": max(0.0, min(1.0, float(action.get("confidence", 0.5)))),
            "protective_action": max(0, min(3, int(action.get("protective_action", 0)))),
        }
    except Exception as e:
        logger.error(f"Action constraint failed: {e}\n{traceback.format_exc()}", extra={"stage": "POSTPROCESSING"})
        return DEFAULT_ACTION.copy()


def safe_post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 10,
    retries: int = 2,
) -> Optional[Dict[str, Any]]:
    last_error = None
    logger.debug(f"Calling endpoint {url}", extra={"stage": "API CALL (REQ)"})
    _start_t = time.time()
    
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            logger.debug(f"Response ok from {url} in {time.time()-_start_t:.4f}s", extra={"stage": "API CALL (RES)"})
            return response.json()
        except Exception as e:
            last_error = e
            logger.warning(
                f"HTTP error calling {url} (attempt {attempt + 1}/{retries + 1}): {e}",
                extra={"stage": "API CALL (ERR)"}
            )
            time.sleep(0.5)
    
    logger.error(f"Giving up on {url}: {last_error}\n{traceback.format_exc()}", extra={"stage": "API CALL (ERR)"})
    return None


def _warmup_worker() -> None:
    """Non-blocking LLM warmup executed inside a thread."""
    if client is None:
        logger.info("LLM proxy warmup skipped (client unavailable).", extra={"stage": "MODEL LOADING"})
        return

    logger.info("Initializing LLM Proxy Warmup Thread...", extra={"stage": "MODEL LOADING"})
    _req_t = time.time()
    try:
        _ = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        logger.info(f"LLM proxy warmup successful in {time.time() - _req_t:.4f}s.", extra={"stage": "MODEL LOADING"})
    except Exception as e:
        logger.error(f"LLM proxy warmup failed: {e}\n{traceback.format_exc()}", extra={"stage": "MODEL LOADING (ERR)"})

def warmup_proxy() -> None:
    """Make one tiny proxy call gracefully via threading to avoid app blocking"""
    t = threading.Thread(target=_warmup_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------
# ZERO-DEPENDENCY HEALTHCHECK SERVER
# ---------------------------------------------------------------------
from http.server import BaseHTTPRequestHandler, HTTPServer

class FastHealthcheck(BaseHTTPRequestHandler):
    def do_GET(self):
        logger.info(f"Healthcheck triggered at {self.path}", extra={"stage": "HEALTHCHECK"})
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
        logger.info("Healthcheck returned 200 OK immediately", extra={"stage": "HEALTHCHECK"})
        
    def log_message(self, format, *args):
        pass  # disable default stdout spam from simple server

def _run_healthcheck() -> None:
    try:
        # Binding to 7860 as Spaces default checks it
        server = HTTPServer(('0.0.0.0', 7860), FastHealthcheck)
        logger.info("Background Healthcheck server bound to 0.0.0.0:7860", extra={"stage": "APP STARTUP"})
        server.serve_forever()
    except Exception as e:
        logger.error(f"Healthcheck server crash: {e}\n{traceback.format_exc()}", extra={"stage": "APP STARTUP (ERR)"})

# Start Healthcheck Thread instantly
t_health = threading.Thread(target=_run_healthcheck, daemon=True)
t_health.start()


def detector_agent(prev_info: dict) -> Optional[dict]:
    det = (prev_info or {}).get("detector", {})
    if not isinstance(det, dict) or "attack_detected" not in det:
        return None
    return {
        "attack_detected": det.get("attack_detected", False),
        "attack_type": det.get("attack_type", 0),
        "confidence": det.get("confidence", 0.5),
        "protective_action": det.get("protective_action", 0),
    }


class HeuristicState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vq_history = []
        self.omega_dev_history = []
        self.attack_detected = False
        self.predicted_type = 0
        self.settled_baseline = None
        self.peak_vq = 0.0


_hstate = HeuristicState()


def heuristic_agent(obs: dict) -> dict:
    global _hstate

    try:
        vq = obs["vq_window"]
        omega_dev = obs["omega_deviation_window"]
        task_id = int(obs["task_id"])
        step = int(obs["step"])
    except Exception:
        return DEFAULT_ACTION.copy()

    if step == 0:
        _hstate.reset()

    try:
        vq_abs = [abs(v) for v in vq]
        vq_mean = sum(vq_abs) / len(vq_abs)
        vq_max = max(vq_abs)
        vq_latest = abs(vq[-1]) if vq else 0.0

        omega_dev_abs = [abs(v) for v in omega_dev]
        omega_dev_mean = sum(omega_dev_abs) / len(omega_dev_abs) if omega_dev_abs else 0.0

        _hstate.vq_history.append(vq_mean)
        _hstate.omega_dev_history.append(omega_dev_mean)
        _hstate.peak_vq = max(_hstate.peak_vq, vq_mean)

        if step == 50:
            _hstate.settled_baseline = omega_dev_mean

        if step < 25:
            detected = False
        else:
            detected = vq_mean > 0.01 or vq_max > 0.025

        if detected:
            _hstate.attack_detected = True

        if task_id == 0:
            return {
                "attack_detected": _hstate.attack_detected,
                "attack_type": 1 if _hstate.attack_detected else 0,
                "confidence": min(1.0, vq_mean * 50) if _hstate.attack_detected else 0.8,
                "protective_action": 1 if _hstate.attack_detected else 0,
            }

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
                recent = elevated[-min(20, len(elevated)):]

                current_vs_peak = vq_mean / _hstate.peak_vq if _hstate.peak_vq > 0 else 0.0
                zero_crossings = sum(1 for i in range(1, len(vq)) if vq[i] * vq[i - 1] < 0)

                if len(recent) >= 6:
                    third = max(1, len(recent) // 3)
                    first_third = sum(recent[:third]) / third
                    last_third = sum(recent[-third:]) / third
                    growth = last_third / first_third if first_third > 0.001 else 1.0
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

        if task_id == 2:
            drift_detected = False
            confidence = 0.3

            if step > 50 and _hstate.settled_baseline is not None:
                baseline = _hstate.settled_baseline
                ratio = omega_dev_mean / baseline if baseline > 0.01 else omega_dev_mean * 100.0

                if len(_hstate.omega_dev_history) > 10:
                    recent_10 = _hstate.omega_dev_history[-10:]
                    old_10 = (
                        _hstate.omega_dev_history[-20:-10]
                        if len(_hstate.omega_dev_history) > 20
                        else _hstate.omega_dev_history[:10]
                    )
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

    except Exception as e:
        logger.warning(f"heuristic_agent failed: {e}\n{traceback.format_exc()}", extra={"stage": "HEURISTIC AGENT (ERR)"})
        return DEFAULT_ACTION.copy()


def parse_llm_response(response_text: str) -> dict:
    try:
        text = (response_text or "").strip()
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
        return safe_action(
            {
                "attack_detected": parsed.get("attack_detected", False),
                "attack_type": parsed.get("attack_type", 0),
                "confidence": parsed.get("confidence", 0.5),
                "protective_action": parsed.get("protective_action", 0),
            }
        )
    except Exception:
        return DEFAULT_ACTION.copy()


def format_observation(obs: dict) -> str:
    try:
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
    except Exception:
        return ""


def llm_agent(obs: dict) -> dict:
    if client is None:
        return heuristic_agent(obs)

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
        llm_response = completion.choices[0].message.content if completion and completion.choices else ""
        return parse_llm_response(llm_response)
    except Exception as e:
        logger.warning(f"LLM error ({type(e).__name__}: {e})\n{traceback.format_exc()}", extra={"stage": "LLM AGENT (ERR)"})
        return heuristic_agent(obs)


def choose_action(obs: dict, prev_info: dict) -> dict:
    # Preserve the baseline heuristic behavior by default.
    try:
        if USE_LLM and client is not None:
            return safe_action(llm_agent(obs))
    except Exception:
        pass
    return safe_action(heuristic_agent(obs))


def run_episode(task_id: int) -> float:
    log_start(
        task=TASK_NAMES[task_id],
        env="pll-cyberattack-detection",
        model=MODEL_NAME if USE_LLM else "rule-based-heuristic",
    )

    print(f"\n{'=' * 60}")
    print(f"Task {task_id}: {TASK_NAMES[task_id]}")
    print(f"Agent: {'LLM (' + MODEL_NAME + ')' if USE_LLM else 'Rule-Based Heuristic'}")
    print(f"{'=' * 60}")

    step_count = 0
    grader_score = 0.0
    rewards = []
    info: Dict[str, Any] = {}
    prev_info: Dict[str, Any] = {}

    try:
        reset_result = safe_post_json(
            f"{ENV_URL}/reset",
            {"task_id": task_id},
            timeout=10,
            retries=2,
        )
        if not isinstance(reset_result, dict):
            logger.error("Reset failed; skipping episode.", extra={"stage": "ENV RESET"})
            return 0.0

        obs = reset_result
        done = False
        total_reward = 0.0

        while not done:
            try:
                action = choose_action(obs, prev_info)
            except Exception as e:
                logger.warning(f"Action selection failed: {e}\n{traceback.format_exc()}", extra={"stage": "ACTION SELECTION"})
                action = DEFAULT_ACTION.copy()

            result = safe_post_json(
                f"{ENV_URL}/step",
                action,
                timeout=10,
                retries=2,
            )
            if not isinstance(result, dict):
                logger.error("Step failed; ending episode early.", extra={"stage": "ENV STEP"})
                break

            obs = result.get("observation", obs)
            reward = result.get("reward", {})
            done = bool(result.get("done", False))
            info = result.get("info", {})

            step_reward = 0.0
            if isinstance(reward, dict):
                try:
                    step_reward = float(reward.get("total", 0.0))
                except Exception:
                    step_reward = 0.0

            total_reward += step_reward
            rewards.append(step_reward)
            log_step(step=step_count, action=action, reward=step_reward, done=done, error=None)

            prev_info = info if isinstance(info, dict) else {}
            step_count += 1

            if step_count % 50 == 0:
                print(
                    f"  Step {step_count:3d} | Reward: {step_reward:+.4f} | "
                    f"Cumulative: {total_reward:+.4f} | "
                    f"Detected: {action.get('attack_detected', False)} | "
                    f"Type: {action.get('attack_type', 0)}",
                    flush=True,
                )

        if isinstance(info, dict):
            try:
                grader_score = float(info.get("grader_score", 0.0))
            except Exception:
                grader_score = 0.0

        print(f"\n  Episode complete: {step_count} steps")
        print(f"  Total reward: {total_reward:+.4f}")
        print(f"  Grader score: {grader_score:.4f}")

        return grader_score

    except Exception as e:
        logger.error(f"Episode crashed safely: {e}\n{traceback.format_exc()}", extra={"stage": "EPISODE SEVERE ERR"})
        return 0.0

    finally:
        log_end(success=grader_score > 0.0, steps=step_count, score=grader_score, rewards=rewards)


if __name__ == "__main__":
    agent_name = f"LLM ({MODEL_NAME})" if USE_LLM else "Rule-Based Heuristic"
    logger.info("PLL Cyberattack Detection — Agentic Inference", extra={"stage": "APP STARTUP"})
    logger.info(f"Agent: {agent_name}", extra={"stage": "APP STARTUP"})
    logger.info(f"Environment: {ENV_URL}", extra={"stage": "APP STARTUP"})
    if not USE_LLM:
        logger.info("(Set USE_LLM=1 to use LLM agent instead of heuristic)", extra={"stage": "APP STARTUP"})

    warmup_proxy()

    start_time = time.time()
    scores = []

    for task_id in range(3):
        score = run_episode(task_id)
        print(f"Task {task_id} score: {score:.4f}")
        scores.append(score)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    for i, score in enumerate(scores):
        print(f"  Task {i} ({TASK_NAMES[i]}): {score:.4f}")
    if scores:
        print(f"\n  Average score: {sum(scores) / len(scores):.4f}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")