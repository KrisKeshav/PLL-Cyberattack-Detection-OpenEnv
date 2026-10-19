"""
Baseline inference — PLL Cyber-Guard OpenEnv
=============================================

Runs an LLM agent (OpenAI-compatible client) on all three tasks with fixed
seeds and prints OpenEnv-style logs plus a final score table.

Environment variables
  API_BASE_URL        OpenAI-compatible endpoint      (default: https://router.huggingface.co/v1)
  MODEL_NAME          model identifier                (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN / API_KEY  API key for the endpoint
  ENV_URL             environment server URL; if unset or unreachable the environment runs in-process
  AGENT               "llm" (default when a key is set) or "heuristic"
  EPISODES_PER_TASK   default 3
  MAX_WORKERS         concurrent episodes, default 3
  TIME_BUDGET_S       wall-clock budget, default 1050 (17.5 min) — steps beyond it fall back to the heuristic

STDOUT (one block per episode, never interleaved)
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from pll_cyber.baselines import PhysicsHeuristicAgent, _v0_decompose

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
ENV_URL = os.environ.get("ENV_URL", "").rstrip("/")
AGENT = os.environ.get("AGENT", "llm" if API_KEY else "heuristic").lower()
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "3"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "1050"))
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "25"))

BENCHMARK = "pll-cyberattack-detection"
TASK_NAMES = {0: "sinusoidal_fdi_detection", 1: "attack_classification_under_disturbances",
              2: "stealthy_phase_spoofing_mitigation"}
# Fixed evaluation seeds → reproducible scenarios. Chosen so the first three per task already cover
# the attack types and an attack-free episode (see scripts/evaluate_baselines.py for 100-seed results).
EVAL_SEEDS = {
    0: [1, 8, 2, 3, 9, 4],      # sinusoidal, none, sinusoidal, ...
    1: [1, 3, 4, 7, 2, 5],      # sinusoidal, pulse, ramp, none, ...
    2: [1, 4, 2, 3, 5, 6],      # spoofing, none, spoofing, ...
}

_t0 = time.monotonic()
_print_lock = threading.Lock()


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Environment adapters (HTTP session or in-process)
# ---------------------------------------------------------------------------
class HTTPEnv:
    def __init__(self, base_url: str, session_id: str):
        import requests
        self.s = requests.Session()
        self.url = base_url
        self.sid = session_id

    def reset(self, task_id: int, seed: int) -> Dict[str, Any]:
        r = self.s.post(f"{self.url}/reset", json={"task_id": task_id, "seed": seed, "session_id": self.sid}, timeout=60)
        r.raise_for_status()
        return r.json()["observation"]

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
        r = self.s.post(f"{self.url}/step", json={"action": action, "session_id": self.sid}, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["observation"], float(j["reward"] or 0.0), bool(j["done"])


class LocalEnv:
    def __init__(self):
        from pll_cyber.environment import PLLCyberEnvironment
        self.env = PLLCyberEnvironment()

    def reset(self, task_id: int, seed: int) -> Dict[str, Any]:
        from pll_cyber.evaluation import obs_to_dict
        return obs_to_dict(self.env.reset(task_id=task_id, seed=seed))

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool]:
        from pll_cyber.evaluation import obs_to_dict
        from pll_cyber.models import PLLAction
        o = self.env.step(PLLAction(**action))
        return obs_to_dict(o), float(o.reward or 0.0), bool(o.done)


def _server_up(url: str) -> bool:
    if not url:
        return False
    try:
        import requests
        return requests.get(f"{url}/health", timeout=5).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the cyber-security guard of a grid-connected solar inverter. Every 50 ms you receive
measurements from its Synchronous-Reference-Frame Phase-Locked Loop (SRF-PLL) and must decide whether a
False-Data-Injection (FDI) attack on the voltage sensors is in progress.

PHYSICS CHEAT-SHEET
- Healthy grid: vd ≈ 1 pu, vq ≈ 0 (rms < 0.012), frequency ≈ 50 Hz (natural wander ±0.03 Hz), small 5th/7th harmonics.
- Natural (NOT attacks): phase jumps (short vq spike), balanced sags (all phases dip), single-phase sags
  (one phase dips; zero-sequence appears AT 50 Hz), harmonic steps, frequency events (RoCoF up to 0.6 Hz/s that the
  trusted PMU also sees). These last ≤ 0.25 s or are mirrored by the PMU.
- Attack 1 sinusoidal: interharmonic injection on one phase → zero-sequence content NOT at 50 Hz
  (diagnostics.v0_offfund_rms high), persistent oscillation of local frequency.
- Attack 2 ramp: growing DC offset on one phase (phase_dc_offset magnitude keeps increasing).
- Attack 3 pulse: constant DC offset on one phase for 0.1-0.3 s, then gone. Real grids never have DC offsets.
- Attack 4 phase spoofing: all three phases rotated together; vq stays clean, but the LOCAL frequency drifts away
  from the PMU frequency. The PMU is trustworthy but delayed (see pmu.age_ms; it reports the average of the
  100 ms ending age_ms ago), so compare the PMU with your local frequency from the same moment in trend.f_local_hz.
  diagnostics.spoof_angle_est_deg integrates that mismatch: |angle| growing past ~3 deg is an attack.
  The inverter desynchronises (damage) at 10 deg, typically 0.35-0.7 s after the attack starts.

TASKS
- task 0: detect a sinusoidal FDI (attack_type 1). task 1: detect AND classify (1/2/3) amid natural events.
- task 2: detect phase spoofing (attack_type 4) and engage protective_action=2 (PMU-aided holdover) in time.
Once an attack has started the system stays compromised: keep attack_detected=true for the rest of the episode.
False alarms are heavily penalised; protective_action 2/3 on a healthy grid is costly; 3 (disconnect) is latched.

Reply with ONLY a JSON object:
{"attack_detected": bool, "attack_type": 0-4, "confidence": 0.0-1.0, "protective_action": 0-3, "reason": "<10 words"}"""


def _diagnostics(obs: Dict[str, Any], tracker: PhysicsHeuristicAgent) -> Dict[str, float]:
    """DSP helpers an engineer would compute on the raw observation (no ground truth involved)."""
    d = _v0_decompose(obs)
    tracker.act(obs)            # updates the latency-compensated spoofing angle estimate
    return {"v0_dc_pu": round(d["dc"], 5), "v0_fundamental_pu": round(d["amp50"], 5),
            "v0_offfund_rms_pu": round(d["resid_rms"], 5),
            "spoof_angle_est_deg": round(tracker.angle_est_deg, 2)}


def _user_message(obs: Dict[str, Any], diag: Dict[str, float], memory: List[str]) -> str:
    payload = {
        "task_id": obs["task_id"], "step": obs["step"], "time_s": obs["time_s"],
        "inverter_mode": obs["inverter_mode"],
        "features": obs["features"], "pmu": obs["pmu"], "trend": obs["trend"],
        "diagnostics": diag, "your_recent_decisions": memory[-6:],
    }
    return json.dumps(payload, separators=(",", ":"))


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        p = json.loads(m.group(0))
        det = p.get("attack_detected", False)
        det = det if isinstance(det, bool) else str(det).lower() == "true"
        return {
            "attack_detected": det,
            "attack_type": max(0, min(4, int(p.get("attack_type", 0)))),
            "confidence": max(0.0, min(1.0, float(p.get("confidence", 0.5)))),
            "protective_action": max(0, min(3, int(p.get("protective_action", 0)))),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


class LLMAgent:
    _consecutive_failures = 0
    _disabled = False
    _lock = threading.Lock()

    def __init__(self, client):
        self.client = client
        self.tracker = PhysicsHeuristicAgent()
        self.fallback = PhysicsHeuristicAgent()
        self.memory: List[str] = []
        self.stats = {"llm": 0, "fallback": 0}

    def act(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        diag = _diagnostics(obs, self.tracker)
        fb = self.fallback.act(obs)
        action, error = None, None
        if not LLMAgent._disabled and time.monotonic() - _t0 < TIME_BUDGET_S:
            for attempt in range(2):
                try:
                    resp = self.client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.0, max_tokens=120, seed=0, timeout=LLM_TIMEOUT_S,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": _user_message(obs, diag, self.memory)}],
                    )
                    action = parse_action(resp.choices[0].message.content)
                    if action is None:
                        error = "unparseable_llm_output"
                    with LLMAgent._lock:
                        LLMAgent._consecutive_failures = 0
                    break
                except Exception as e:  # network / auth / rate limit
                    error = f"{type(e).__name__}"
                    if getattr(e, "status_code", None) in (401, 402, 403):   # permanent: bad key / no credits
                        with LLMAgent._lock:
                            if not LLMAgent._disabled:
                                LLMAgent._disabled = True
                                _err(f"[WARN] LLM unavailable (HTTP {e.status_code}); using heuristic fallback")
                        break
                    time.sleep(1.5 * (attempt + 1))
            if action is None and error and error != "unparseable_llm_output":
                with LLMAgent._lock:
                    LLMAgent._consecutive_failures += 1
                    if LLMAgent._consecutive_failures >= 5 and not LLMAgent._disabled:
                        LLMAgent._disabled = True
                        _err(f"[WARN] LLM disabled after repeated failures ({error}); using heuristic fallback")
        elif LLMAgent._disabled:
            error = "llm_disabled_fallback"
        else:
            error = "time_budget_fallback"
        if action is None:
            action = fb
            self.stats["fallback"] += 1
        else:
            self.stats["llm"] += 1
        self.memory.append(f"t={obs['time_s']:.2f}:det={int(action['attack_detected'])},type={action['attack_type']},"
                           f"prot={action['protective_action']}")
        return action, error


class HeuristicAgentAdapter:
    def __init__(self):
        self.agent = PhysicsHeuristicAgent()
        self.stats = {"llm": 0, "fallback": 0}

    def act(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.agent.act(obs), None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(task_id: int, seed: int, make_env, make_agent, label: str) -> Dict[str, Any]:
    lines = [f"[START] task={TASK_NAMES[task_id]} env={BENCHMARK} model={label}"]
    rewards: List[float] = []
    score, success, steps = 0.0, False, 0
    agent = make_agent()
    try:
        env = make_env(f"t{task_id}-s{seed}")
        obs = env.reset(task_id, seed)
        done = False
        while not done:
            action, error = agent.act(obs)
            obs, reward, done = env.step(action)
            steps += 1
            rewards.append(reward)
            lines.append(f"[STEP] step={steps} action={json.dumps(action, separators=(',', ':'))} "
                         f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}")
        res = obs.get("episode_result") or {}
        score = float(res.get("grader_score", 0.0))
        success = True
    except Exception as e:
        _err(f"[DEBUG] episode task={task_id} seed={seed} failed: {type(e).__name__}: {e}")
    lines.append(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
                 f"rewards={','.join(f'{r:.2f}' for r in rewards)}")
    with _print_lock:
        print("\n".join(lines), flush=True)
    return {"task_id": task_id, "seed": seed, "score": score, "success": success, "stats": agent.stats}


def main() -> None:
    use_http = _server_up(ENV_URL)
    if ENV_URL and not use_http:
        _err(f"[WARN] ENV_URL={ENV_URL} unreachable — running the environment in-process")
    make_env = (lambda sid: HTTPEnv(ENV_URL, sid)) if use_http else (lambda sid: LocalEnv())

    if AGENT == "llm" and API_KEY:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        make_agent = lambda: LLMAgent(client)  # noqa: E731
        label = MODEL_NAME
    else:
        if AGENT == "llm":
            _err("[WARN] No HF_TOKEN/API_KEY set — running the physics-heuristic baseline instead of the LLM")
        make_agent = HeuristicAgentAdapter
        label = "physics-heuristic"

    jobs = [(t, s) for t in range(3) for s in EVAL_SEEDS[t][:EPISODES_PER_TASK]]
    _err(f"[DEBUG] agent={label} env={'http ' + ENV_URL if use_http else 'in-process'} episodes={len(jobs)}")
    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as pool:
        results = list(pool.map(lambda j: run_episode(j[0], j[1], make_env, make_agent, label), jobs))

    print("\n=== Baseline scores ===", flush=True)
    overall = []
    for t in range(3):
        sc = [r["score"] for r in results if r["task_id"] == t]
        mean = sum(sc) / len(sc)
        overall.append(mean)
        print(f"task {t} ({TASK_NAMES[t]}): mean={mean:.3f} scores={[round(x, 3) for x in sc]}", flush=True)
    llm_steps = sum(r["stats"]["llm"] for r in results)
    fb_steps = sum(r["stats"]["fallback"] for r in results)
    print(f"overall mean={sum(overall) / 3:.3f}  agent={label}  llm_steps={llm_steps} fallback_steps={fb_steps} "
          f"elapsed={time.monotonic() - _t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
