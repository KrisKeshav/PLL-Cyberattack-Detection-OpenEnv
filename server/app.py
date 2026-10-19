"""
FastAPI server for the PLL Cyber-Guard OpenEnv environment.

Built on openenv-core's `create_app`, which provides the standard surface:
    WS   /ws         persistent sessions (one environment per connection) — used by EnvClient
    GET  /schema     action / observation / state JSON schemas
    GET  /metadata   environment metadata
    GET  /health     {"status": "healthy"}
    POST /mcp        MCP JSON-RPC endpoint

openenv's HTTP /reset and /step are stateless (a fresh environment per request),
which cannot run a multi-step episode. They are replaced here by session-aware
versions with the same request/response shapes:
    POST /reset      body: {"task_id": 0-2, "seed": int, "session_id": str?}
    POST /step       body: {"action": {...}, "session_id": str?}  (a bare action dict is also accepted)
    GET  /state      ?session_id=...
Requests without a session_id share the "default" session (handy for curl).

Extras:
    GET  /tasks      task catalogue
    GET  /demo       run the physics-heuristic baseline on one episode and return the trace
    GET  /           landing page with a live demo plot
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pll_cyber import __version__
from pll_cyber.baselines import PhysicsHeuristicAgent
from pll_cyber.environment import PLLCyberEnvironment
from pll_cyber.evaluation import obs_to_dict, run_episode
from pll_cyber.graders import GRADERS
from pll_cyber.models import PLLAction, PLLObservation, PLLState
from pll_cyber.rewards import REWARD_MAX, REWARD_MIN
from pll_cyber.scenarios import MAX_STEPS, TASKS

MAX_HTTP_SESSIONS = int(os.environ.get("MAX_HTTP_SESSIONS", "256"))
SESSION_TTL_S = float(os.environ.get("SESSION_TTL_S", "1800"))
MAX_WS_SESSIONS = int(os.environ.get("MAX_WS_SESSIONS", "64"))

app = create_app(
    PLLCyberEnvironment,
    PLLAction,
    PLLObservation,
    env_name="pll-cyberattack-detection",
    max_concurrent_envs=MAX_WS_SESSIONS,
)
app.title = "PLL Cyber-Guard — OpenEnv"
app.description = (
    "Detect, classify and mitigate false-data-injection attacks on the SRF-PLL of a grid-connected "
    "inverter, without alarming on natural grid disturbances. See /tasks and /docs."
)

# Drop openenv's stateless HTTP simulation routes; session-aware versions follow.
app.router.routes = [
    r for r in app.router.routes
    if not (isinstance(r, APIRoute) and r.path in {"/reset", "/step", "/state"})
]


# ---------------------------------------------------------------------------
# HTTP session store
# ---------------------------------------------------------------------------
class _Session:
    def __init__(self):
        self.env = PLLCyberEnvironment()
        self.lock = asyncio.Lock()
        self.touched = time.monotonic()
        self.started = False


_sessions: "OrderedDict[str, _Session]" = OrderedDict()
_sessions_lock = asyncio.Lock()


async def _get_session(session_id: str, create: bool) -> _Session:
    async with _sessions_lock:
        now = time.monotonic()
        for sid in [k for k, s in _sessions.items() if now - s.touched > SESSION_TTL_S]:
            _sessions.pop(sid, None)
        sess = _sessions.get(session_id)
        if sess is None:
            if not create:
                raise HTTPException(404, f"Unknown session_id '{session_id}'. Call POST /reset first.")
            while len(_sessions) >= MAX_HTTP_SESSIONS:
                _sessions.popitem(last=False)          # evict least-recently used
            sess = _Session()
            _sessions[session_id] = sess
        _sessions.move_to_end(session_id)
        sess.touched = now
        return sess


class ResetBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: Optional[int] = Field(None, ge=0, le=2, description="0=easy, 1=medium, 2=hard")
    task: Optional[str] = Field(None, description="Alternatively the task name or difficulty")
    seed: Optional[int] = Field(None, ge=0, description="Seed for a reproducible episode")
    episode_id: Optional[str] = None
    session_id: Optional[str] = Field(None, description="Isolates concurrent clients; default 'default'")


class EnvResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False
    session_id: str


def _response(obs: PLLObservation, session_id: str) -> EnvResponse:
    return EnvResponse(observation=obs_to_dict(obs), reward=obs.reward, done=obs.done, session_id=session_id)


@app.post("/reset", response_model=EnvResponse, tags=["Environment Control"],
          summary="Start a new episode (session-aware)")
async def reset(body: Optional[ResetBody] = Body(default=None)) -> EnvResponse:
    body = body or ResetBody()
    sid = body.session_id or "default"
    sess = await _get_session(sid, create=True)
    async with sess.lock:
        try:
            obs = await run_in_threadpool(sess.env.reset, seed=body.seed, episode_id=body.episode_id,
                                          task_id=body.task_id, task=body.task)
        except ValueError as e:
            raise HTTPException(422, str(e))
        sess.started = True
        return _response(obs, sid)


@app.post("/step", response_model=EnvResponse, tags=["Environment Control"],
          summary="Apply an action to the session's episode")
async def step(body: Dict[str, Any] = Body(..., examples=[{
    "action": {"attack_detected": False, "attack_type": 0, "confidence": 0.9, "protective_action": 0},
    "session_id": "default"}])) -> EnvResponse:
    sid = body.get("session_id") or "default"
    raw = body.get("action", {k: v for k, v in body.items() if k not in ("session_id", "timeout_s", "request_id")})
    try:
        action = PLLAction.model_validate(raw)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False))
    sess = await _get_session(sid, create=False)
    async with sess.lock:
        if not sess.started:
            raise HTTPException(400, "Call POST /reset before /step.")
        obs = await run_in_threadpool(sess.env.step, action)
        return _response(obs, sid)


@app.get("/state", response_model=PLLState, tags=["State Management"],
         summary="Episode state (ground truth is revealed only after done)")
async def state(session_id: str = Query("default")) -> PLLState:
    sess = await _get_session(session_id, create=True)
    return sess.env.state


@app.get("/tasks", tags=["Environment Info"], summary="Task catalogue")
async def tasks() -> Dict[str, Any]:
    return {
        "episode_steps": MAX_STEPS,
        "decision_interval_ms": 50,
        "reward_range": [round(REWARD_MIN, 4), round(REWARD_MAX, 4)],
        "tasks": [
            {"id": t.task_id, "name": t.name, "difficulty": t.difficulty,
             "attack_probability": t.attack_probability, "grader": GRADERS[t.task_id].__name__,
             "score_range": [0.0, 1.0], "description": t.brief}
            for t in TASKS.values()
        ],
        "action_schema": PLLAction.model_json_schema(),
    }


@app.get("/demo", tags=["Environment Info"], summary="Run the heuristic baseline on one episode")
async def demo(task_id: int = Query(2, ge=0, le=2), seed: int = Query(7, ge=0)) -> Dict[str, Any]:
    trace = []

    def record(i, action, obs):
        f = obs.features
        trace.append({"step": i, "time_s": obs.time_s, "f_local_hz": f.f_local_hz, "pmu_f_hz": obs.pmu.f_hz,
                      "vq_rms_pu": f.vq_rms_pu, "v_zero_seq_rms_pu": f.v_zero_seq_rms_pu,
                      "mode": obs.inverter_mode, "reward": obs.reward, "action": action})

    res = await run_in_threadpool(run_episode, PhysicsHeuristicAgent(), task_id, seed, None, record)
    return {"task_id": task_id, "seed": seed, "score": res["score"], "total_reward": res["total_reward"],
            "breakdown": res["breakdown"], "scenario": res["scenario"], "trace": trace}


_LANDING = Path(__file__).with_name("landing.html")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing() -> HTMLResponse:
    html = _LANDING.read_text(encoding="utf-8") if _LANDING.exists() else "<h1>PLL Cyber-Guard</h1>"
    return HTMLResponse(html.replace("{{VERSION}}", __version__))


def main(host: str = "0.0.0.0", port: int = int(os.environ.get("PORT", "7860"))) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    main(args.host, args.port)
