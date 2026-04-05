"""
FastAPI application for the PLL Cyberattack Detection OpenEnv.

Exposes HTTP endpoints for environment interaction:
  POST /reset   — Reset environment with task_id
  POST /step    — Submit an action and advance one step
  GET  /state   — Get current internal state
  GET  /health  — Health check (returns 200)
"""

import asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.models import Observation, Action, Reward, State
from src.env import PLLAttackEnv

app = FastAPI(
    title="PLL Cyberattack Detection OpenEnv",
    description="OpenEnv for AI-driven cyberattack detection on SRF-PLLs",
    version="1.0.0",
)

env = PLLAttackEnv()
env_lock = asyncio.Lock()


class ResetRequest(BaseModel):
    task_id: int = 0
    seed: Optional[int] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


@app.post("/reset", response_model=Observation)
async def reset(req: Request):
    """Reset the environment and return initial observation."""
    async with env_lock:
        try:
            body = await req.body()
            if body:
                data = await req.json()
                request = ResetRequest(**data)
            else:
                request = ResetRequest()
        except Exception:
            request = ResetRequest()
        return env.reset(task_id=request.task_id, seed=request.seed)


@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    async with env_lock:
        if env.attack_generator is None:
            raise HTTPException(status_code=400, detail="Call /reset before /step")
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=State)
async def get_state():
    async with env_lock:
        return env.get_state()


@app.get("/health")
async def health():
    return {"status": "ok"}
