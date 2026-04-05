"""
FastAPI application for the PLL Cyberattack Detection OpenEnv.

Exposes HTTP endpoints for environment interaction:
  POST /reset   — Reset environment with task_id
  POST /step    — Submit an action and advance one step
  GET  /state   — Get current internal state
  GET  /health  — Health check (returns 200)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from src.models import Observation, Action, Reward, State
from src.env import PLLAttackEnv


app = FastAPI(
    title="PLL Cyberattack Detection OpenEnv",
    description="OpenEnv for AI-driven cyberattack detection on SRF-PLLs",
    version="1.0.0",
)

# Global environment instance
env = PLLAttackEnv()


class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    task_id: int = 0
    seed: int = None


class StepResponse(BaseModel):
    """Response body for /step endpoint."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest):
    """Reset the environment and return initial observation."""
    obs = env.reset(task_id=request.task_id, seed=request.seed)
    return obs


@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    """Submit an action and advance the environment one step."""
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=State)
async def get_state():
    """Return the current internal state."""
    return env.get_state()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
