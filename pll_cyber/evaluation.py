"""In-process rollout helpers (used by tests, the benchmark script and inference.py)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from .environment import PLLCyberEnvironment
from .models import PLLAction


def obs_to_dict(obs) -> Dict[str, Any]:
    return obs.model_dump(exclude={"metadata"})


def run_episode(agent, task_id: int, seed: int, env: PLLCyberEnvironment | None = None,
                on_step: Callable[[int, Dict[str, Any], Any], None] | None = None) -> Dict[str, Any]:
    env = env or PLLCyberEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    if hasattr(agent, "reset"):
        agent.reset()
    rewards: List[float] = []
    while not obs.done:
        action = agent.act(obs_to_dict(obs))
        obs = env.step(PLLAction(**action))
        rewards.append(float(obs.reward or 0.0))
        if on_step:
            on_step(len(rewards), action, obs)
    res = obs.episode_result
    return {"task_id": task_id, "seed": seed, "score": res.grader_score,
            "total_reward": res.total_reward, "rewards": rewards,
            "breakdown": res.score_breakdown, "scenario": res.scenario}


def evaluate(make_agent: Callable[[], Any], task_ids: Iterable[int], seeds: Iterable[int]) -> Dict[int, Dict[str, float]]:
    env = PLLCyberEnvironment()
    out: Dict[int, Dict[str, float]] = {}
    seeds = list(seeds)
    for tid in task_ids:
        scores, rets = [], []
        for s in seeds:
            r = run_episode(make_agent(), tid, s, env)
            scores.append(r["score"])
            rets.append(r["total_reward"])
        out[tid] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                    "return": float(np.mean(rets)), "n": len(seeds)}
    return out
