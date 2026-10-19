"""
Typed OpenEnv client (persistent WebSocket session).

    from pll_cyber.client import PLLCyberEnv
    from pll_cyber.models import PLLAction

    with PLLCyberEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id=2, seed=7)
        while not result.done:
            result = env.step(PLLAction(attack_detected=False))
        print(result.observation.episode_result.grader_score)
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import PLLAction, PLLObservation, PLLState


class PLLCyberEnv(EnvClient[PLLAction, PLLObservation, PLLState]):
    def _step_payload(self, action: PLLAction) -> Dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PLLObservation]:
        obs = PLLObservation(**payload.get("observation", {}),
                             done=payload.get("done", False), reward=payload.get("reward"))
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict[str, Any]) -> PLLState:
        return PLLState(**payload)
