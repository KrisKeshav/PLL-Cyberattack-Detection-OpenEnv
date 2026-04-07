import sys
sys.path.insert(0, ".")
import pytest
import math
from src.pll_sim import SRFPLLSimulator, OMEGA0
from src.env import PLLAttackEnv
from src.models import Action
from src.attacks import AttackGenerator, sample_sinusoidal_params
import numpy as np

DUMMY_ACTION = Action(
    attack_detected=False,
    attack_type=0,
    confidence=0.5,
    protective_action=0
)


def test_episode_terminates_at_500():
    """Episode must terminate with done=True at step 500."""
    env = PLLAttackEnv()
    env.reset(task_id=0, seed=42)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(DUMMY_ACTION)
        steps += 1
    assert steps == 500, f"Episode ended at step {steps}, expected 500"

def test_all_tasks_reset():
    """All three tasks must reset without error."""
    env = PLLAttackEnv()
    for task_id in range(3):
        obs = env.reset(task_id=task_id, seed=42)
        assert obs.task_id == task_id
        assert obs.step == 0
        assert len(obs.vq_window) == 20

def test_oracle_agent_nonzero_reward():
    """An oracle agent should accumulate positive reward."""
    env = PLLAttackEnv()
    env.reset(task_id=0, seed=42)
    total_reward = 0.0
    done = False
    while not done:
        action = Action(
            attack_detected=env.attack_active,
            attack_type=env.true_attack_type if env.attack_active else 0,
            confidence=1.0,
            protective_action=0
        )
        _, reward, done, _ = env.step(action)
        total_reward += reward.total
    assert total_reward > 0, f"Oracle agent got non-positive reward: {total_reward}"

def test_reward_bounds():
    """Reward total must stay within [-2.5, 1.5] per step."""
    env = PLLAttackEnv()
    env.reset(task_id=2, seed=42)
    done = False
    while not done:
        _, reward, done, _ = env.step(DUMMY_ACTION)
        assert -2.5 <= reward.total <= 1.5, (
            f"Reward out of bounds: {reward.total}"
        )
