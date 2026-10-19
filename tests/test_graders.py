"""Graders: range, determinism, non-constancy and resistance to degenerate strategies."""

import itertools

import numpy as np
import pytest

from pll_cyber.baselines import ConstantAgent, PhysicsHeuristicAgent, RandomAgent
from pll_cyber.environment import PLLCyberEnvironment
from pll_cyber.evaluation import evaluate, run_episode
from pll_cyber.graders import grade_episode

SEEDS = range(40)


class Oracle:
    """Cheats by reading ground truth — upper bound used only to prove tasks are solvable."""

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        sc = self.env._sc
        lab = sc.onset_step is not None and obs["step"] >= sc.onset_step
        return {"attack_detected": lab, "attack_type": sc.attack.type_id if lab else 0, "confidence": 1.0,
                "protective_action": 2 if (lab and sc.task_id == 2) else 0}


def _scores(make_agent, task_id, seeds=SEEDS):
    env = PLLCyberEnvironment()
    return np.array([run_episode(make_agent(env), task_id, s, env)["score"] for s in seeds])


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_oracle_solves_every_task(task_id):
    s = _scores(lambda env: Oracle(env), task_id)
    assert s.mean() > 0.95 and s.min() >= 0.9


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_degenerate_policies_score_low(task_id):
    always = _scores(lambda env: ConstantAgent(True, [1, 1, 4][task_id]), task_id)
    never = _scores(lambda env: ConstantAgent(False), task_id)
    seeds = itertools.count(100)
    rand = _scores(lambda env: RandomAgent(seed=next(seeds)), task_id)
    assert always.max() <= 0.05            # "always alarm" earns nothing anywhere
    assert never.mean() < 0.30             # only the attack-free share
    assert rand.mean() < 0.30


def test_always_mitigate_is_punished_on_hard_task():
    hold = _scores(lambda env: ConstantAgent(True, 4, protective_action=2), 2)
    trip = _scores(lambda env: ConstantAgent(True, 4, protective_action=3), 2)
    assert hold.mean() < 0.05 and trip.mean() < 0.05


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_scores_in_range_and_not_constant(task_id):
    s = _scores(lambda env: PhysicsHeuristicAgent(), task_id)
    assert ((s >= 0.0) & (s <= 1.0)).all()
    seeds = itertools.count(500)
    r = _scores(lambda env: RandomAgent(seed=next(seeds)), task_id)
    assert np.std(np.concatenate([s, r])) > 0.05


def test_heuristic_baseline_is_meaningful_but_not_saturated_on_hard():
    res = evaluate(PhysicsHeuristicAgent, [0, 1, 2], SEEDS)
    assert res[0]["mean"] > 0.9 and res[1]["mean"] > 0.8
    assert 0.5 < res[2]["mean"] < 0.9      # headroom left for better agents


def test_grader_is_deterministic_function_of_log():
    log = [{"step": t, "detected": t >= 12, "attack_type": 2, "protective_action": 0} for t in range(40)]
    a = grade_episode(1, log, 10, 50.0, true_type=2)
    b = grade_episode(1, log, 10, 50.0, true_type=2)
    assert a == b
    assert a["detection_delay_ms"] == 100.0 and a["false_alarm_rate"] == 0.0


def test_hard_grader_rewards_timely_mitigation():
    def log_for(mitigate_at):
        return [{"step": t, "detected": t >= 10, "attack_type": 4,
                 "protective_action": 2 if t >= mitigate_at else 0} for t in range(40)]
    early = grade_episode(2, log_for(11), 10, 50.0, unmitigated_desync_step=18, actual_desync_step=None)
    late = grade_episode(2, log_for(25), 10, 50.0, unmitigated_desync_step=18, actual_desync_step=18)
    assert early["protected"] and not late["protected"]
    assert early["score"] > late["score"] + 0.4
