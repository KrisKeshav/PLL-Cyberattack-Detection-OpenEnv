"""Environment contract: reset/step/state, determinism, hidden ground truth, reward bounds."""

import pytest

from pll_cyber.environment import PLLCyberEnvironment
from pll_cyber.models import PLLAction, PLLObservation
from pll_cyber.rewards import REWARD_MAX, REWARD_MIN
from pll_cyber.scenarios import MAX_STEPS

QUIET = PLLAction(attack_detected=False, confidence=0.9)


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_reset_returns_clean_initial_observation(task_id):
    env = PLLCyberEnvironment()
    obs = env.reset(task_id=task_id, seed=3)
    assert isinstance(obs, PLLObservation)
    assert obs.step == 0 and not obs.done and obs.reward is None
    assert obs.task_id == task_id and obs.task_brief
    assert len(obs.waveforms.va) == 25 and len(obs.trend.f_local_hz) >= 1
    assert env.state.step_count == 0 and env.state.scenario is None


def test_episode_length_and_termination():
    env = PLLCyberEnvironment()
    obs = env.reset(task_id=1, seed=0)
    n = 0
    while not obs.done:
        obs = env.step(QUIET)
        n += 1
    assert n == MAX_STEPS
    assert obs.episode_result is not None and 0.0 <= obs.episode_result.grader_score <= 1.0
    after = env.step(QUIET)                     # stepping a finished episode is harmless
    assert after.done and after.reward == 0.0


def test_reset_fully_clears_previous_episode():
    env = PLLCyberEnvironment()
    env.reset(task_id=2, seed=1)
    for _ in range(15):
        env.step(PLLAction(attack_detected=True, protective_action=3))
    obs = env.reset(task_id=2, seed=1)
    assert obs.inverter_mode == "normal" and obs.step == 0
    assert env.state.cumulative_reward == 0.0


def _trajectory(task_id, seed, policy):
    env = PLLCyberEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    traj = [obs.model_dump_json()]
    while not obs.done:
        obs = env.step(policy(obs))
        traj.append(obs.model_dump_json())
    return traj


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_same_seed_is_bit_identical(task_id):
    pol = lambda o: PLLAction(attack_detected=o.step > 12, attack_type=1, protective_action=2 if o.step > 20 else 0)
    assert _trajectory(task_id, 42, pol) == _trajectory(task_id, 42, pol)


def test_different_seeds_differ():
    pol = lambda o: QUIET
    assert _trajectory(0, 1, pol)[1] != _trajectory(0, 2, pol)[1]


def test_unseeded_reset_reports_its_seed():
    env = PLLCyberEnvironment()
    env.reset(task_id=0)
    assert isinstance(env.state.seed, int)


def test_ground_truth_hidden_until_done():
    env = PLLCyberEnvironment()
    obs = env.reset(task_id=1, seed=5)
    dumped = obs.model_dump_json()
    for leak in ("onset", "attack_params", "theta_true", "true_attack"):
        assert leak not in dumped
    while not obs.done:
        assert env.state.scenario is None and env.state.grader_score is None
        obs = env.step(QUIET)
    assert env.state.scenario is not None and env.state.grader_score is not None


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_rewards_within_declared_range(task_id):
    env = PLLCyberEnvironment()
    import random
    rng = random.Random(task_id)
    for seed in range(8):
        obs = env.reset(task_id=task_id, seed=seed)
        while not obs.done:
            a = PLLAction(attack_detected=rng.random() < 0.5, attack_type=rng.randint(0, 4),
                          confidence=rng.random(), protective_action=rng.choice([0, 1, 2, 3]))
            obs = env.step(a)
            assert REWARD_MIN - 1e-9 <= obs.reward <= REWARD_MAX + 1e-9
            b = obs.reward_breakdown
            parts = (b.detection + b.early_detection_bonus + b.calibration + b.classification
                     + b.mitigation_cost + b.harm_penalty)
            assert parts == pytest.approx(b.total)


def test_task_selection_by_name_and_validation():
    env = PLLCyberEnvironment()
    assert env.reset(task="hard", seed=0).task_id == 2
    assert env.reset(task="attack_classification_under_disturbances", seed=0).task_id == 1
    with pytest.raises(ValueError):
        env.reset(task_id=7)
    with pytest.raises(Exception):
        PLLAction(attack_detected=True, attack_type=9)


def test_disconnect_is_latched_and_holdover_reversible():
    env = PLLCyberEnvironment()
    env.reset(task_id=2, seed=2)
    assert env.step(PLLAction(attack_detected=True, protective_action=2)).inverter_mode == "holdover"
    assert env.step(PLLAction(attack_detected=False)).inverter_mode == "normal"
    assert env.step(PLLAction(attack_detected=True, protective_action=3)).inverter_mode == "disconnected"
    assert env.step(PLLAction(attack_detected=False)).inverter_mode == "disconnected"


def test_openenv_yaml_matches_code():
    from pathlib import Path

    import yaml

    from pll_cyber.scenarios import TASKS

    spec = yaml.safe_load((Path(__file__).parents[1] / "openenv.yaml").read_text(encoding="utf-8"))
    assert spec["app"] == "server.app:app" and spec["port"] == 7860
    assert spec["reward_range"] == [round(REWARD_MIN, 2), round(REWARD_MAX, 2)]
    assert spec["episode"]["steps"] == MAX_STEPS
    assert [(t["id"], t["name"]) for t in spec["tasks"]] == [(i, TASKS[i].name) for i in range(3)]
