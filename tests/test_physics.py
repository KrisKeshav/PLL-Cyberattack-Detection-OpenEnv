"""Physics regression tests: transforms, PLL dynamics, grid events and attacks."""

import math

import numpy as np
import pytest

from pll_cyber.attacks import Attack, AttackSpec
from pll_cyber.environment import PLLCyberEnvironment
from pll_cyber.evaluation import run_episode
from pll_cyber.grid import GridConfig, GridEvent, GridModel
from pll_cyber.models import PLLAction
from pll_cyber.pll import PLLGains, SRFPLL, clarke, park, wrap_angle
from pll_cyber.scenarios import DT


def balanced(theta, v=1.0):
    return [v * math.cos(theta - k * 2 * math.pi / 3) for k in range(3)]


@pytest.mark.parametrize("theta", np.linspace(-3, 3, 13))
def test_clarke_park_identities(theta):
    a, b = clarke(*balanced(theta))
    assert a == pytest.approx(math.cos(theta), abs=1e-12)
    assert b == pytest.approx(math.sin(theta), abs=1e-12)
    vd, vq = park(a, b, theta)                       # locked
    assert vd == pytest.approx(1.0, abs=1e-12) and vq == pytest.approx(0.0, abs=1e-12)
    vd, vq = park(a, b, theta - 0.1)                 # θ̂ lags by 0.1 rad → vq = sin(0.1)
    assert vq == pytest.approx(math.sin(0.1), abs=1e-12)


def test_gain_design():
    g = PLLGains(omega_n=100.0, zeta=0.7071)
    assert g.ki == pytest.approx(1e4)
    assert g.kp == pytest.approx(141.42, rel=1e-3)


def _simulate(events=(), n_windows=20, theta0_err=0.0):
    grid = GridModel(GridConfig(h5_pu=0.0, h7_pu=0.0, f_drift_amp_hz=0.0, events=list(events)))
    pll = SRFPLL(DT, theta0=theta0_err)
    errs, freqs = [], []
    for _ in range(n_windows):
        out = grid.generate(250, DT)
        r = pll.run(*out["v"])
        errs += [math.degrees(wrap_angle(h - t)) for h, t in zip(r.theta_hat, out["theta"])]
        freqs += r.f_hat
    return np.array(errs), np.array(freqs)


def test_pll_locks_from_initial_error():
    errs, _ = _simulate(theta0_err=math.radians(30), n_windows=6)
    assert abs(errs[0]) == pytest.approx(30 - 3.6, abs=0.5)   # θ advanced one sample (ω·Ts = 3.6°)
    assert np.abs(errs[750:]).max() < 0.2           # locked within 150 ms


def test_pll_tracks_frequency_step_without_steady_state_error():
    # Type-2 loop: zero steady-state phase error for a frequency step
    ev = GridEvent("frequency_event", 0.1, {"rocof_hz_s": 100.0, "delta_f_hz": 0.5})
    errs, freqs = _simulate([ev], n_windows=20)
    assert np.abs(errs[-1000:]).max() < 0.05
    assert freqs[-1] == pytest.approx(50.5, abs=0.01)


def test_phase_jump_recovers():
    errs, _ = _simulate([GridEvent("phase_jump", 0.2, {"angle_deg": 25.0})], n_windows=12)
    jump = int(0.2 / DT)
    assert np.abs(errs[jump:jump + 20]).max() > 20     # the jump is seen…
    assert np.abs(errs[jump + 500:]).max() < 0.2        # …and tracked out within 100 ms


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_natural_events_never_desynchronise(task_id):
    env = PLLCyberEnvironment()
    for seed in range(40):
        obs = env.reset(task_id=task_id, seed=seed)
        if env._sc.attack.present:
            continue
        while not obs.done:
            obs = env.step(PLLAction(attack_detected=False))
        assert obs.episode_result.scenario["unmitigated_desync_step"] is None


def _vq_trace(task_id, seed, remove_attack=False):
    env = PLLCyberEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    sc = env._sc
    if remove_attack:                        # same grid, same noise streams, no attacker
        env._attack = Attack(AttackSpec())
    vq = []
    while not obs.done:
        obs = env.step(PLLAction(attack_detected=False))
        vq.append(obs.features.vq_rms_pu)
    return sc, obs.episode_result.scenario, np.array(vq)


def test_phase_spoofing_is_stealthy_but_destructive():
    checked = 0
    for seed in range(30):
        sc, res, vq_att = _vq_trace(2, seed)
        if not sc.attack.present:
            continue
        _, _, vq_clean = _vq_trace(2, seed, remove_attack=True)
        # destructive: the unprotected inverter always loses synchronism, 0.3–0.75 s after onset
        assert res["unmitigated_desync_step"] is not None
        assert 6 <= res["unmitigated_desync_step"] - sc.onset_step <= 15
        # stealthy: the attack leaves the PLL's own residual (vq) essentially unchanged
        assert (np.abs(vq_att - vq_clean) <= 0.002 + 0.15 * vq_clean).all()
        checked += 1
    assert checked >= 15


def test_pmu_disciplined_holdover_protects():
    class Oracle:
        def __init__(self, env):
            self.env = env

        def act(self, obs):
            sc = self.env._sc
            lab = sc.onset_step is not None and obs["step"] >= sc.onset_step
            return {"attack_detected": lab, "attack_type": 4 if lab else 0, "confidence": 1.0,
                    "protective_action": 2 if lab else 0}

    env = PLLCyberEnvironment()
    for seed in range(30):
        r = run_episode(Oracle(env), 2, seed, env)
        if r["scenario"]["attack"]["kind"] != "none":
            assert r["scenario"]["actual_desync_step"] is None, seed
            assert r["scenario"]["max_sync_error_deg"] < 10.0
