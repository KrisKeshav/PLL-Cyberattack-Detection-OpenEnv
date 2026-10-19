"""Baseline inference script: parsing, LLM loop with a fake client, log format, failure fallback."""

import re
import types

import pytest

import inference
from inference import LLMAgent, LocalEnv, parse_action, run_episode


def test_parse_action_variants():
    assert parse_action('{"attack_detected": true, "attack_type": 2, "confidence": 0.7, "protective_action": 1}') == \
        {"attack_detected": True, "attack_type": 2, "confidence": 0.7, "protective_action": 1}
    fenced = 'Sure!\n```json\n{"attack_detected": "true", "attack_type": 9, "confidence": 3}\n```'
    assert parse_action(fenced) == {"attack_detected": True, "attack_type": 4, "confidence": 1.0, "protective_action": 0}
    assert parse_action("no json here") is None
    assert parse_action('{"attack_type": "x"}') is None


class _FakeClient:
    """Mimics openai.OpenAI().chat.completions.create."""

    def __init__(self, reply=None, exc=None):
        self.calls = 0

        def create(**kw):
            self.calls += 1
            assert kw["messages"][0]["role"] == "system"
            if exc:
                raise exc
            msg = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))


@pytest.fixture(autouse=True)
def _reset_breaker():
    LLMAgent._disabled = False
    LLMAgent._consecutive_failures = 0
    yield
    LLMAgent._disabled = False


LOG = re.compile(r"^\[(START|STEP|END)\] ")


def test_llm_episode_and_log_format(capsys):
    fake = _FakeClient('{"attack_detected": false, "attack_type": 0, "confidence": 0.9, "protective_action": 0}')
    res = run_episode(0, 1, lambda sid: LocalEnv(), lambda: LLMAgent(fake), "fake-model")
    out = capsys.readouterr().out.strip().splitlines()
    assert fake.calls == 40 and res["success"] and res["stats"]["llm"] == 40
    assert out[0].startswith("[START] task=sinusoidal_fdi_detection env=pll-cyberattack-detection model=fake-model")
    assert all(LOG.match(line) for line in out)
    assert sum(line.startswith("[STEP]") for line in out) == 40
    end = out[-1]
    assert re.match(r"^\[END\] success=true steps=40 score=\d\.\d{3} rewards=(-?\d+\.\d\d,){39}-?\d+\.\d\d$", end)


def test_auth_failure_falls_back_to_heuristic(capsys):
    class E(Exception):
        status_code = 402
    fake = _FakeClient(exc=E("no credits"))
    res = run_episode(2, 1, lambda sid: LocalEnv(), lambda: LLMAgent(fake), "fake-model")
    assert res["success"] and fake.calls == 1          # disabled after the first permanent error
    assert res["stats"]["fallback"] == 40 and res["score"] > 0.01
    assert "llm_disabled_fallback" in capsys.readouterr().out


def test_unparseable_output_does_not_crash(capsys):
    fake = _FakeClient("I think there might be an attack")
    res = run_episode(1, 3, lambda sid: LocalEnv(), lambda: LLMAgent(fake), "fake-model")
    assert res["success"] and res["stats"]["fallback"] == 40 and not LLMAgent._disabled
    assert "unparseable_llm_output" in capsys.readouterr().out


def test_eval_seeds_cover_scenarios():
    from pll_cyber.scenarios import sample_scenario
    kinds = {t: {sample_scenario(t, s).attack.kind for s in inference.EVAL_SEEDS[t][:3]} for t in range(3)}
    assert kinds[0] == {"sinusoidal", "none"}
    assert kinds[1] == {"sinusoidal", "pulse", "ramp"}
    assert kinds[2] == {"phase_spoofing", "none"}
