---
title: PLL Cyber-Guard
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - power-systems
  - cybersecurity
---

# PLL Cyber-Guard — an OpenEnv environment for cyber-attacks on grid inverters

[![HF Space](https://img.shields.io/badge/🤗%20Space-live-blue)](https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-validated-green)](openenv.yaml)
[![CI](https://github.com/KrisKeshav/PLL-Cyberattack-Detection-OpenEnv/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

An agent plays the **cyber-security guard of a grid-connected solar/battery inverter**. Every 50 ms it reads
the signals of the inverter's phase-locked loop (PLL) and a trusted but delayed phasor-measurement (PMU)
stream. It must **detect, classify and mitigate false-data-injection (FDI) attacks** on the voltage sensors,
without raising alarms on the natural disturbances a real grid produces all the time.

## Why this matters

Inverter-based resources (solar, wind, batteries) now supply a large share of many grids. A grid-following inverter
locks to the grid through a **Synchronous-Reference-Frame PLL** (SRF-PLL). If an attacker can rewrite the voltage
samples the PLL sees (compromised sensor firmware, a man-in-the-middle on the sampled-values bus, or malware on the
controller), they can make the inverter inject current out of phase with the real grid. The result is power-quality
problems, protection trips, and, at scale, a stability risk. PLL-targeted FDI is an active research topic. What makes
it hard in practice is not detecting anomalies. It is **telling attacks apart from ordinary grid events**, and
**acting before damage while avoiding false trips**, because every false trip throws away generation.

This environment turns that operator problem into a benchmark. The physics is simulated faithfully: a 5 kHz sampled
SRF-PLL, a sensor chain, natural grid events and a PMU. The reward is dense and accounts for cost, and the graders are
deterministic.

## What the agent faces

```
 true grid (50 Hz ± wander, 5th/7th harmonics, natural events)
        │                                     │
        ▼                                     ▼
 [FDI attacker rewrites samples]       PMU at substation (GPS-timed, not spoofable)
        │                              10 reports/s · noisy · 40–100 ms latency
        ▼                                     │
 sensors: noise σ=5 mpu + 12-bit ADC          │
        ▼                                     │
 SRF-PLL (Clarke → Park → PI, ωn=100 rad/s,   │
          ζ=0.707, amplitude-normalised)      │
        │   ▲ holdover: loop opened,          │
        │   └ PMU-disciplined oscillator ◄────┤
        ▼                                     ▼
 features + waveforms ─────────►  AGENT  ◄── PMU frequency / magnitude
                                    │
          attack_detected · attack_type · confidence · protective_action
```

**Physics notes.** The Clarke/Park transforms are cosine-referenced and amplitude-invariant, so `vq = V·sin(θ − θ̂)`.
The PLL is type-2 and has zero steady-state error to frequency steps; the tests verify this. The simulator also tracks
the **true** grid angle, so it knows when the inverter really drifts out of synchronism. That is defined as
**|θ̂ − θ| > 10° sustained for 30 ms**, and the agent never sees it. A "shadow" PLL that never mitigates runs alongside
and gives the counterfactual time at which an unprotected inverter would have lost sync. The hard-task grader uses
that time as its deadline.

### Natural grid events (not attacks)
| Event | What it looks like to the PLL |
|---|---|
| Phase jump (5–25°) | vq spike, frequency transient; recovers in about 30 ms; the PMU frequency also spikes |
| Balanced sag (0.5–0.9 pu) | vd drops on all phases |
| Single-phase sag (0.3–0.8 pu) | zero/negative sequence **at 50 Hz**, 1.5 Hz frequency ripple, which looks like single-phase FDI |
| Harmonic step | 300 Hz ripple in vq grows |
| Frequency event (RoCoF 0.1–0.6 Hz/s) | local frequency drifts, and the PMU sees the same drift (with a delay) |

### Attacks
| id | Attack | Footprint |
|---|---|---|
| 1 | Sinusoidal interharmonic injection, one phase | zero-sequence content **off** 50 Hz, persistent frequency oscillation |
| 2 | Ramp bias, one phase | growing DC offset (real grids have no DC) |
| 3 | Pulse bias, one phase (100–300 ms) | constant DC offset, then gone |
| 4 | **Coordinated phase spoofing** | all three phases rotated by a slowly accelerating angle δ(t). The PLL tracks it perfectly, so vq is unchanged, and the local frequency drifts like a natural frequency event. The only evidence is a mismatch against the *delayed* PMU. Unprotected, the inverter loses sync 0.35–0.7 s after onset. |

## Tasks

| id | name | difficulty | what the agent must do |
|---|---|---|---|
| 0 | `sinusoidal_fdi_detection` | easy | Detect a sinusoidal FDI quickly. 20% of episodes are attack-free; occasional mild sags. |
| 1 | `attack_classification_under_disturbances` | medium | Detect **and classify** sinusoidal / ramp / pulse attacks while 1–3 natural events occur. At least one event always lands before the attack, so false-alarm resistance is always tested. |
| 2 | `stealthy_phase_spoofing_mitigation` | hard | Detect phase spoofing hidden in natural frequency events using the delayed PMU, and engage **holdover before the inverter desynchronises**. Holding over or tripping on a healthy grid is penalised. |

Each episode is 40 decisions × 50 ms (2 s after a 200 ms warm-up). A scenario is fully determined by
`(task_id, seed)`: attack type, onset, parameters, grid events, harmonics, PMU latency and every noise stream.

## Action space (`PLLAction`)
| field | type | meaning |
|---|---|---|
| `attack_detected` | bool | an attack is in progress |
| `attack_type` | int 0–4 | 0 none, 1 sinusoidal, 2 ramp, 3 pulse, 4 phase spoofing |
| `confidence` | float 0–1 | probability that `attack_detected` is right (scored with a Brier rule) |
| `protective_action` | int 0–3 | 0 none, 1 alert operator, **2 PMU-aided holdover** (open the PLL loop; free-run on a PMU-disciplined oscillator), **3 disconnect** (latched) |

## Observation space (`PLLObservation`)
| group | content |
|---|---|
| `waveforms` | the last 50 ms window at 2 ms resolution (25 points each): `va, vb, vc, vd, vq, f_hz` |
| `features` | on-board signal processing: positive-sequence voltage, vq mean/RMS, local frequency, mean/std/RoCoF, zero-sequence RMS, per-phase RMS and DC offset, dominant vq oscillation (FFT over 200 ms), local-minus-PMU frequency |
| `pmu` | `f_hz`, `v_pos_pu`, `age_ms` of the latest PMU report that has arrived |
| `trend` | the last 10 windows of local frequency, PMU frequency, vq RMS, zero-sequence RMS and positive-sequence voltage |
| status | `task_id`, `task_name`, `task_brief` (on reset), `step`, `time_s`, `inverter_mode` (normal / holdover / disconnected) |
| `reward_breakdown` | the typed reward model with every component of the last step's reward |
| `episode_result` | sent only when `done`: `grader_score`, score breakdown, total reward, and the revealed ground-truth scenario |

`GET /state` returns `PLLState`. The scenario ground truth stays `null` until the episode ends, so it can't be used
to cheat.

## Reward (dense, every step)
Action *t* is judged against window *t*, the data the agent actually saw. Once an attack starts, the system counts as
compromised until the episode ends.

| component | value |
|---|---|
| true positive / true negative | +0.10 / +0.02 |
| missed detection / **false alarm** | −0.10 / **−0.25** |
| early-detection bonus (one-time) | up to +0.5, decaying linearly to 0 at 500 ms |
| calibration (Brier) | 0.04 · (1 − 4·Brier) ∈ [−0.12, +0.04] |
| classification (tasks 1–2) | +0.05 correct / −0.02 wrong |
| holdover / disconnected (per window) | −0.02 / −0.04 (degraded operation / lost generation) |
| trip | −0.3 if under attack, **−1.5 on a healthy grid** (destructive action) |
| loss of synchronism | −1.0 once, plus −0.3 per window while desynchronised |

Per-step range: **[−3.21, +0.69]**. "Always alarm", "always hold over" and "always trip" all give negative returns.

## Graders (deterministic, 0–1)
They share these building blocks: false-alarm rate **FA** over clean windows, specificity **max(0, 1 − 2·FA)**,
timeliness **T** of the first detection, recall **R**, and classification accuracy **C** over compromised windows.

* **easy:** `(0.6·T + 0.4·R) · spec`, where T = 1 up to 100 ms of delay and falls to 0 at 1 s
* **medium:** `(0.35·T + 0.25·R + 0.40·C) · spec`
* **hard:** `(0.55·P·Q + 0.30·T + 0.15·R) · spec`. P = 1 if holdover or disconnect was engaged before the
  counterfactual desync time *and* the inverter never lost sync. Q = 1.0 for holdover, 0.6 for disconnect. T is
  measured against the time-to-desync budget.
* **attack-free episodes:** `spec`, multiplied by 0.5 for a false trip (0.4 in the hard task) and by 0.75 for a false
  holdover.

Scores are clamped to [0.01, 0.99] for validators that require strictly open (0, 1) scores.

## Baseline scores

100 seeds per task (seeds 1000–1099), from `python scripts/evaluate_baselines.py --seeds 100`:

| Policy | Task 0 (easy) | Task 1 (medium) | Task 2 (hard) | Mean |
|---|---|---|---|---|
| Always alarm | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 |
| Never alarm | 0.196 ± 0.384 | 0.226 ± 0.406 | 0.167 ± 0.359 | 0.196 |
| Random (p = 0.5) | 0.078 ± 0.106 | 0.063 ± 0.075 | 0.043 ± 0.056 | 0.061 |
| **Physics heuristic** (`pll_cyber/baselines.py`) | 0.986 ± 0.003 | 0.920 ± 0.150 | 0.762 ± 0.177 | 0.889 |
| Oracle (reads ground truth, upper bound) | 0.990 ± 0.000 | 0.990 ± 0.000 | 0.990 ± 0.000 | 0.990 |

The physics heuristic is what a protection engineer would write by hand. It fits the zero-sequence voltage with
DC + 50 Hz terms and checks the residual, and it integrates the latency-compensated local-vs-PMU frequency mismatch
into an angle estimate. The hard task leaves a lot of headroom: it requires reasoning about latency, separating a
spoofed frequency drift from a real one, and deciding when to mitigate.

**`inference.py`** (fixed seeds, 3 episodes per task) runs the **LLM agent**. The model gets the features, PMU data,
trends and a few DSP diagnostics, and makes every decision itself. Without an API key the script falls back to the
physics heuristic, which scores **0.989 / 0.944 / 0.717** on its evaluation seeds. LLM baseline numbers depend on
the model and endpoint you use; run the command below to reproduce them.

## Quick start

```bash
pip install -r requirements.txt

# run the server (HTTP + WebSocket) on :7860
uvicorn server.app:app --host 0.0.0.0 --port 7860        # or: python -m server.app

# baseline (LLM via any OpenAI-compatible endpoint; heuristic if no key)
export HF_TOKEN=...            # API_BASE_URL / MODEL_NAME optional
python inference.py            # in-process env; set ENV_URL=http://localhost:7860 to go through the server
```

Docker / Hugging Face Space:
```bash
docker build -t pll-cyberguard .
docker run -p 7860:7860 pll-cyberguard
```

### HTTP API
| endpoint | purpose |
|---|---|
| `POST /reset` | `{"task_id": 0-2, "seed": int, "session_id": str}` → `{observation, reward, done, session_id}` |
| `POST /step` | `{"action": {...}, "session_id": str}` → `{observation, reward, done, session_id}` |
| `GET /state?session_id=` | `PLLState` (ground truth hidden until done) |
| `GET /tasks` | task catalogue, reward range, action schema |
| `GET /schema`, `/metadata`, `/health`, `POST /mcp` | OpenEnv standard endpoints |
| `WS /ws` | persistent session per connection (used by `EnvClient`) |
| `GET /demo?task_id=&seed=` | runs the heuristic baseline and returns the trace (used by the landing page) |

openenv-core's own HTTP `/reset` and `/step` are stateless. This server replaces them with **session-aware**
versions that keep the same request and response shapes. Clients that omit `session_id` share a `"default"` session.

### Python client (WebSocket)
```python
from pll_cyber.client import PLLCyberEnv
from pll_cyber.models import PLLAction

with PLLCyberEnv(base_url="http://localhost:7860").sync() as env:
    r = env.reset(task_id=2, seed=7)
    while not r.done:
        r = env.step(PLLAction(attack_detected=False))
    print(r.observation.episode_result.grader_score)
```

## Validation & tests
```bash
pip install -r requirements-dev.txt
pytest -q                                   # 65 tests: physics, determinism, graders, API, inference
openenv validate .                          # local structure
openenv validate --url http://localhost:7860   # runtime contract (all 6 criteria pass)
```
The tests cover:
- transform identities and PLL lock, frequency-step and phase-jump dynamics
- natural events never desynchronise the inverter, while spoofing always does yet leaves vq unchanged
- holdover protects
- identical trajectories for the same seed
- ground truth stays hidden until done
- every reward stays inside the declared range
- degenerate policies (always / never / random / always-mitigate) score low and the oracle scores at least 0.9 on every seed
- session isolation and the API contract
- the inference log format and its LLM failure fallback

## Repository layout
```
pll_cyber/
  pll.py          SRF-PLL (Clarke/Park, PI loop, holdover mode)
  grid.py         true grid + natural events
  attacks.py      FDI attack models
  sensors.py      sensor noise/ADC, PMU stream, PMU-disciplined holdover
  scenarios.py    tasks, timing constants, seeded scenario sampling
  features.py     on-board signal processing for observations
  rewards.py      dense reward
  graders.py      per-task graders
  environment.py  OpenEnv Environment (reset/step/state)
  models.py       typed Action / Observation / Reward / State
  baselines.py    physics heuristic + trivial policies
  client.py       typed EnvClient (WebSocket)
server/app.py     FastAPI app (openenv create_app + session-aware HTTP)
inference.py      LLM baseline
scripts/          benchmark + submission validator
tests/            pytest suite
```

## Design choices & limitations
* **The agent gets the PMU's frequency and magnitude channels, not its synchrophasor angle.** The angle is used
  only by the inverter's holdover controller. Exposing it would make spoofing trivially visible; withholding it
  models a bandwidth-limited SCADA status feed.
* Injection attacks hit one phase, so they have a zero-sequence footprint. A balanced attacker is modelled by
  phase spoofing.
* The grid is a stiff source behind the point of common coupling. The inverter's current controller and its
  feedback on the grid voltage are not modelled; the harm is measured as angle desynchronisation.
* **Possible extensions:** grid-impedance and weak-grid dynamics, attacks on the PMU channel, replay attacks, and
  multi-inverter coordination.
