---
title: PLL Cyberattack Detection
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_file: Dockerfile
pinned: false
---
# PLL Cyberattack Detection — OpenEnv

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space%20Live-blue)](https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](openenv.yaml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)

> AI-driven cyberattack detection on SRF Phase-Locked Loops (PLLs) in grid-connected inverters.

## Overview

Phase-Locked Loops (PLLs) are critical components in grid-connected power converters, responsible for synchronizing the inverter's output with the utility grid. The Synchronous Reference Frame PLL (SRF-PLL) estimates grid frequency and phase angle by tracking the q-axis voltage component. Because of its critical role and reliance on sensor data, the SRF-PLL is a high-value target for **False Data Injection (FDI)** cyberattacks.

This OpenEnv environment simulates an SRF-PLL subjected to varied FDI attack scenarios. An AI agent acts as a cyber-guard: it monitors arriving time-windowed sensor observations—such as voltages and frequency deviations—and must accurately detect, classify, and mitigate attacks in real-time before grid synchronization is lost.

## Architecture

The environment relies on a discrete-time SRF-PLL simulation running at a 1 ms step size. A streamlined view of the signal flow is below:

```text
Grid Voltage (50Hz)
     │
     ▼
[FDI Attack Injection]  ◄── Attacker injects a malicious signal on phase `va`
     │
     ▼
Clarke Transform (αβ)
     │
     ▼
Park Transform (dq)     ◄── Uses the currently estimated angle θ̂
     │
     ▼
PI Controller           ──► ω̂, θ̂ are updated continuously
     │
     ▼
Agent Observation       ──► Agent receives: `vq_window`, `omega_deviation_window`, `raw_voltages`
     │
     ▼
Agent Action            ──► Agent outputs: `attack_detected`, `attack_type`, `confidence`
```

## Inference Flow & Detector Walkthrough

To balance speed and accuracy across thousands of steps, the standard inference client (`inference.py`) deploys a **Smart Blending Strategy**:

1. **Environment Simulation (`env.py`)**: 
   Every step, the PLL updates its internal math based on potential attack injections. It yields a rich observation window of the last 20 frames for variables like $V_q$ and $\omega_{dev}$.
2. **Adaptive Physics-Informed Detector (`src/detector.py`)**: 
   Before returning the observation to the client, the environment evaluates the data using an intrinsic physics-based detector. This detector calibrates anomaly residuals during the first 20 "healthy" warm-up steps. It tracks variances and symmetry to identify stealthy voltage anomalies, providing a baseline `confidence` score.
3. **Smart Blending Client (`inference.py`)**: 
   The client receives the observation and the detector's baseline prediction. 
   * If the intrinsic detector has high confidence (> 50%), the client adopts its recommendation.
   * If the anomaly is ambiguous (confidence < 50%), the client queries its own **Rule-Based Heuristic Agent**, which monitors historical $V_q$ growth, monotonicity, and zero-crossing density.
   * *Optional*: If `USE_LLM=1` is set, the client uses an LLM (e.g., `Qwen2.5-72B`) for advanced reasoning. A resilient "circuit breaker" automatically transitions to the heuristic model if network or authentication failures occur.

## Tasks

The environment supports three sequentially evaluated difficulty levels:

| Task | ID | Difficulty | Attack Type | Objective | Score Metric |
|------|----|-----------|-------------|-----------|-------|
| Sinusoidal FDI | 0 | Easy | Sinusoidal Injection | Detect attack within 100 steps of initiation. | Time-decaying detection reward. |
| Multi-Attack Class. | 1 | Medium | Sinusoidal, Ramp, Pulse | Safely and correctly classify the specific attack type. | Accuracy and speed aggregate. |
| Stealthy Detection | 2 | Hard | Low-amplitude phase drift | Detect slow deviations before the PLL loses lock (θ_error > 5°). | Preventative lock-loss metric. |

## Observation Space

At each step, the environment provides a JSON observation containing:

| Field | Shape | Description |
|-------|-------|-------------|
| `vq_window` | `[20]` | q-axis voltage error signal (pu). |
| `vd_window` | `[20]` | d-axis voltage (pu). |
| `omega_window` | `[20]` | Normalized frequency deviation from nominal. |
| `omega_deviation_window` | `[20]` | Frequency deviation from nominal (rad/s). |
| `raw_voltages` | `[3]` | Raw three-phase voltages `[va, vb, vc]` (pu). |
| `step` | `scalar` | Current simulation time step. |
| `task_id` | `scalar` | Current task identifier (0, 1, or 2). |

**Total observation dimension**: 83 ($20 \times 4 + 3$)

## Action Space

Agents must return a structured JSON response predicting the system state:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `attack_detected` | `bool` | — | True if malicious injection is suspected. |
| `attack_type` | `int` | 0–4 | 0=None, 1=Sinusoidal, 2=Ramp, 3=Pulse, 4=Stealthy. |
| `confidence` | `float` | 0.0–1.0 | Absolute predictive certainty. |
| `protective_action` | `int` | 0–3 | Suggested mitigation: 0=None, 1=Alert, 2=Reduce Power, 3=Disconnect. |

## Setup & API Usage

The system acts as a standard REST API server over port `7860`.

### Local Setup

**Via Python (Recommended)**:
```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

**Via Docker**:
```bash
docker build -t pll-cyberattack-env .
docker run -p 7860:7860 pll-cyberattack-env
```

### Environment Variables

Configure execution behavior locally via a `.env` file (see `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Custom endpoint for Language Models. |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Internal Model identifier. |
| `HF_TOKEN` | — | HuggingFace or valid proxy API key. |
| `USE_LLM` | `1` | Set to `1` to run the active LLM agent, `0` for pure heuristics. |

### REST Endpoints

1. **POST `/reset`**
   Initializes the environment for a specific task.
   ```bash
   curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": 0, "seed": 42}'
   ```

2. **POST `/step`**
   Submit an action based on recent observations and advance by one tick.
   ```bash
   curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"attack_detected": false, "attack_type": 0, "confidence": 0.5, "protective_action": 0}'
   ```

3. **GET `/health`**
   Returns operational status and step numbers.

## Baseline Performance

The default hybrid strategy outlined in `inference.py` consistently yields the following evaluation bounds across a full 500-step envelope:

* **Task 0 (Sinusoidal FDI):** 0.9900 
* **Task 1 (Multi-Attack Classification):** ~0.8720
* **Task 2 (Stealthy Drift):** ~0.1639
* **Aggregate System Average:** `0.6786`

---
🚀 **Live Environment Hosted on HuggingFace Spaces**: [krishuggingface/CyberAttack-PLL](https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL)
