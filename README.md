# PLL Cyberattack Detection — OpenEnv

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space%20Live-blue)](https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](openenv.yaml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)

> AI-driven cyberattack detection on SRF Phase-Locked Loops (PLLs) in grid-connected inverters.

## Overview

Phase-Locked Loops (PLLs) are critical components in grid-connected power converters that synchronize the inverter's output with the utility grid. The Synchronous Reference Frame PLL (SRF-PLL) estimates grid frequency and phase angle by tracking the q-axis voltage component — making it a high-value target for **False Data Injection (FDI)** cyberattacks.

This OpenEnv environment simulates an SRF-PLL under various FDI attack scenarios. An AI agent monitors time-windowed sensor observations (voltages, frequency deviations) and must detect, classify, and respond to attacks in real time before they cause loss of grid synchronization.

## Architecture

```
Grid Voltage (50Hz)
     │
     ▼
[FDI Attack Injection] ◄── Attacker injects false signal on va
     │
     ▼
Clarke Transform (αβ)
     │
     ▼
Park Transform (dq) ◄── uses estimated angle θ̂
     │
     ▼
PI Controller ──► ω̂, θ̂ updated
     │
     ▼
Agent observes: vq_window, omega_deviation_window, raw_voltages
     │
     ▼
Agent outputs: attack_detected, attack_type, confidence
```

## Inference & Detection Strategy

The environment natively features an **Adaptive Physics-Informed Detector** (`src/detector.py`) that calibrates anomaly residuals (R1, R3, R4, R5) during the PLL warm-up phase to identify stealthy voltage and frequency deviations.

The default inference client (`inference.py`) deploys a **Smart Blending Agent** strategy:
1. It relies primarily on the environment's `AdaptiveDetector` output passed via `info["detector"]`.
2. As a **safety net**, if the detector's classification confidence drops below 50% (`< 0.5`) on ambiguous anomalies, the client dynamically falls back to an independent, cumulative **Rule-Based Heuristic Agent**.
3. Optionally, an LLM agent (e.g., `Qwen/Qwen2.5-72B-Instruct`) can be enabled natively via the `USE_LLM=1` environment variable.

## Tasks

| Task | ID | Difficulty | Attack Type | Objective | Score |
|------|----|-----------|-------------|-----------|-------|
| Sinusoidal FDI Detection | 0 | Easy | Sinusoidal injection | Detect within 100 steps | Time-based decay |
| Multi-Attack Classification | 1 | Medium | Sinusoidal/Ramp/Pulse | Classify attack type | Accuracy + speed |
| Stealthy Attack Detection | 2 | Hard | Low-amplitude phase drift | Detect before lock loss | Prevention score |

## Observation Space

Each step provides a JSON observation with the following fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `vq_window` | `[20]` | q-axis voltage error signal (pu) |
| `vd_window` | `[20]` | d-axis voltage (pu) |
| `omega_window` | `[20]` | Normalized frequency deviation from nominal |
| `omega_deviation_window` | `[20]` | Frequency deviation from nominal (rad/s) |
| `raw_voltages` | `[3]` | Raw three-phase voltages `[va, vb, vc]` (pu) |
| `step` | scalar | Current simulation step |
| `task_id` | scalar | Task identifier (0, 1, or 2) |

**Total observation dimension**: 83 (20+20+20+20+3)

## Action Space

Agents return a JSON action each step:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `attack_detected` | `bool` | — | Whether an attack is detected |
| `attack_type` | `int` | 0–4 | 0=none, 1=sinusoidal, 2=ramp, 3=pulse, 4=stealthy |
| `confidence` | `float` | 0.0–1.0 | Agent's confidence in its classification |
| `protective_action` | `int` | 0–3 | 0=none, 1=alert, 2=reduce power, 3=disconnect |

## API Endpoints

### Reset Environment
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 0, "seed": 42}'
```

### Step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"attack_detected": false, "attack_type": 0, "confidence": 0.5, "protective_action": 0}'
```

### Get State
```bash
curl http://localhost:7860/state
```

### Health Check
```bash
curl http://localhost:7860/health
```

## Quick Start

### With Docker

```bash
docker build -t pll-cyberattack-env .
docker run -p 7860:7860 pll-cyberattack-env
```

### Without Docker

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace API token |

## Baseline Performance

The default hybrid strategy (Adaptive Detector + Heuristic Fallback) achieves the following baseline scores evaluated locally over 500-step episodes:

* **Task 0 (Sinusoidal FDI):** 1.0000 
* **Task 1 (Multi-Attack Classification):** 0.8720
* **Task 2 (Stealthy Drift):** 0.1639
* **Average Score:** `0.6786`

## Live Demo

🚀 **HuggingFace Space**: [https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL](https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL)
