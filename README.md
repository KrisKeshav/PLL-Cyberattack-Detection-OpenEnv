# PLL Cyberattack Detection — OpenEnv

> AI-driven cyberattack detection on SRF Phase-Locked Loops (PLLs) in grid-connected inverters.

## Overview

Phase-Locked Loops (PLLs) are critical components in grid-connected power converters that synchronize the inverter's output with the utility grid. A Synchronous Reference Frame PLL (SRF-PLL) estimates grid frequency and phase angle — making it a high-value target for **False Data Injection (FDI)** cyberattacks.

This OpenEnv environment simulates an SRF-PLL under various cyberattack scenarios and challenges AI agents to detect, classify, and respond to attacks in real time using only time-windowed sensor observations.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| **Task 0** | Easy | Detect whether a sinusoidal FDI attack is present (binary detection) |
| **Task 1** | Medium | Detect and classify the attack type — sinusoidal, ramp, or pulse |
| **Task 2** | Hard | Detect stealthy, low-amplitude attacks before the PLL loses lock |

## Observation Space

Each step provides a JSON observation:

| Field | Shape | Description |
|-------|-------|-------------|
| `vq_window` | `[20]` | q-axis voltage error (last 20 steps) |
| `vd_window` | `[20]` | d-axis voltage (last 20 steps) |
| `omega_window` | `[20]` | Estimated frequency, normalized (last 20 steps) |
| `omega_deviation_window` | `[20]` | Frequency deviation from nominal in rad/s |
| `raw_voltages` | `[3]` | Three-phase voltages `[va, vb, vc]` at current step |
| `step` | `int` | Current simulation step |
| `task_id` | `int` | Task identifier (0, 1, or 2) |

## Action Space

Agents return a JSON action each step:

```json
{
  "attack_detected": true,
  "attack_type": 1,
  "confidence": 0.85,
  "protective_action": 1
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `attack_detected` | `bool` | — | Whether an attack is detected |
| `attack_type` | `int` | 0–4 | 0=none, 1=sinusoidal, 2=ramp, 3=pulse, 4=stealthy |
| `confidence` | `float` | 0.0–1.0 | Agent's confidence in its classification |
| `protective_action` | `int` | 0–3 | 0=none, 1=alert, 2=reduce power, 3=disconnect |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /reset` | Reset | Start a new episode. Body: `{"task_id": 0}` |
| `POST /step` | Step | Submit an action and receive the next observation |
| `GET /state` | State | Get the current environment state |
| `GET /health` | Health | Health check endpoint |

## Running Locally

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

### Running the Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"
python inference.py
```

Set `USE_LLM=1` to use the LLM agent instead of the default rule-based heuristic.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace API token |
| `ENV_URL` | No | HF Space URL | Environment server URL |
| `USE_LLM` | No | `0` | Set to `1` to use LLM agent |

## Live Demo

🚀 **HuggingFace Space**: [https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL](https://huggingface.co/spaces/krishuggingface/CyberAttack-PLL)

## License

MIT
