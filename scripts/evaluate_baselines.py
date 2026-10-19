"""
Benchmark reference policies over many seeds (in-process, no network).

    python scripts/evaluate_baselines.py --seeds 100

Prints a Markdown table (mean ± std of grader scores per task) used in the README.
The oracle reads ground truth and is an upper bound, not a real agent.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pll_cyber.baselines import ConstantAgent, PhysicsHeuristicAgent, RandomAgent  # noqa: E402
from pll_cyber.environment import PLLCyberEnvironment  # noqa: E402
from pll_cyber.evaluation import run_episode  # noqa: E402


class Oracle:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        sc = self.env._sc
        lab = sc.onset_step is not None and obs["step"] >= sc.onset_step
        return {"attack_detected": lab, "attack_type": sc.attack.type_id if lab else 0, "confidence": 1.0,
                "protective_action": 2 if (lab and sc.task_id == 2) else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    seeds = range(1000, 1000 + args.seeds)          # disjoint from inference.py's evaluation seeds
    rnd = itertools.count()
    policies = {
        "Always alarm": lambda env, t: ConstantAgent(True, [1, 1, 4][t]),
        "Never alarm": lambda env, t: ConstantAgent(False),
        "Random (p=0.5)": lambda env, t: RandomAgent(seed=next(rnd)),
        "Physics heuristic": lambda env, t: PhysicsHeuristicAgent(),
        "Oracle (ground truth)": lambda env, t: Oracle(env),
    }
    env = PLLCyberEnvironment()
    t0 = time.time()
    print("| Policy | Task 0 (easy) | Task 1 (medium) | Task 2 (hard) | Mean |")
    print("|---|---|---|---|---|")
    for name, make in policies.items():
        cells, means = [], []
        for t in range(3):
            s = np.array([run_episode(make(env, t), t, seed, env)["score"] for seed in seeds])
            cells.append(f"{s.mean():.3f} ± {s.std():.3f}")
            means.append(s.mean())
        print(f"| {name} | " + " | ".join(cells) + f" | {np.mean(means):.3f} |", flush=True)
    print(f"\n{args.seeds} seeds per task, {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
