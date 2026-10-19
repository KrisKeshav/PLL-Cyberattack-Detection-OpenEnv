"""
Deterministic per-episode graders, one per task, scores in [0, 1].

All graders share the same building blocks, computed from the decision log:

    FA   false-alarm rate: fraction of clean windows (before onset) flagged
    spec specificity factor max(0, 1 − 2·FA) — every false-alarm window costs
         twice its share, so random / chattering policies are punished, not
         just "always on" ones (an operator stops trusting a noisy IDS)
    T    timeliness of the first detection after onset
    R    recall: fraction of compromised windows flagged
    C    classification: fraction of compromised windows flagged with the
         correct attack_type

Attack-free episodes are scored on specificity alone (and on avoiding
destructive protective actions), so "always alarm" scores ≈ 0 everywhere and
"never alarm" only collects the attack-free share of episodes.

Scores are clamped to [0.01, 0.99] for compatibility with validators that
require strictly-open (0, 1) scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SCORE_MIN, SCORE_MAX = 0.01, 0.99


def _clamp(x: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, x))


def _linear_timeliness(delay_ms: Optional[float], full_ms: float, zero_ms: float) -> float:
    if delay_ms is None:
        return 0.0
    if delay_ms <= full_ms:
        return 1.0
    return max(0.0, 1.0 - (delay_ms - full_ms) / (zero_ms - full_ms))


def _common(log: List[Dict[str, Any]], onset: Optional[int], interval_ms: float) -> Dict[str, Any]:
    clean = [e for e in log if onset is None or e["step"] < onset]
    post = [e for e in log if onset is not None and e["step"] >= onset]
    fa = sum(e["detected"] for e in clean) / len(clean) if clean else 0.0
    first = next((e["step"] for e in post if e["detected"]), None)
    delay_ms = (first - onset) * interval_ms if first is not None else None
    recall = sum(e["detected"] for e in post) / len(post) if post else 0.0
    false_trip = any(e["protective_action"] == 3 for e in clean)
    false_holdover = any(e["protective_action"] == 2 for e in clean)
    return {
        "false_alarm_rate": fa,
        "specificity": max(0.0, 1.0 - 2.0 * fa),
        "first_detection_step": first,
        "detection_delay_ms": delay_ms,
        "recall": recall,
        "false_trip": false_trip,
        "false_holdover": false_holdover,
        "_post": post,
    }


def _finish(score: float, parts: Dict[str, Any]) -> Dict[str, Any]:
    parts.pop("_post", None)
    parts["raw_score"] = round(score, 6)
    parts["score"] = round(_clamp(score), 6)
    return parts


def grade_easy(log, onset, interval_ms, **_) -> Dict[str, Any]:
    p = _common(log, onset, interval_ms)
    if onset is None:
        score = p["specificity"]
    else:
        p["timeliness"] = _linear_timeliness(p["detection_delay_ms"], 100.0, 1000.0)
        score = (0.6 * p["timeliness"] + 0.4 * p["recall"]) * p["specificity"]
    if p["false_trip"]:
        score *= 0.5
    return _finish(score, p)


def grade_medium(log, onset, interval_ms, true_type: int = 0, **_) -> Dict[str, Any]:
    p = _common(log, onset, interval_ms)
    if onset is None:
        score = p["specificity"]
    else:
        post = p["_post"]
        p["timeliness"] = _linear_timeliness(p["detection_delay_ms"], 100.0, 1000.0)
        p["classification"] = sum(e["detected"] and e["attack_type"] == true_type for e in post) / len(post)
        score = (0.35 * p["timeliness"] + 0.25 * p["recall"] + 0.40 * p["classification"]) * p["specificity"]
    if p["false_trip"]:
        score *= 0.5
    return _finish(score, p)


def grade_hard(log, onset, interval_ms, unmitigated_desync_step: Optional[int] = None,
               actual_desync_step: Optional[int] = None, **_) -> Dict[str, Any]:
    """Stealthy phase spoofing: detection *and* timely mitigation.

    protection P = 1 if the inverter never lost synchronism and a protective
    action (holdover/disconnect) was engaged by the time the unprotected
    inverter would have desynchronised. Holdover keeps generating, so it is
    worth more than a disconnect (quality Q = 1.0 vs 0.6).
    """
    p = _common(log, onset, interval_ms)
    p["unmitigated_desync_step"] = unmitigated_desync_step
    p["actual_desync_step"] = actual_desync_step

    if onset is None:
        score = p["specificity"]
        if p["false_trip"]:
            score *= 0.4
        elif p["false_holdover"]:
            score *= 0.75
        if actual_desync_step is not None:
            score *= 0.5
        return _finish(score, p)

    post = p["_post"]
    deadline = unmitigated_desync_step if unmitigated_desync_step is not None else log[-1]["step"] + 1
    mitig = next((e for e in post if e["protective_action"] in (2, 3)), None)
    engaged_in_time = mitig is not None and mitig["step"] < deadline
    protected = engaged_in_time and actual_desync_step is None
    q = 0.6 if any(e["protective_action"] == 3 for e in post) else 1.0
    p["mitigation_step"] = mitig["step"] if mitig else None
    p["protected"] = protected
    p["mitigation_quality"] = q if protected else 0.0

    budget_ms = max(interval_ms, (deadline - onset) * interval_ms)
    delay = p["detection_delay_ms"]
    p["timeliness"] = 0.0 if delay is None else max(0.0, 1.0 - delay / budget_ms)

    score = (0.55 * (q if protected else 0.0) + 0.30 * p["timeliness"] + 0.15 * p["recall"]) * p["specificity"]
    if p["false_trip"]:
        score *= 0.4
    elif p["false_holdover"]:
        score *= 0.75
    return _finish(score, p)


GRADERS = {0: grade_easy, 1: grade_medium, 2: grade_hard}


def grade_episode(task_id: int, log: List[Dict[str, Any]], onset: Optional[int],
                  interval_ms: float, **kwargs) -> Dict[str, Any]:
    return GRADERS[task_id](log, onset, interval_ms, **kwargs)
