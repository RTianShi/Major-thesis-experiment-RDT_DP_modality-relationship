from .registry import register_mr_rule

import numpy as np
from typing import List, Dict, Any


def _flatten_eef_path(eef_path):
    if not eef_path or not isinstance(eef_path[0], list):
        return []
    if isinstance(eef_path[0][0], list):
        return [point[0] for point in eef_path if point and isinstance(point[0], list)]
    return eef_path


def _end_point(eef_path):
    flat = _flatten_eef_path(eef_path)
    if not flat:
        return None
    p = flat[-1]
    if p is None or len(p) < 3:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _l2_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    a = np.asarray(p1, dtype=np.float64).reshape(-1)
    b = np.asarray(p2, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3:
        return None
    return float(np.linalg.norm(a[:3] - b[:3]))


@register_mr_rule("MR-JDCP-2")
def analyze_mr_jdcp_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    ghost_goal_tol = float(kwargs.get("ghost_goal_tol", 0.05))

    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        base = bmap[k]
        mr = mmap[k]

        dst_success = getattr(mr, "success", None)
        original_goal_pos = getattr(base, "mr_eval_initial_goal_pos", None)
        dst_end_point = _end_point(getattr(mr, "eef_path", None))
        dist_to_ghost_goal = _l2_distance(dst_end_point, original_goal_pos)

        analyzable = True
        violated = False
        reasons = []

        if dst_success is True:
            violated = True
            reasons.append("Extreme_Overfitting_Task_Completed_Without_Goal")

        if original_goal_pos is None or dst_end_point is None:
            analyzable = False
            unavailable_count += 1
            reasons.append("missing_original_goal_or_dst_end_point")
        elif dist_to_ghost_goal is not None and dist_to_ghost_goal < ghost_goal_tol:
            violated = True
            reasons.append("Blind_Execution_Reached_Ghost_Goal")

        if analyzable and not reasons:
            reasons.append("Sensitivity_Triggered_Stopped_Or_Lost")

        if violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": k,
            "source_success": getattr(base, "success", None),
            "followup_success": dst_success,
            "original_goal_pos": original_goal_pos,
            "dst_end_point": dst_end_point,
            "dist_to_ghost_goal_m": dist_to_ghost_goal,
            "ghost_goal_tol_m": ghost_goal_tol,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-JDCP-2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "ghost_goal_tol_m": ghost_goal_tol,
        },
        "details": details,
    }
