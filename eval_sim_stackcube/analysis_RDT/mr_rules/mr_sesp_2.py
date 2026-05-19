from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SESP2_DIST_TOL = 0.03


def _index_by_seed_or_episode(records: List[Any]) -> Dict[int, Any]:
    out = {}
    for r in records:
        key = r.seed if r.seed is not None else r.episode_id
        if key is not None:
            out[key] = r
    return out


def _xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _final_valid_xyz(seq: Any) -> Optional[np.ndarray]:
    items = seq or []
    for item in reversed(items):
        arr = _xyz(item)
        if arr is not None:
            return arr
    return None


def _goal_xyz(rec: Any) -> Optional[np.ndarray]:
    return _xyz(getattr(rec, "goal_point", None))


def _cube_xyz(rec: Any) -> Optional[np.ndarray]:
    for attr in ("cube_pos", "src_cube_pos"):
        arr = _final_valid_xyz(getattr(rec, attr, None))
        if arr is not None:
            return arr
    return None


def _dxy_to_goal(rec: Any) -> Optional[float]:
    cube_xyz = _cube_xyz(rec)
    goal_xyz = _goal_xyz(rec)
    if cube_xyz is None or goal_xyz is None:
        return None
    return float(np.linalg.norm(cube_xyz[:2] - goal_xyz[:2]))


@register_mr_rule("MR-SESP2")
@register_mr_rule("MR-SESP-2")
@register_mr_rule("mr_sesp_2")
def analyze_mr_sesp_2(base_records, mr_records, **kwargs):
    dist_tol = float(kwargs.get("dist_tol", SESP2_DIST_TOL))

    bmap = _index_by_seed_or_episode(base_records)
    mmap = _index_by_seed_or_episode(mr_records)
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        b = bmap[k]
        m = mmap[k]

        base_dist = _dxy_to_goal(b)
        mr_dist = _dxy_to_goal(m)

        analyzable = base_dist is not None and mr_dist is not None
        violated = False
        reasons = []

        if not analyzable:
            unavailable_count += 1
            if base_dist is None:
                reasons.append("missing_base_dist_to_goal")
            if mr_dist is None:
                reasons.append("missing_mr_dist_to_goal")
        else:
            dist_delta = float(mr_dist) - float(base_dist)
            if dist_delta > dist_tol:
                violated = True
                reasons.append(f"slip_failure_dist_increased({dist_delta:.4f}m)")
            else:
                reasons.append("distance_pass")
        dist_delta = (float(mr_dist) - float(base_dist)) if analyzable else None

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "base_success": bool(b.success) if b.success is not None else None,
                "mr_success": bool(m.success) if m.success is not None else None,
                "base_dist_to_goal_dxy_m": base_dist,
                "mr_dist_to_goal_dxy_m": mr_dist,
                "dist_delta_m": dist_delta,
                "dist_tol_m": dist_tol,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None
    return {
        "mr_id": "MR-SESP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "dist_tol_m": dist_tol,
            "metric": "horizontal_distance_to_goal_dxy",
            "violation_rule": "mr_dist_to_goal_dxy > base_dist_to_goal_dxy + dist_tol",
        },
        "details": details,
    }
