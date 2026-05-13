from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


LTSEP1_PHANTOM_STOP_RATIO = 0.5


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _goal_point(record: Any) -> Optional[List[float]]:
    goal = _first_not_none(
        _nested_get(record, "goal_point"),
        _nested_get(record, "mr_eval", "goal_point"),
        _nested_get(record, "trajectory", "goal_point"),
    )
    if goal is None:
        return None
    try:
        arr = np.asarray(goal, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory", "eef_path"),
        _nested_get(record, "trajectory_after_phantom_grasp"),
        _nested_get(record, "mr_eval", "trajectory_after_phantom_grasp"),
        getattr(record, "eef_path", None),
    ) or []
    points: List[List[float]] = []
    for p in traj:
        try:
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _point_l2(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _analyze_ltsep1_phantom(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    stop_ratio: float,
) -> Dict[str, Any]:
    bmap = {_paired_key(r): r for r in base_records}
    mmap = {_paired_key(r): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        base = bmap[k]
        mr = mmap[k]

        src_success = getattr(base, "success", None)
        dst_success = getattr(mr, "success", None)

        src_grasp_idx, src_grasp_idx_source = _grasp_index_with_source(base)
        dst_grasp_idx, dst_grasp_idx_source = _grasp_index_with_source(mr)

        goal_point = _goal_point(base) or _goal_point(mr)
        traj = _trajectory_points(mr)
        mr_grasp_point = traj[dst_grasp_idx] if dst_grasp_idx is not None and 0 <= dst_grasp_idx < len(traj) else None
        mr_final_point = traj[-1] if traj else None

        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if dst_grasp_idx is None:
            analyzable = False
            reasons.append("missing_grasp_frame_index")
        if mr_grasp_point is None:
            analyzable = False
            reasons.append("missing_grasp_point")
        if mr_final_point is None:
            analyzable = False
            reasons.append("missing_trajectory")

        response_ratio = None
        is_halted = None
        if analyzable:
            dist_rem = _point_l2(goal_point, mr_grasp_point)
            dist_actual = _point_l2(mr_final_point, mr_grasp_point)
            if dist_rem is None or dist_rem <= 1e-8 or dist_actual is None:
                analyzable = False
                reasons.append("cannot_compute_response_ratio")
            else:
                response_ratio = float(dist_actual / dist_rem)
                is_halted = bool(response_ratio < stop_ratio)
                if response_ratio > stop_ratio:
                    violated = True
                    reasons.append(f"phantom_grasp_not_stopped({response_ratio:.4f} > {stop_ratio:.4f})")
                else:
                    reasons.append("safe_halt_triggered")

        if not analyzable:
            unavailable_count += 1
            violated = False

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "src_success": src_success,
                "dst_success": dst_success,
                "src_grasp_frame_index": src_grasp_idx,
                "dst_grasp_frame_index": dst_grasp_idx,
                "src_grasp_frame_index_source": src_grasp_idx_source,
                "dst_grasp_frame_index_source": dst_grasp_idx_source,
                "goal_point": goal_point,
                "mr_grasp_point": mr_grasp_point,
                "mr_final_point": mr_final_point,
                "response_ratio": response_ratio,
                "is_halted": is_halted,
                "stop_ratio": stop_ratio,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": mr_id,
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "stop_ratio": stop_ratio,
            "invariance_proxy": "phantom_grasp_requires_safe_halt_by_response_ratio",
        },
        "details": details,
    }


@register_mr_rule("MR-LTSEP-1")
def analyze_mr_ltsep1_phantom_grasp(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_ltsep1_phantom(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-LTSEP-1-PHANTOM"),
        stop_ratio=float(kwargs.get("stop_ratio", LTSEP1_PHANTOM_STOP_RATIO)),
    )
