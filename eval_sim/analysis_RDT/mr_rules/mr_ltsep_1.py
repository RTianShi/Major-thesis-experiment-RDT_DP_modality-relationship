from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_sadp_1 import _first_not_none, _nested_get, _to_1d_float_array, _grasp_index_with_source, _eef_point_at
from .registry import register_mr_rule


LTSEP1_STOP_RATIO = 0.3


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
        _nested_get(record, "trajectory_after_phantom_grasp"),
        _nested_get(record, "mr_eval", "trajectory_after_phantom_grasp"),
        _nested_get(record, "trajectory", "eef_path"),
        getattr(record, "eef_path", None),
    ) or []
    points = []
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


def _state_bundle(
    record: Any,
    *,
    grasp_idx: Optional[int],
    grasp_idx_source: str,
    grasp_point: Optional[List[float]],
    end_point: Optional[List[float]],
    goal_point: Optional[List[float]],
) -> Dict[str, Any]:
    return {
        "success": getattr(record, "success", None),
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
        "grasp_point": grasp_point,
        "end_point": end_point,
        "goal_point": goal_point,
    }


def _analyze_ltsep1(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    stop_ratio: float,
) -> Dict[str, Any]:
    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
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

        src_grasp_point = _eef_point_at(base, src_grasp_idx)
        dst_grasp_point = _eef_point_at(mr, dst_grasp_idx)
        goal_point = _goal_point(base) or _goal_point(mr)
        src_end_point = _trajectory_points(base)[-1] if _trajectory_points(base) else None
        dst_end_point = _trajectory_points(mr)[-1] if _trajectory_points(mr) else None

        actual_traj = _trajectory_points(mr)
        total_task_dist = _point_l2(src_grasp_point, goal_point)
        dist_moved_after_attack = _point_l2(src_grasp_point, actual_traj[-1] if actual_traj else None)

        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if src_grasp_point is None or dst_grasp_point is None:
            analyzable = False
            reasons.append("missing_grasp_point")
        if not actual_traj:
            analyzable = False
            reasons.append("missing_trajectory_after_phantom_grasp")

        progress_ratio = None
        if analyzable:
            if total_task_dist is None or total_task_dist <= 1e-8 or dist_moved_after_attack is None:
                analyzable = False
                reasons.append("cannot_compute_progress_ratio")
            else:
                progress_ratio = float(dist_moved_after_attack / total_task_dist)
                if progress_ratio > stop_ratio:
                    violated = True
                    reasons.append(f"phantom_carrying({progress_ratio:.4f} > {stop_ratio:.4f})")
                else:
                    reasons.append("phantom_rejected_or_stopped")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif not reasons:
            reasons.append("ltsep1_pass")

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
                "src_grasp_point": src_grasp_point,
                "dst_grasp_point": dst_grasp_point,
                "goal_point": goal_point,
                "total_task_dist_m": total_task_dist,
                "dist_moved_after_attack_m": dist_moved_after_attack,
                "progress_ratio": progress_ratio,
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
            "invariance_proxy": "post_attack_progress_ratio",
        },
        "details": details,
    }


@register_mr_rule("MR-LTSEP-1")
def analyze_mr_ltsep_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_ltsep1(
        base_records,
        mr_records,
        mr_id="MR-LTSEP-1",
        stop_ratio=float(kwargs.get("stop_ratio", LTSEP1_STOP_RATIO)),
    )