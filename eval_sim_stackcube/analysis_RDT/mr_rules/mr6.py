from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


MR6_PROGRESS_RATIO_DELTA_TOL = 0.2


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory", "eef_path"),
        getattr(record, "eef_path", None),
    ) or []
    points: List[List[float]] = []
    for point in traj:
        try:
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _path_len(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _release_index(record: Any, grasp_idx: Optional[int]) -> Optional[int]:
    cmd = _first_not_none(
        _nested_get(record, "trajectory", "gripper_action_cmd"),
        getattr(record, "gripper_action_cmd", None),
    ) or []
    if grasp_idx is None:
        return None
    for idx in range(max(0, grasp_idx + 1), len(cmd)):
        try:
            value = float(cmd[idx])
        except Exception:
            continue
        if value >= 0.95:
            return idx
    if len(cmd) > grasp_idx + 1:
        return len(cmd) - 1
    return None


def _progress_ratio(record: Any) -> Optional[float]:
    traj = _trajectory_points(record)
    grasp_idx, _ = _grasp_index_with_source(record)
    release_idx = _release_index(record, grasp_idx)
    total_len = _path_len(traj)
    if grasp_idx is None or release_idx is None or total_len is None or total_len <= 1e-12:
        return None
    if not (0 <= grasp_idx < len(traj)) or not (0 <= release_idx < len(traj)) or release_idx <= grasp_idx:
        return None
    post_grasp_len = _path_len(traj[grasp_idx:release_idx + 1])
    if post_grasp_len is None:
        return None
    return float(post_grasp_len / total_len)


@register_mr_rule("MR6")
@register_mr_rule("MR-6")
@register_mr_rule("Action-Optimality-Completeness")
def analyze_mr6_action_optimality_completeness(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    progress_ratio_delta_tol = float(
        kwargs.get("progress_ratio_delta_tol", MR6_PROGRESS_RATIO_DELTA_TOL)
    )

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        base_ratio = _progress_ratio(base)
        mr_ratio = _progress_ratio(mr)
        analyzable = True
        violated = False
        reasons: List[str] = []

        if base_ratio is None:
            analyzable = False
            reasons.append("missing_base_progress_ratio")
        if mr_ratio is None:
            analyzable = False
            reasons.append("missing_mr_progress_ratio")

        progress_ratio_delta = None
        if analyzable:
            progress_ratio_delta = float(mr_ratio - base_ratio)
            if progress_ratio_delta > progress_ratio_delta_tol:
                violated = True
                reasons.append("VIOLATION: mr_progress_ratio - base_progress_ratio > 0.2")
            else:
                reasons.append("action_optimality_completeness_pass")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "base_success": getattr(base, "success", None),
            "mr_success": getattr(mr, "success", None),
            "base_progress_ratio": base_ratio,
            "mr_progress_ratio": mr_ratio,
            "progress_ratio_delta": progress_ratio_delta,
            "progress_ratio_delta_tol": progress_ratio_delta_tol,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": kwargs.get("mr_id", "MR6"),
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "progress_ratio_delta_tol": progress_ratio_delta_tol,
            "metric": "post_grasp_progress_ratio",
            "violation_rule": "mr_progress_ratio - base_progress_ratio > 0.2",
            "expectation": "injected_pause_or_detour_should_not_increase_release_progress_ratio_too_much",
        },
        "details": details,
    }
