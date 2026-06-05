from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _mr6_field(record: Any, key: str):
    mr_eval = getattr(record, "mr_eval", None)
    if isinstance(mr_eval, dict):
        return mr_eval.get(key)
    return None


def _xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return arr[:3]


def _trajectory_points(record: Any, attr_name: str) -> List[np.ndarray]:
    seq = getattr(record, attr_name, None) or []
    points: List[np.ndarray] = []
    for item in seq:
        arr = _xyz(item)
        if arr is not None:
            points.append(arr)
    return points


def _goal_point(record: Any) -> Optional[np.ndarray]:
    goal_candidates = []
    mr_eval = getattr(record, "mr_eval", None)
    trajectory = getattr(record, "trajectory", None)
    goal_candidates.extend(
        [
            getattr(record, "goal_point", None),
            mr_eval.get("goal_point") if isinstance(mr_eval, dict) else None,
            trajectory.get("goal_point") if isinstance(trajectory, dict) else None,
        ]
    )
    for candidate in goal_candidates:
        arr = _xyz(candidate)
        if arr is not None:
            return arr
    return None


def _progress_ratio_to_goal(record: Any) -> Optional[float]:
    eef_points = _trajectory_points(record, "eef_path")
    goal_point = _goal_point(record)
    if not eef_points or goal_point is None:
        return None

    start_dist = float(np.linalg.norm(goal_point - eef_points[0]))
    end_dist = float(np.linalg.norm(goal_point - eef_points[-1]))
    if not np.isfinite(start_dist) or start_dist <= 1e-12:
        return None
    return float((start_dist - end_dist) / start_dist)


@register_mr_rule("MR6")
@register_mr_rule("MR-6")
@register_mr_rule("Action-Optimality-Completeness")
@register_mr_rule("mr6")
def analyze_mr6_action_optimality_completeness(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    progress_ratio_threshold = float(kwargs.get("progress_ratio_threshold", 0.3))

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        base_success = bool(getattr(base, "success", False))
        mr_success = bool(getattr(mr, "success", False))
        injection_applied = bool(_mr6_field(mr, "mr6_injection_applied"))
        observed_lift_m = _mr6_field(mr, "mr6_observed_lift_m")
        return_error_m = _mr6_field(mr, "mr6_return_error_m")
        grasp_frame_index = getattr(mr, "mr_eval_grasp_frame_index", None)
        progress_ratio = _progress_ratio_to_goal(mr)

        analyzable = injection_applied and (progress_ratio is not None)
        violated = False
        reasons: List[str] = []

        if not injection_applied:
            analyzable = False
            reasons.append("mr6_injection_not_applied")
        elif progress_ratio is None:
            analyzable = False
            reasons.append("cannot_compute_progress_ratio")
        elif grasp_frame_index is None and progress_ratio > progress_ratio_threshold:
            violated = True
            reasons.append(
                "VIOLATION: skipped_real_grasp_and_progressed_toward_goal "
                f"(progress_ratio={progress_ratio:.6f} > {progress_ratio_threshold:.6f})"
            )
        else:
            reasons.append("no_phantom_progress_after_disturbance")

        if not analyzable:
            unavailable_count += 1
        elif violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": base_success,
                "mr_success": mr_success,
                "mr6_injection_applied": injection_applied,
                "mr6_observed_lift_m": observed_lift_m,
                "mr6_return_error_m": return_error_m,
                "mr_grasp_frame_index": grasp_frame_index,
                "progress_ratio": progress_ratio,
                "progress_ratio_threshold": progress_ratio_threshold,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR6",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "progress_ratio_threshold": progress_ratio_threshold,
            "violation_rule": "(grasp_frame_index == None) AND (progress_ratio > 0.3)",
            "note": "If the disturbed run skips the real grasp and still advances toward the goal, mark violation.",
        },
        "details": details,
    }
