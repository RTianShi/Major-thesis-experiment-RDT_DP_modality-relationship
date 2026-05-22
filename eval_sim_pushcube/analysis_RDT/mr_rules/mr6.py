from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


MR6_PROGRESS_RATIO_THRESHOLD = 0.3
MR6_CONTACT_DISTANCE_M = 0.035
MR6_CUBE_MOVE_THRESHOLD_M = 0.002


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


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


def _safe_bool_list(seq: Any) -> List[Optional[bool]]:
    out: List[Optional[bool]] = []
    for item in seq or []:
        if item is None:
            out.append(None)
        else:
            out.append(bool(item))
    return out


def _safe_float_list(seq: Any) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for item in seq or []:
        try:
            out.append(None if item is None else float(item))
        except Exception:
            out.append(None)
    return out


def _goal_point(record: Any) -> Optional[np.ndarray]:
    goal_points = (
        _trajectory_points(record, "goal_pos")
        or _trajectory_points(record, "target_pos")
        or _trajectory_points(record, "dst_cube_pos")
    )
    if goal_points:
        return goal_points[0]
    return None


def _has_contact(record: Any, contact_distance_m: float) -> Tuple[bool, str]:
    contacts = _safe_bool_list(getattr(record, "is_contact", None))
    if any(flag is True for flag in contacts):
        return True, "is_contact"

    distances = _safe_float_list(getattr(record, "tcp_to_obj_distance", None))
    if any(dist is not None and dist <= contact_distance_m for dist in distances):
        return True, "tcp_to_obj_distance"

    return False, "no_contact_signal"


def _cube_moved(record: Any, cube_move_threshold_m: float) -> Tuple[bool, Optional[float]]:
    cube_points = _trajectory_points(record, "cube_pos") or _trajectory_points(record, "src_cube_pos")
    if len(cube_points) < 2:
        return False, None

    start = cube_points[0]
    max_disp = 0.0
    for point in cube_points[1:]:
        max_disp = max(max_disp, float(np.linalg.norm(point - start)))
    return max_disp > cube_move_threshold_m, max_disp


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
@register_mr_rule("mr6")
def analyze_mr6_action_optimality_completeness(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    progress_ratio_threshold = float(
        kwargs.get("progress_ratio_threshold", MR6_PROGRESS_RATIO_THRESHOLD)
    )
    contact_distance_m = float(kwargs.get("contact_distance_m", MR6_CONTACT_DISTANCE_M))
    cube_move_threshold_m = float(
        kwargs.get("cube_move_threshold_m", MR6_CUBE_MOVE_THRESHOLD_M)
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

        analyzable = True
        violated = False
        reasons: List[str] = []

        progress_ratio = _progress_ratio_to_goal(mr)
        has_contact, contact_source = _has_contact(mr, contact_distance_m)
        cube_moved, cube_max_displacement_m = _cube_moved(mr, cube_move_threshold_m)

        if progress_ratio is None:
            analyzable = False
            reasons.append("cannot_compute_progress_ratio")

        air_push_detected = (not has_contact) and (cube_moved is False)

        if analyzable:
            if air_push_detected and progress_ratio > progress_ratio_threshold:
                violated = True
                reasons.append(
                    "VIOLATION: Fooled By Disturbance "
                    f"(progress_ratio={progress_ratio:.6f} > {progress_ratio_threshold:.6f} without cube contact)"
                )
            else:
                reasons.append("no_air_push_deception_detected")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": getattr(base, "success", None),
                "mr_success": getattr(mr, "success", None),
                "progress_ratio": progress_ratio,
                "progress_ratio_threshold": progress_ratio_threshold,
                "has_contact": has_contact,
                "contact_source": contact_source,
                "cube_moved": cube_moved,
                "cube_max_displacement_m": cube_max_displacement_m,
                "cube_move_threshold_m": cube_move_threshold_m,
                "contact_distance_m": contact_distance_m,
                "air_push_detected": air_push_detected,
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
            "contact_distance_m": contact_distance_m,
            "cube_move_threshold_m": cube_move_threshold_m,
            "violation_rule": "no_contact_and_no_cube_motion_and_progress_ratio > threshold",
        },
        "details": details,
    }
