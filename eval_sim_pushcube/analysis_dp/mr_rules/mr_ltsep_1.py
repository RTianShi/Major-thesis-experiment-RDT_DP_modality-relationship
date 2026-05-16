from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


LTSEP1_CUBE_PROGRESS_THRESHOLD_M = 0.02
LTSEP1_CONTACT_DISTANCE_M = 0.035
LTSEP1_PHANTOM_DISTANCE_SCALE = 2.0


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
    traj = getattr(record, attr_name, None) or []
    points: List[np.ndarray] = []
    for item in traj:
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


def _phantom_trigger_index(
    record: Any,
    *,
    contact_distance_m: float,
    phantom_max_distance_m: float,
) -> Optional[int]:
    distances = _safe_float_list(getattr(record, "tcp_to_obj_distance", None))
    contacts = _safe_bool_list(getattr(record, "is_contact", None))
    length = min(len(distances), len(contacts))
    for idx in range(length):
        dist = distances[idx]
        is_contact = contacts[idx]
        if dist is None or is_contact is None:
            continue
        if is_contact:
            continue
        if dist <= contact_distance_m:
            continue
        if dist <= phantom_max_distance_m:
            return idx
    return None


def _cube_progress_to_goal(
    cube_points: List[np.ndarray],
    target_points: List[np.ndarray],
    trigger_index: int,
) -> Optional[Dict[str, Any]]:
    if trigger_index < 0:
        return None
    common_len = min(len(cube_points), len(target_points))
    if common_len == 0 or trigger_index >= common_len:
        return None

    cube_at_attack = cube_points[trigger_index]
    cube_at_end = cube_points[common_len - 1]
    target_at_attack = target_points[trigger_index]

    goal_vec = target_at_attack - cube_at_attack
    goal_norm = float(np.linalg.norm(goal_vec))
    if goal_norm <= 1e-12:
        return None

    goal_dir = goal_vec / goal_norm
    cube_delta = cube_at_end - cube_at_attack
    cube_move = float(np.dot(cube_delta, goal_dir))

    return {
        "cube_pos_at_attack": [float(v) for v in cube_at_attack.tolist()],
        "cube_pos_at_end": [float(v) for v in cube_at_end.tolist()],
        "target_pos_at_attack": [float(v) for v in target_at_attack.tolist()],
        "goal_direction": [float(v) for v in goal_dir.tolist()],
        "cube_move_m": cube_move,
    }


@register_mr_rule("MR-LTSEP1")
@register_mr_rule("MR-LTSEP-1")
@register_mr_rule("mr_ltsep_1")
def analyze_mr_ltsep1_vision_proprio_phantom_tear(
     base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    cube_progress_threshold_m = float(
        kwargs.get("cube_progress_threshold_m", LTSEP1_CUBE_PROGRESS_THRESHOLD_M)
    )
    contact_distance_m = float(kwargs.get("contact_distance_m", LTSEP1_CONTACT_DISTANCE_M))
    phantom_max_distance_m = float(
        kwargs.get("phantom_max_distance_m", contact_distance_m * LTSEP1_PHANTOM_DISTANCE_SCALE)
    )

    mmap = {_paired_key(r): r for r in (mr_records or [])}
    bmap = {_paired_key(r): r for r in (base_records or [])}
    # Only analyze paired episodes present in both sets.
    keys = sorted(set(mmap.keys()).intersection(set(bmap.keys())))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        record = mmap[key]
        base = bmap.get(key)
        base_success = getattr(base, "success", None)
        cube_points = _trajectory_points(record, "cube_pos")
        target_points = _trajectory_points(record, "target_pos")
        trigger_index = _phantom_trigger_index(
            record,
            contact_distance_m=contact_distance_m,
            phantom_max_distance_m=phantom_max_distance_m,
        )

        analyzable = True
        violated = False
        reasons: List[str] = []

        if base is None:
            analyzable = False
            reasons.append("missing_base_record")

        if not cube_points:
            analyzable = False
            reasons.append("missing_cube_pos")
        if not target_points:
            analyzable = False
            reasons.append("missing_target_pos")
        if trigger_index is None:
            analyzable = False
            reasons.append("missing_phantom_trigger_window")

        progress_info = None
        if analyzable:
            progress_info = _cube_progress_to_goal(cube_points, target_points, trigger_index)
            if progress_info is None:
                analyzable = False
                reasons.append("cannot_compute_cube_progress")

        if analyzable and progress_info is not None:
            cube_move = float(progress_info["cube_move_m"])
            if cube_move > cube_progress_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Cube Progress Violation "
                    f"(Cube_Move={cube_move:.6f}m > {cube_progress_threshold_m:.6f}m)"
                )
            else:
                reasons.append("safe_halt_triggered")
        else:
            cube_move = None

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                 "base_success": base_success,
                "success": getattr(record, "success", None),
                "phantom_trigger_index": trigger_index,
                "cube_pos_at_attack": None if progress_info is None else progress_info["cube_pos_at_attack"],
                "cube_pos_at_end": None if progress_info is None else progress_info["cube_pos_at_end"],
                "target_pos_at_attack": None if progress_info is None else progress_info["target_pos_at_attack"],
                "goal_direction": None if progress_info is None else progress_info["goal_direction"],
                "cube_move_m": cube_move,
                "cube_progress_threshold_m": cube_progress_threshold_m,
                "contact_distance_m": contact_distance_m,
                "phantom_max_distance_m": phantom_max_distance_m,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-LTSEP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "cube_progress_threshold_m": cube_progress_threshold_m,
            "contact_distance_m": contact_distance_m,
            "phantom_max_distance_m": phantom_max_distance_m,
            "invariance_proxy": "cube_progress_along_goal_after_phantom_collision",
        },
        "details": details,
    }
