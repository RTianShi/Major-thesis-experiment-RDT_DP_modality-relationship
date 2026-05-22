from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


MR5_TARGET_ANGLE_THRESHOLD_DEG = 45.0
MR5_CONTACT_MOVE_THRESH_M = 0.002


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


def _contact_index_from_cube_motion(record: Any, move_thresh_m: float) -> Tuple[Optional[int], str]:
    cube_points = _trajectory_points(record, "cube_pos") or _trajectory_points(record, "src_cube_pos")
    if len(cube_points) < 2:
        return None, "missing_cube_motion"

    start = cube_points[0]
    for idx, point in enumerate(cube_points[1:], start=1):
        if float(np.linalg.norm(point[:3] - start[:3])) > move_thresh_m:
            return idx, "cube_motion_onset"
    return None, "cube_never_moved"


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vec / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if a_norm is None or b_norm is None:
        return None
    dot = float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _goal_point(record: Any) -> Optional[np.ndarray]:
    goal_points = (
        _trajectory_points(record, "goal_pos")
        or _trajectory_points(record, "target_pos")
        or _trajectory_points(record, "dst_cube_pos")
    )
    if goal_points:
        return goal_points[0]
    return None


@register_mr_rule("MR5")
@register_mr_rule("MR-5")
@register_mr_rule("mr5")
def analyze_mr5_instruction_specialization(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    target_angle_threshold_deg = float(
        kwargs.get("target_angle_threshold_deg", MR5_TARGET_ANGLE_THRESHOLD_DEG)
    )
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", MR5_CONTACT_MOVE_THRESH_M))

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        base_success = getattr(base, "success", None)
        mr_success = getattr(mr, "success", None)
        mr_eef = _trajectory_points(mr, "eef_path")
        contact_idx, contact_idx_source = _contact_index_from_cube_motion(mr, contact_move_thresh_m)
        goal_point = _goal_point(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not mr_eef:
            analyzable = False
            reasons.append("missing_mr_eef_path")
        if contact_idx is None:
            analyzable = False
            reasons.append("missing_mr_contact_frame_index")
        elif contact_idx >= len(mr_eef) - 1:
            analyzable = False
            reasons.append("contact_frame_too_late_for_push_vector")
        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")

        push_vector = None
        target_vector = None
        angle_to_target_deg = None

        if analyzable:
            contact_point = mr_eef[contact_idx]
            next_point = mr_eef[contact_idx + 1]
            push_vector = np.asarray(next_point, dtype=np.float64) - np.asarray(contact_point, dtype=np.float64)
            target_vector = np.asarray(goal_point, dtype=np.float64) - np.asarray(contact_point, dtype=np.float64)
            angle_to_target_deg = _angle_deg(push_vector, target_vector)

            if angle_to_target_deg is None:
                analyzable = False
                reasons.append("cannot_compute_angle_to_target")
            elif angle_to_target_deg >= target_angle_threshold_deg:
                violated = True
                reasons.append(
                    "VIOLATION: Directional Constraint Failure "
                    f"(angle_to_target={angle_to_target_deg:.6f} >= {target_angle_threshold_deg:.6f} deg)"
                )
            else:
                reasons.append("push_direction_respects_specialized_instruction")

        if not analyzable:
            unavailable_count += 1
            violated = False
        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": base_success,
                "mr_success": mr_success,
                "mr_contact_frame_index": contact_idx,
                "mr_contact_frame_index_source": contact_idx_source,
                "goal_point": None if goal_point is None else goal_point.tolist(),
                "push_vector": None if push_vector is None else push_vector.tolist(),
                "target_vector": None if target_vector is None else target_vector.tolist(),
                "angle_to_target_deg": angle_to_target_deg,
                "target_angle_threshold_deg": target_angle_threshold_deg,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR5",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "target_angle_threshold_deg": target_angle_threshold_deg,
            "contact_move_thresh_m": contact_move_thresh_m,
            "violation_rule": "angle_to_target >= target_angle_threshold_deg",
        },
        "details": details,
    }
