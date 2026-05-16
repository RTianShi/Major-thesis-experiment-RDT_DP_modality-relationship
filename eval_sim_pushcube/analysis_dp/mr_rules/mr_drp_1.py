from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


DRP1_CONTACT_TARGET_ERROR_THRESHOLD_M = 0.04
DRP1_CONTACT_MOVE_THRESH_M = 0.002


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


def _first_valid_xyz(seq: Any) -> Optional[np.ndarray]:
    for item in seq or []:
        arr = _xyz(item)
        if arr is not None:
            return arr
    return None


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


def _contact_and_cube_target(record: Any, move_thresh_m: float) -> Dict[str, Any]:
    eef_points = _trajectory_points(record, "eef_path")
    cube_target = _first_valid_xyz(getattr(record, "src_cube_pos", None) or getattr(record, "cube_pos", None))
    contact_idx, contact_idx_source = _contact_index_from_cube_motion(record, move_thresh_m)
    contact_point = _xyz(eef_points[contact_idx]) if contact_idx is not None and 0 <= contact_idx < len(eef_points) else None
    return {
        "traj_len": len(eef_points),
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "contact_point": contact_point,
        "cube_target": cube_target,
    }


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


@register_mr_rule("MR-DRP1")
@register_mr_rule("MR-DRP-1")
@register_mr_rule("mr_drp_1")
def analyze_mr_drp1_extreme_diagonal_perturbation(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    contact_target_error_threshold_m = float(
        kwargs.get("contact_target_error_threshold_m", DRP1_CONTACT_TARGET_ERROR_THRESHOLD_M)
    )
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", DRP1_CONTACT_MOVE_THRESH_M))

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
        binfo = _contact_and_cube_target(base, contact_move_thresh_m)
        minfo = _contact_and_cube_target(mr, contact_move_thresh_m)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not bool(base_success):
            analyzable = False
            reasons.append("base_not_successful")
        if minfo["cube_target"] is None:
            analyzable = False
            reasons.append("missing_followup_cube_target")
        if minfo["contact_point"] is None:
            analyzable = False
            reasons.append("missing_followup_contact_point")

        base_contact_error_m = _distance(binfo["contact_point"], binfo["cube_target"])
        followup_contact_error_m = _distance(minfo["contact_point"], minfo["cube_target"])

        if analyzable and followup_contact_error_m is not None:
            if followup_contact_error_m > contact_target_error_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Target Missing "
                    f"(contact_error={followup_contact_error_m:.6f}m > "
                    f"{contact_target_error_threshold_m:.6f}m)"
                )
            else:
                reasons.append("contact_point_tracks_shifted_cube")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": base_success,
                "mr_success": mr_success,
                "base_contact_frame_index": binfo["contact_frame_index"],
                "mr_contact_frame_index": minfo["contact_frame_index"],
                "base_contact_frame_index_source": binfo["contact_frame_index_source"],
                "mr_contact_frame_index_source": minfo["contact_frame_index_source"],
                "base_contact_point": None if binfo["contact_point"] is None else binfo["contact_point"].tolist(),
                "mr_contact_point": None if minfo["contact_point"] is None else minfo["contact_point"].tolist(),
                "base_cube_target": None if binfo["cube_target"] is None else binfo["cube_target"].tolist(),
                "mr_cube_target": None if minfo["cube_target"] is None else minfo["cube_target"].tolist(),
                "base_contact_error_m": base_contact_error_m,
                "mr_contact_error_m": followup_contact_error_m,
                "contact_target_error_threshold_m": contact_target_error_threshold_m,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-DRP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "contact_target_error_threshold_m": contact_target_error_threshold_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "invariance_proxy": "followup_contact_point_alignment_to_shifted_cube",
            "attack_type": "extreme_diagonal_cube_relocation",
        },
        "details": details,
    }
