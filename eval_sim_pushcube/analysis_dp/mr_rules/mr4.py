from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


MR4_POSITION_TOL_M = 0.025
MR4_CONTACT_MOVE_THRESH_M = 0.002


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = getattr(record, "eef_path", None) or []
    points: List[List[float]] = []
    for point in traj:
        try:
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


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
    items = seq or []
    for item in items:
        arr = _xyz(item)
        if arr is not None:
            return arr
    return None


def _cube_points(record: Any) -> List[np.ndarray]:
    seq = getattr(record, "cube_pos", None) or getattr(record, "src_cube_pos", None) or []
    points: List[np.ndarray] = []
    for point in seq:
        arr = _xyz(point)
        if arr is not None:
            points.append(arr)
    return points


def _contact_index_with_source(record: Any, move_thresh_m: float) -> Tuple[Optional[int], str]:
    cube_points = _cube_points(record)
    if len(cube_points) < 2:
        return None, "missing_cube_motion"

    start = cube_points[0]
    for idx, point in enumerate(cube_points[1:], start=1):
        if float(np.linalg.norm(point[:3] - start[:3])) > move_thresh_m:
            return idx, "cube_motion_onset"
    return None, "cube_never_moved"


def _contact_point(record: Any, move_thresh_m: float) -> Dict[str, Any]:
    traj = _trajectory_points(record)
    contact_idx, contact_idx_source = _contact_index_with_source(record, move_thresh_m)
    contact_point = _xyz(traj[contact_idx]) if contact_idx is not None and 0 <= contact_idx < len(traj) else None
    return {
        "traj_len": len(traj),
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "contact_point": contact_point,
    }


def _expected_shift(base: Any, mr: Any) -> Optional[np.ndarray]:
    base_src = _first_valid_xyz(getattr(base, "src_cube_pos", None) or getattr(base, "cube_pos", None))
    mr_src = _first_valid_xyz(getattr(mr, "src_cube_pos", None) or getattr(mr, "cube_pos", None))
    if base_src is not None and mr_src is not None:
        return mr_src - base_src

    base_dst = _first_valid_xyz(getattr(base, "dst_cube_pos", None) or getattr(base, "target_pos", None))
    mr_dst = _first_valid_xyz(getattr(mr, "dst_cube_pos", None) or getattr(mr, "target_pos", None))
    if base_dst is not None and mr_dst is not None:
        return mr_dst - base_dst
    return None


def _anchor_drift(
    base_point: Optional[np.ndarray],
    mr_point: Optional[np.ndarray],
    expected_shift: Optional[np.ndarray],
) -> Optional[float]:
    if base_point is None or mr_point is None or expected_shift is None:
        return None
    actual_shift = np.asarray(mr_point, dtype=np.float64)[:3] - np.asarray(base_point, dtype=np.float64)[:3]
    return float(np.linalg.norm(actual_shift - np.asarray(expected_shift, dtype=np.float64)[:3]))


@register_mr_rule("MR4")
@register_mr_rule("MR-4")
@register_mr_rule("mr4")
def analyze_mr4_target_object_relocation(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    position_tol_m = float(kwargs.get("position_tol_m", MR4_POSITION_TOL_M))
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", MR4_CONTACT_MOVE_THRESH_M))

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
        binfo = _contact_point(base, contact_move_thresh_m)
        minfo = _contact_point(mr, contact_move_thresh_m)
        expected_shift = _expected_shift(base, mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if binfo["contact_point"] is None:
            analyzable = False
            reasons.append("missing_base_contact_point")
        if minfo["contact_point"] is None:
            analyzable = False
            reasons.append("missing_mr_contact_point")
        if expected_shift is None:
            analyzable = False
            reasons.append("missing_expected_translation_shift")

        contact_drift_m = None
        if analyzable:
            contact_drift_m = _anchor_drift(
                binfo["contact_point"],
                minfo["contact_point"],
                expected_shift,
            )
            if contact_drift_m is None:
                analyzable = False
                reasons.append("cannot_compute_contact_anchor_drift")
            elif contact_drift_m > position_tol_m:
                violated = True
                reasons.append(
                    "VIOLATION: Spatial Drift At Initial Contact "
                    f"(drift={contact_drift_m:.6f}m > {position_tol_m:.6f}m)"
                )
            else:
                reasons.append("contact_anchor_translates_with_object")

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
                "base_contact_frame_index": binfo["contact_frame_index"],
                "mr_contact_frame_index": minfo["contact_frame_index"],
                "base_contact_frame_index_source": binfo["contact_frame_index_source"],
                "mr_contact_frame_index_source": minfo["contact_frame_index_source"],
                "base_contact_point": None if binfo["contact_point"] is None else binfo["contact_point"].tolist(),
                "mr_contact_point": None if minfo["contact_point"] is None else minfo["contact_point"].tolist(),
                "expected_shift": None if expected_shift is None else expected_shift.tolist(),
                "contact_anchor_drift_m": contact_drift_m,
                "position_tol_m": position_tol_m,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR4",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "position_tol_m": position_tol_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "violation_rule": "||(Pcontact_f - Pcontact_s) - Δd|| > position_tol_m",
        },
        "details": details,
    }
