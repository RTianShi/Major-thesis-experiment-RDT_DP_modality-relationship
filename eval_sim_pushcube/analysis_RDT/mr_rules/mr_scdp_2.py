from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SCDP2_PATH_LENGTH_RATIO_THRESHOLD = 1.15
SCDP2_CONTACT_DRIFT_THRESHOLD_M = 0.03
SCDP2_FINAL_DRIFT_THRESHOLD_M = 0.03
SCDP2_CONTACT_MOVE_THRESH_M = 0.002


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


def _trajectory_points(record: Any) -> List[np.ndarray]:
    traj = getattr(record, "eef_path", None) or []
    points: List[np.ndarray] = []
    for point in traj:
        arr = _xyz(point)
        if arr is not None:
            points.append(arr)
    return points


def _cube_points(record: Any) -> List[np.ndarray]:
    seq = getattr(record, "cube_pos", None) or getattr(record, "src_cube_pos", None) or []
    points: List[np.ndarray] = []
    for point in seq:
        arr = _xyz(point)
        if arr is not None:
            points.append(arr)
    return points


def _contact_index_from_cube_motion(record: Any, move_thresh_m: float) -> Tuple[Optional[int], str]:
    cube_points = _cube_points(record)
    if len(cube_points) < 2:
        return None, "missing_cube_motion"

    start = cube_points[0]
    for idx, point in enumerate(cube_points[1:], start=1):
        if float(np.linalg.norm(point - start)) > move_thresh_m:
            return idx, "cube_motion_onset"
    return None, "cube_never_moved"


def _contact_and_final_points(record: Any, move_thresh_m: float) -> Dict[str, Any]:
    traj = _trajectory_points(record)
    contact_idx, contact_idx_source = _contact_index_from_cube_motion(record, move_thresh_m)
    contact_point = traj[contact_idx] if contact_idx is not None and 0 <= contact_idx < len(traj) else None
    final_point = traj[-1] if traj else None
    return {
        "traj_len": len(traj),
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "contact_point": contact_point,
        "final_point": final_point,
    }


def _path_length(points: List[np.ndarray]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


@register_mr_rule("MR-SCDP2")
@register_mr_rule("MR-SCDP-2")
@register_mr_rule("mr_scdp_2")
def analyze_mr_scdp2_linguistic_syntax_debunking(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    path_length_ratio_threshold = float(
        kwargs.get("path_length_ratio_threshold", SCDP2_PATH_LENGTH_RATIO_THRESHOLD)
    )
    contact_drift_threshold_m = float(
        kwargs.get("contact_drift_threshold_m", SCDP2_CONTACT_DRIFT_THRESHOLD_M)
    )
    final_drift_threshold_m = float(
        kwargs.get("final_drift_threshold_m", SCDP2_FINAL_DRIFT_THRESHOLD_M)
    )
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", SCDP2_CONTACT_MOVE_THRESH_M))

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
        binfo = _contact_and_final_points(base, contact_move_thresh_m)
        minfo = _contact_and_final_points(mr, contact_move_thresh_m)
        base_points = _trajectory_points(base)
        mr_points = _trajectory_points(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not base_points:
            analyzable = False
            reasons.append("missing_base_eef_path")
        if not mr_points:
            analyzable = False
            reasons.append("missing_mr_eef_path")
        if binfo["contact_point"] is None:
            analyzable = False
            reasons.append("missing_base_contact_point")
        if minfo["contact_point"] is None:
            analyzable = False
            reasons.append("missing_mr_contact_point")
        if binfo["final_point"] is None:
            analyzable = False
            reasons.append("missing_base_final_point")
        if minfo["final_point"] is None:
            analyzable = False
            reasons.append("missing_mr_final_point")

        path_length_ratio = None
        contact_drift_m = None
        final_drift_m = None

        if analyzable:
            base_path_len = _path_length(base_points)
            mr_path_len = _path_length(mr_points)
            if base_path_len is None or mr_path_len is None:
                analyzable = False
                reasons.append("cannot_compute_path_length")
            else:
                path_length_ratio = float(mr_path_len / max(base_path_len, 1e-12))
                if path_length_ratio > path_length_ratio_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Path Length Inflation "
                        f"(ratio={path_length_ratio:.6f} > {path_length_ratio_threshold:.6f})"
                    )

            contact_drift_m = _distance(minfo["contact_point"], binfo["contact_point"])
            if contact_drift_m is None:
                analyzable = False
                reasons.append("cannot_compute_contact_drift")
            elif contact_drift_m > contact_drift_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Spatial Drift At Contact "
                    f"(drift={contact_drift_m:.6f}m > {contact_drift_threshold_m:.6f}m)"
                )

            final_drift_m = _distance(minfo["final_point"], binfo["final_point"])
            if final_drift_m is None:
                analyzable = False
                reasons.append("cannot_compute_final_drift")
            elif final_drift_m > final_drift_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Spatial Drift At Final Point "
                    f"(drift={final_drift_m:.6f}m > {final_drift_threshold_m:.6f}m)"
                )

            if not violated:
                reasons.append("syntax_variation_preserves_trajectory")

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
                "base_final_point": None if binfo["final_point"] is None else binfo["final_point"].tolist(),
                "mr_final_point": None if minfo["final_point"] is None else minfo["final_point"].tolist(),
                "path_length_ratio": path_length_ratio,
                "contact_drift_m": contact_drift_m,
                "final_drift_m": final_drift_m,
                "path_length_ratio_threshold": path_length_ratio_threshold,
                "contact_drift_threshold_m": contact_drift_threshold_m,
                "final_drift_threshold_m": final_drift_threshold_m,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SCDP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "path_length_ratio_threshold": path_length_ratio_threshold,
            "contact_drift_threshold_m": contact_drift_threshold_m,
            "final_drift_threshold_m": final_drift_threshold_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "violation_rule": (
                "path_length_ratio > threshold OR contact_drift > threshold OR final_drift > threshold"
            ),
        },
        "details": details,
    }
