from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


SCDP3_JITTER_RATIO_THRESHOLD = 3.0
SCDP3_PATH_LENGTH_RATIO_THRESHOLD = 1.3
SCDP3_GRASP_DRIFT_TOL_M = 0.03


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


def _point_l2(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _path_length(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _jitter_amplitude(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 3:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    velocities = np.diff(arr[:, :3], axis=0)
    if velocities.shape[0] < 2:
        return None
    accel_like = np.diff(velocities, axis=0)
    if accel_like.shape[0] == 0:
        return None
    return float(np.linalg.norm(accel_like, axis=1).mean())


def _grasp_point(record: Any, traj: List[List[float]]) -> Dict[str, Any]:
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    grasp_point = traj[grasp_idx] if grasp_idx is not None and 0 <= grasp_idx < len(traj) else None
    return {
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
        "grasp_point": grasp_point,
    }


def _analyze_scdp3_camera_viewpoint_jitter(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    jitter_ratio_threshold: float,
    path_length_ratio_threshold: float,
    grasp_drift_tol_m: float,
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

        base_success = getattr(base, "success", None)
        mr_success = getattr(mr, "success", None)

        base_traj = _trajectory_points(base)
        mr_traj = _trajectory_points(mr)
        base_grasp = _grasp_point(base, base_traj)
        mr_grasp = _grasp_point(mr, mr_traj)

        analyzable = True
        violated = False
        reasons = []

        if not base_traj:
            analyzable = False
            reasons.append("missing_base_eef_trajectory")
        if not mr_traj:
            analyzable = False
            reasons.append("missing_mr_eef_trajectory")

        base_jitter = None
        mr_jitter = None
        base_path_len = None
        mr_path_len = None
        jitter_ratio = None
        path_len_ratio = None
        grasp_drift_m = None

        if analyzable:
            base_jitter = _jitter_amplitude(base_traj)
            mr_jitter = _jitter_amplitude(mr_traj)
            base_path_len = _path_length(base_traj)
            mr_path_len = _path_length(mr_traj)

            if base_jitter is None:
                analyzable = False
                reasons.append("missing_base_jitter_metric")
            if mr_jitter is None:
                analyzable = False
                reasons.append("missing_mr_jitter_metric")
            if base_path_len is None or base_path_len <= 1e-12:
                analyzable = False
                reasons.append("missing_base_path_length")
            if mr_path_len is None:
                analyzable = False
                reasons.append("missing_mr_path_length")
            if base_grasp["grasp_point"] is None:
                analyzable = False
                reasons.append("missing_base_grasp_point")
            if mr_grasp["grasp_point"] is None:
                analyzable = False
                reasons.append("missing_mr_grasp_point")

        if analyzable:
            jitter_ratio = float(mr_jitter / max(base_jitter, 1e-12))
            path_len_ratio = float(mr_path_len / base_path_len)
            grasp_drift_m = _point_l2(base_grasp["grasp_point"], mr_grasp["grasp_point"])

            if jitter_ratio > jitter_ratio_threshold:
                violated = True
                reasons.append("VIOLATION: Violent command twitching synchronized with camera jitter")
            if path_len_ratio > path_length_ratio_threshold:
                violated = True
                reasons.append("VIOLATION: Excessive path length (ratio > 1.3) due to jittery commands")
            if grasp_drift_m is not None and grasp_drift_m > grasp_drift_tol_m:
                violated = True
                reasons.append("VIOLATION: Jitter-induced spatial drift at critical grasp point")
            if not violated:
                reasons.append("robust_invariance_pass")

        if not analyzable:
            unavailable_count += 1
            violated = False

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "base_success": base_success,
                "mr_success": mr_success,
                "base_grasp_frame_index": base_grasp["grasp_frame_index"],
                "mr_grasp_frame_index": mr_grasp["grasp_frame_index"],
                "base_grasp_frame_index_source": base_grasp["grasp_frame_index_source"],
                "mr_grasp_frame_index_source": mr_grasp["grasp_frame_index_source"],
                "base_point_count": len(base_traj),
                "mr_point_count": len(mr_traj),
                "base_jitter_amplitude": base_jitter,
                "mr_jitter_amplitude": mr_jitter,
                "jitter_ratio": jitter_ratio,
                "base_path_len_m": base_path_len,
                "mr_path_len_m": mr_path_len,
                "path_len_ratio": path_len_ratio,
                "base_grasp_point": base_grasp["grasp_point"],
                "mr_grasp_point": mr_grasp["grasp_point"],
                "grasp_point_drift_m": grasp_drift_m,
                "jitter_ratio_threshold": jitter_ratio_threshold,
                "path_length_ratio_threshold": path_length_ratio_threshold,
                "grasp_drift_tol_m": grasp_drift_tol_m,
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
            "jitter_ratio_threshold": jitter_ratio_threshold,
            "path_length_ratio_threshold": path_length_ratio_threshold,
            "grasp_drift_tol_m": grasp_drift_tol_m,
            "metric": "eef_jitter_path_and_grasp_stability",
            "violation_rule": "jitter_ratio > threshold OR path_len_ratio > threshold OR grasp_drift > tol",
            "invariance_expectation": "camera_view_jitter_should_be_filtered_by_multimodal_stability",
        },
        "details": details,
    }


@register_mr_rule("MR-SCDP3")
@register_mr_rule("MR-SCDP-3")
@register_mr_rule("mr_scdp_3")
def analyze_mr_scdp3_camera_viewpoint_jitter(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_scdp3_camera_viewpoint_jitter(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-SCDP3-CAMERA-VIEWPOINT-JITTER"),
        jitter_ratio_threshold=float(kwargs.get("jitter_ratio_threshold", SCDP3_JITTER_RATIO_THRESHOLD)),
        path_length_ratio_threshold=float(
            kwargs.get("path_length_ratio_threshold", SCDP3_PATH_LENGTH_RATIO_THRESHOLD)
        ),
        grasp_drift_tol_m=float(kwargs.get("grasp_drift_tol_m", SCDP3_GRASP_DRIFT_TOL_M)),
    )
