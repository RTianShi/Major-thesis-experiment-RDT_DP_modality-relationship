from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


DRP2_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85
DRP2_CONTACT_MOVE_THRESH_M = 0.002
DRP2_MIN_ALIGNMENT_POINTS = 4
DRP2_SAMPLE_POINTS = 24
DRP2_MIN_START_POSE_DELTA_M = 0.03
DRP2_CONTACT_POINT_DRIFT_THRESHOLD_M = 0.025


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


def _approach_points(record: Any, move_thresh_m: float) -> Dict[str, Any]:
    eef_points = _trajectory_points(record, "eef_path")
    contact_idx, contact_idx_source = _contact_index_from_cube_motion(record, move_thresh_m)
    if contact_idx is None:
        approach = list(eef_points)
    else:
        upper = min(len(eef_points), int(contact_idx) + 1)
        approach = eef_points[:upper]
    return {
        "traj_len": len(eef_points),
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "approach_points": approach,
        "start_point": None if not eef_points else eef_points[0],
        "contact_point": _xyz(eef_points[contact_idx]) if contact_idx is not None and 0 <= contact_idx < len(eef_points) else None,
    }


def _path_length(points: List[np.ndarray]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _resample_polyline(points: List[np.ndarray], sample_count: int) -> Optional[np.ndarray]:
    if sample_count <= 0:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    arr = arr[:, :3]
    if arr.shape[0] == 0:
        return None
    if arr.shape[0] == 1:
        return np.repeat(arr[:1], sample_count, axis=0)

    deltas = np.diff(arr, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return np.repeat(arr[:1], sample_count, axis=0)

    targets = np.linspace(0.0, total_length, sample_count)
    resampled = np.empty((sample_count, 3), dtype=np.float64)
    for dim in range(3):
        resampled[:, dim] = np.interp(targets, cumulative, arr[:, dim])
    return resampled


def _trajectory_similarity_metrics(
    source_points: List[np.ndarray],
    followup_points: List[np.ndarray],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    sample_count = min(int(sample_points), len(source_points), len(followup_points))
    if sample_count < DRP2_MIN_ALIGNMENT_POINTS:
        return None

    source_resampled = _resample_polyline(source_points, sample_count)
    followup_resampled = _resample_polyline(followup_points, sample_count)
    if source_resampled is None or followup_resampled is None:
        return None

    pointwise_dist = np.linalg.norm(followup_resampled - source_resampled, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())

    source_path_len = _path_length(source_points)
    followup_path_len = _path_length(followup_points)
    path_scale = source_path_len if source_path_len is not None and source_path_len > 1e-12 else followup_path_len
    if path_scale is None or path_scale <= 1e-12:
        return None

    relative_deviation = float(mean_pointwise_deviation / path_scale)
    similarity_score = float(1.0 / (1.0 + relative_deviation))
    return {
        "mean_pointwise_deviation_m": mean_pointwise_deviation,
        "source_path_len_m": float(source_path_len) if source_path_len is not None else None,
        "followup_path_len_m": float(followup_path_len) if followup_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "sample_points": float(sample_count),
    }


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


@register_mr_rule("MR-DRP2")
@register_mr_rule("MR-DRP-2")
@register_mr_rule("mr_drp_2")
def analyze_mr_drp2_bilateral_extreme_entry(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    mode = "trajectory_only"
    similarity_threshold = float(
        kwargs.get("trajectory_similarity_threshold", DRP2_TRAJECTORY_SIMILARITY_THRESHOLD)
    )
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", DRP2_CONTACT_MOVE_THRESH_M))
    sample_points = int(kwargs.get("sample_points", DRP2_SAMPLE_POINTS))
    min_start_pose_delta_m = float(kwargs.get("min_start_pose_delta_m", DRP2_MIN_START_POSE_DELTA_M))
    contact_point_drift_threshold_m = float(
        kwargs.get("contact_point_drift_threshold_m", DRP2_CONTACT_POINT_DRIFT_THRESHOLD_M)
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

        base_success = getattr(base, "success", None)
        mr_success = getattr(mr, "success", None)
        binfo = _approach_points(base, contact_move_thresh_m)
        minfo = _approach_points(mr, contact_move_thresh_m)

        analyzable = True
        violated = False
        reasons: List[str] = []

        start_pose_delta_m = _distance(binfo["start_point"], minfo["start_point"])
        contact_point_drift_m = _distance(binfo["contact_point"], minfo["contact_point"])
        metrics = None
        trajectory_similarity = None

        if start_pose_delta_m is None:
            analyzable = False
            reasons.append("missing_start_pose_points")
        elif start_pose_delta_m < min_start_pose_delta_m:
            analyzable = False
            reasons.append(
                f"insufficient_start_pose_delta({start_pose_delta_m:.6f}m < {min_start_pose_delta_m:.6f}m)"
            )

        if len(binfo["approach_points"]) < DRP2_MIN_ALIGNMENT_POINTS:
            analyzable = False
            reasons.append("insufficient_base_trajectory_points")
        if len(minfo["approach_points"]) < DRP2_MIN_ALIGNMENT_POINTS:
            analyzable = False
            reasons.append("insufficient_followup_trajectory_points")

        if analyzable and not violated:
            metrics = _trajectory_similarity_metrics(
                binfo["approach_points"],
                minfo["approach_points"],
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("cannot_compute_approach_trajectory_similarity")
            else:
                trajectory_similarity = float(metrics["trajectory_similarity"])
                if trajectory_similarity > similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Trajectory Conflict "
                        f"(trajectory_similarity={trajectory_similarity:.6f} > {similarity_threshold:.6f})"
                    )
                else:
                    reasons.append("trajectory_diverged_under_pose_change")

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
                "base_start_point": None if binfo["start_point"] is None else binfo["start_point"].tolist(),
                "mr_start_point": None if minfo["start_point"] is None else minfo["start_point"].tolist(),
                "start_pose_delta_m": start_pose_delta_m,
                "base_contact_point": None if binfo["contact_point"] is None else binfo["contact_point"].tolist(),
                "mr_contact_point": None if minfo["contact_point"] is None else minfo["contact_point"].tolist(),
                "contact_point_drift_m": contact_point_drift_m,
                "base_approach_len": len(binfo["approach_points"]),
                "mr_approach_len": len(minfo["approach_points"]),
                "trajectory_similarity": trajectory_similarity,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "followup_path_len_m": None if metrics is None else metrics["followup_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "sample_points": None if metrics is None else metrics["sample_points"],
                "trajectory_similarity_threshold": similarity_threshold,
                "contact_point_drift_threshold_m": contact_point_drift_threshold_m,
                "min_start_pose_delta_m": min_start_pose_delta_m,
                "mode": mode,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-DRP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "mode": mode,
            "trajectory_similarity_threshold": similarity_threshold,
            "contact_point_drift_threshold_m": contact_point_drift_threshold_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "sample_points": sample_points,
            "min_start_pose_delta_m": min_start_pose_delta_m,
            "invariance_proxy": "absolute_approach_trajectory_divergence_under_bilateral_start_pose_change",
            "attack_type": "left_biased_vs_right_biased_initial_proprioceptive_pose",
        },
        "details": details,
    }
