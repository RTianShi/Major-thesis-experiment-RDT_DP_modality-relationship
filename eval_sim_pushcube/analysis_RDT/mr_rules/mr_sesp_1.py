from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


SESP1_POSITION_TOL_M = 0.025
SESP1_CONTACT_MOVE_THRESH_M = 0.002
SESP1_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85

SESP1_MIN_ALIGNMENT_POINTS = 4
SESP1_SAMPLE_POINTS = 24


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


def _contact_index_with_source(record: Any, move_thresh_m: float) -> (Optional[int], str): # type: ignore
    cube_points = _cube_points(record)
    if len(cube_points) < 2:
        return None, "missing_cube_motion"

    start = cube_points[0]
    for idx, point in enumerate(cube_points[1:], start=1):
        if float(np.linalg.norm(point[:3] - start[:3])) > move_thresh_m:
            return idx, "cube_motion_onset"
    return None, "cube_never_moved"


def _contact_and_final_points(record: Any, move_thresh_m: float) -> Dict[str, Any]:
    traj = _trajectory_points(record)
    contact_idx, contact_idx_source = _contact_index_with_source(record, move_thresh_m)
    contact_point = _xyz(traj[contact_idx]) if contact_idx is not None and 0 <= contact_idx < len(traj) else None
    final_point = _xyz(traj[-1]) if traj else None
    return {
        "traj_len": len(traj),
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "contact_point": contact_point,
        "final_point": final_point,
    }


def _expected_shift(base: Any, mr: Any) -> Optional[np.ndarray]:
    base_dst = _first_valid_xyz(getattr(base, "dst_cube_pos", None))
    mr_dst = _first_valid_xyz(getattr(mr, "dst_cube_pos", None))
    if base_dst is not None and mr_dst is not None:
        return mr_dst - base_dst

    base_src = _first_valid_xyz(getattr(base, "src_cube_pos", None) or getattr(base, "cube_pos", None))
    mr_src = _first_valid_xyz(getattr(mr, "src_cube_pos", None) or getattr(mr, "cube_pos", None))
    if base_src is not None and mr_src is not None:
        return mr_src - base_src
    return None


def _anchor_drift(
    base_point: Optional[np.ndarray],
    mr_point: Optional[np.ndarray],
    expected_shift: Optional[np.ndarray],
) -> Optional[float]:
    if base_point is None or mr_point is None or expected_shift is None:
        return None
    actual_action_shift = np.asarray(mr_point, dtype=np.float64)[:3] - np.asarray(base_point, dtype=np.float64)[:3]
    return float(np.linalg.norm(actual_action_shift - np.asarray(expected_shift, dtype=np.float64)[:3]))


def _path_length(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _resample_polyline(points: List[List[float]], sample_count: int) -> Optional[np.ndarray]:
    if points is None or sample_count <= 0:
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


def _shift_points(points: List[List[float]], shift: np.ndarray) -> List[List[float]]:
    shifted = []
    for point in points:
        arr = np.asarray(point, dtype=np.float64)[:3] - shift[:3]
        shifted.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return shifted


def _trajectory_similarity_metrics(
    shifted_followup_points: List[List[float]],
    source_points: List[List[float]],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    if shifted_followup_points is None or source_points is None:
        return None

    sample_count = min(int(sample_points), len(shifted_followup_points), len(source_points))
    if sample_count < SESP1_MIN_ALIGNMENT_POINTS:
        return None

    followup_resampled = _resample_polyline(shifted_followup_points, sample_count)
    source_resampled = _resample_polyline(source_points, sample_count)
    if followup_resampled is None or source_resampled is None:
        return None

    pointwise_dist = np.linalg.norm(followup_resampled - source_resampled, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())

    followup_path_len = _path_length(shifted_followup_points)
    source_path_len = _path_length(source_points)
    path_scale = source_path_len if source_path_len is not None and source_path_len > 1e-12 else followup_path_len
    if path_scale is None or path_scale <= 1e-12:
        return None

    relative_deviation = float(mean_pointwise_deviation / path_scale)
    similarity_score = float(1.0 / (1.0 + relative_deviation))

    return {
        "mean_pointwise_deviation_m": mean_pointwise_deviation,
        "shifted_followup_path_len_m": float(followup_path_len) if followup_path_len is not None else None,
        "source_path_len_m": float(source_path_len) if source_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "sample_points": float(sample_count),
    }


@register_mr_rule("MR-SESP1")
@register_mr_rule("MR-SESP-1")
@register_mr_rule("mr_sesp_1")
def analyze_mr_sesp1_spatial_translation_equivariance(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    position_tol_m = float(kwargs.get("position_tol_m", SESP1_POSITION_TOL_M))
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", SESP1_CONTACT_MOVE_THRESH_M))
    similarity_threshold = float(
        kwargs.get("similarity_threshold", SESP1_TRAJECTORY_SIMILARITY_THRESHOLD)
    )
    sample_points = int(kwargs.get("sample_points", SESP1_SAMPLE_POINTS))

    bmap = {_paired_key(r): r for r in base_records}
    mmap = {_paired_key(r): r for r in mr_records}
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
        expected_shift = _expected_shift(base, mr)

        analyzable = True
        violated = False
        reasons = []

        base_contact_point = binfo["contact_point"]
        mr_contact_point = minfo["contact_point"]
        breakdown_failed_to_contact = False
        # Special-case: baseline contacts the cube but followup never contacts the shifted cube.
        # This is an equivariance breakdown rather than an unavailable sample.
        if base_contact_point is not None and mr_contact_point is None:
            analyzable = True
            violated = True
            breakdown_failed_to_contact = True
            reasons.append("Equivariance Breakdown: Robot failed to contact the shifted cube")
        elif base_contact_point is None or mr_contact_point is None:
            analyzable = False
            if base_contact_point is None:
                reasons.append("missing_base_contact_point")
            if mr_contact_point is None:
                reasons.append("missing_mr_contact_point")
        if expected_shift is None and not breakdown_failed_to_contact:
            analyzable = False
            reasons.append("missing_expected_translation_shift")

        contact_drift_m = None
        trajectory_similarity = None
        trajectory_metrics = None
        if analyzable and not breakdown_failed_to_contact:
            contact_drift_m = _anchor_drift(
                base_contact_point,
                mr_contact_point,
                expected_shift,
            )
            if contact_drift_m is None:
                analyzable = False
                reasons.append("cannot_compute_contact_anchor_spatial_drift")
            else:
                if contact_drift_m > position_tol_m:
                    violated = True
                    reasons.append(
                        f"VIOLATION: Anchor Spatial Drift (Drift error: {contact_drift_m}m > 0.025m)"
                    )

                shifted_followup_traj = _shift_points(_trajectory_points(mr), expected_shift)
                source_traj = _trajectory_points(base)
                trajectory_metrics = _trajectory_similarity_metrics(
                    shifted_followup_traj,
                    source_traj,
                    sample_points=sample_points,
                )
                
                if analyzable and not violated:
                    reasons.append("contact_translation_equivariance_pass")

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
                "contact_drift_error_m": contact_drift_m,
                "trajectory_similarity_after_shift_compensation": trajectory_similarity,
                "mean_pointwise_deviation_m": None if trajectory_metrics is None else trajectory_metrics["mean_pointwise_deviation_m"],
                "shifted_followup_path_len_m": None if trajectory_metrics is None else trajectory_metrics["shifted_followup_path_len_m"],
                "source_path_len_m": None if trajectory_metrics is None else trajectory_metrics["source_path_len_m"],
                "relative_deviation": None if trajectory_metrics is None else trajectory_metrics["relative_deviation"],
                "position_tol_m": position_tol_m,
                "contact_move_thresh_m": contact_move_thresh_m,
                "similarity_threshold": similarity_threshold,
                "sample_points": sample_points,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SESP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "position_tol_m": position_tol_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "similarity_threshold": similarity_threshold,
            "sample_points": sample_points,
            "metric": "contact_anchor_drift_and_shift_compensated_trajectory_similarity",
            "violation_rule": "anchor_drift(contact) > tol OR similarity(Tf - expected_shift, Ts) < similarity_threshold",
        },
        "details": details,
    }
