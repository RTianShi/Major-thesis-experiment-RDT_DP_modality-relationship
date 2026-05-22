from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SEMP3_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85
SEMP3_CONTACT_MOVE_THRESH_M = 0.002
SEMP3_STOP_MOVE_THRESH_M = 0.001
SEMP3_MIN_PUSH_POINTS = 4
SEMP3_SAMPLE_POINTS = 24
SEMP3_PUSH_POINT_CHANGE_THRESH_M = 0.01


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


def _push_end_index(cube_points: List[np.ndarray], contact_idx: int, stop_move_thresh_m: float) -> int:
    if not cube_points:
        return contact_idx

    last_motion_idx = contact_idx
    prev = cube_points[contact_idx]
    for idx in range(contact_idx + 1, len(cube_points)):
        cur = cube_points[idx]
        if float(np.linalg.norm(cur[:3] - prev[:3])) > stop_move_thresh_m:
            last_motion_idx = idx
        prev = cur
    return last_motion_idx


def _push_segment(record: Any, contact_move_thresh_m: float, stop_move_thresh_m: float) -> Dict[str, Any]:
    eef_points = _trajectory_points(record, "eef_path")
    cube_points = _trajectory_points(record, "cube_pos") or _trajectory_points(record, "src_cube_pos")
    contact_idx, contact_idx_source = _contact_index_from_cube_motion(record, contact_move_thresh_m)

    if contact_idx is None:
        return {
            "contact_frame_index": None,
            "contact_frame_index_source": contact_idx_source,
            "contact_point": None,
            "push_end_index": None,
            "push_points": [],
        }

    if contact_idx >= len(eef_points) or contact_idx >= len(cube_points):
        return {
            "contact_frame_index": contact_idx,
            "contact_frame_index_source": "contact_index_out_of_range",
            "contact_point": None,
            "push_end_index": None,
            "push_points": [],
        }

    push_end_idx = _push_end_index(cube_points, contact_idx, stop_move_thresh_m)
    upper = min(len(eef_points), push_end_idx + 1)
    return {
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "contact_point": eef_points[contact_idx],
        "push_end_index": push_end_idx,
        "push_points": eef_points[contact_idx:upper],
    }


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


def _path_length(points: List[np.ndarray]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _trajectory_similarity_metrics(
    source_points: List[np.ndarray],
    followup_points: List[np.ndarray],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    sample_count = min(int(sample_points), len(source_points), len(followup_points))
    if sample_count < SEMP3_MIN_PUSH_POINTS:
        return None

    source_resampled = _resample_polyline(source_points, sample_count)
    followup_resampled = _resample_polyline(followup_points, sample_count)
    if source_resampled is None or followup_resampled is None:
        return None

    pointwise_delta = followup_resampled - source_resampled
    pointwise_dist = np.linalg.norm(pointwise_delta, axis=1)
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


def _shift_points(points: List[np.ndarray], shift: np.ndarray) -> List[np.ndarray]:
    return [np.asarray(point, dtype=np.float64)[:3] - shift[:3] for point in points]


@register_mr_rule("MR-SEMP3")
@register_mr_rule("MR-SEMP-3")
@register_mr_rule("mr_semp_3")
def analyze_mr_semp3_ssgep_translation_equivariance(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    similarity_threshold = float(kwargs.get("similarity_threshold", SEMP3_TRAJECTORY_SIMILARITY_THRESHOLD))
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", SEMP3_CONTACT_MOVE_THRESH_M))
    stop_move_thresh_m = float(kwargs.get("stop_move_thresh_m", SEMP3_STOP_MOVE_THRESH_M))
    sample_points = int(kwargs.get("sample_points", SEMP3_SAMPLE_POINTS))
    push_point_change_thresh_m = float(
        kwargs.get("push_point_change_thresh_m", SEMP3_PUSH_POINT_CHANGE_THRESH_M)
    )
    expected_translation = kwargs.get("expected_translation", None)
    if expected_translation is not None:
        expected_translation = np.asarray(expected_translation, dtype=np.float64).reshape(-1)
        if expected_translation.size < 3:
            expected_translation = None

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
        binfo = _push_segment(base, contact_move_thresh_m, stop_move_thresh_m)
        minfo = _push_segment(mr, contact_move_thresh_m, stop_move_thresh_m)

        analyzable = True
        violated = False
        reasons: List[str] = []

        base_push_len = len(binfo["push_points"])
        mr_push_len = len(minfo["push_points"])
        use_fallback = base_push_len < SEMP3_MIN_PUSH_POINTS or mr_push_len < SEMP3_MIN_PUSH_POINTS

        base_points = binfo["push_points"]
        mr_points = minfo["push_points"]
        if use_fallback:
            base_points = _trajectory_points(base, "eef_path")
            mr_points = _trajectory_points(mr, "eef_path")
            reasons.append(
                "fallback_to_eef_path_due_to_insufficient_push_points"
                f"(base_push_len={base_push_len}, followup_push_len={mr_push_len})"
            )
            if base_push_len < SEMP3_MIN_PUSH_POINTS:
                reasons.append("insufficient_base_push_points")
            if mr_push_len < SEMP3_MIN_PUSH_POINTS:
                reasons.append("insufficient_followup_push_points")

        if expected_translation is None:
            cube_src_base = _trajectory_points(base, "src_cube_pos") or _trajectory_points(base, "cube_pos")
            cube_src_mr = _trajectory_points(mr, "src_cube_pos") or _trajectory_points(mr, "cube_pos")
            if cube_src_base and cube_src_mr:
                expected_translation = np.asarray(cube_src_mr[0] - cube_src_base[0], dtype=np.float64)

        metrics = None
        similarity = None
        translated_followup_points = None
        if expected_translation is not None:
            translated_followup_points = _shift_points(mr_points, expected_translation)
            metrics = _trajectory_similarity_metrics(
                base_points,
                translated_followup_points,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append(
                    "cannot_compute_shifted_trajectory_similarity"
                    if use_fallback
                    else "cannot_compute_shifted_push_trajectory_similarity"
                )
            else:
                similarity = float(metrics["trajectory_similarity"])
                if similarity > similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Translation Trajectory Equivariance Breakdown "
                        f"(similarity={similarity:.6f} < {similarity_threshold:.6f})"
                    )
                else:
                    reasons.append(
                        "trajectory_preserves_translation_equivariance"
                        if use_fallback
                        else "push_trajectory_preserves_translation_equivariance"
                    )
        else:
            analyzable = False
            reasons.append("missing_expected_translation_delta")

        contact_point_distance_m = None
        push_point_changed = None
        if binfo["contact_point"] is not None and minfo["contact_point"] is not None:
            contact_point_distance_m = float(
                np.linalg.norm(
                    np.asarray(minfo["contact_point"], dtype=np.float64)[:3]
                    - np.asarray(binfo["contact_point"], dtype=np.float64)[:3]
                )
            )
            push_point_changed = bool(contact_point_distance_m >= push_point_change_thresh_m)
            if push_point_changed:
                reasons.append("push_contact_point_changed")
            else:
                reasons.append(
                    "push_contact_point_unchanged"
                    f"(distance={contact_point_distance_m:.6f}m < {push_point_change_thresh_m:.6f}m)"
                )
        else:
            reasons.append("push_contact_point_change_unavailable")

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
                "contact_point_distance_m": contact_point_distance_m,
                "push_point_changed": push_point_changed,
                "push_point_change_thresh_m": push_point_change_thresh_m,
                "expected_translation": None if expected_translation is None else expected_translation.tolist(),
                "trajectory_similarity": similarity,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "followup_path_len_m": None if metrics is None else metrics["followup_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "sample_points": None if metrics is None else metrics["sample_points"],
                "similarity_threshold": similarity_threshold,
                "contact_move_thresh_m": contact_move_thresh_m,
                "stop_move_thresh_m": stop_move_thresh_m,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SEMP3",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "similarity_threshold": similarity_threshold,
            "contact_move_thresh_m": contact_move_thresh_m,
            "stop_move_thresh_m": stop_move_thresh_m,
            "push_point_change_thresh_m": push_point_change_thresh_m,
            "invariance_proxy": "translated_push_trajectory_similarity_under_block_rotation",
            "attack_type": "state_space_geometric_equivariance_pattern",
        },
        "details": details,
    }
