from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SEMP1_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85
SEMP1_MSE_THRESHOLD_M = 0.01
SEMP1_CONTACT_MOVE_THRESH_M = 0.002
SEMP1_STOP_MOVE_THRESH_M = 0.001
SEMP1_MIN_PUSH_POINTS = 4
SEMP1_SAMPLE_POINTS = 24


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


def _first_valid_xyz(seq: Any) -> Optional[np.ndarray]:
    for item in seq or []:
        arr = _xyz(item)
        if arr is not None:
            return arr
    return None


def _last_valid_xyz(seq: Any) -> Optional[np.ndarray]:
    items = seq or []
    for item in reversed(items):
        arr = _xyz(item)
        if arr is not None:
            return arr
    return None


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
            "push_end_index": None,
            "push_points": [],
            "cube_points_len": len(cube_points),
            "eef_points_len": len(eef_points),
        }

    if contact_idx >= len(eef_points) or contact_idx >= len(cube_points):
        return {
            "contact_frame_index": contact_idx,
            "contact_frame_index_source": "contact_index_out_of_range",
            "push_end_index": None,
            "push_points": [],
            "cube_points_len": len(cube_points),
            "eef_points_len": len(eef_points),
        }

    push_end_idx = _push_end_index(cube_points, contact_idx, stop_move_thresh_m)
    upper = min(len(eef_points), push_end_idx + 1)
    push_points = eef_points[contact_idx:upper]
    return {
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "push_end_index": push_end_idx,
        "push_points": push_points,
        "cube_points_len": len(cube_points),
        "eef_points_len": len(eef_points),
    }


def _expected_translation(base: Any, mr: Any) -> Optional[np.ndarray]:
    base_src = _first_valid_xyz(getattr(base, "src_cube_pos", None) or getattr(base, "cube_pos", None))
    mr_src = _first_valid_xyz(getattr(mr, "src_cube_pos", None) or getattr(mr, "cube_pos", None))
    if base_src is not None and mr_src is not None:
        delta = mr_src - base_src
        if float(np.linalg.norm(delta[:2])) > 1e-12:
            return delta

    base_dst = _first_valid_xyz(getattr(base, "dst_cube_pos", None) or getattr(base, "target_pos", None))
    mr_dst = _first_valid_xyz(getattr(mr, "dst_cube_pos", None) or getattr(mr, "target_pos", None))
    if base_dst is not None and mr_dst is not None:
        delta = mr_dst - base_dst
        if float(np.linalg.norm(delta[:2])) > 1e-12:
            return delta

    base_goal = _last_valid_xyz(getattr(base, "target_pos", None) or getattr(base, "dst_cube_pos", None))
    mr_goal = _last_valid_xyz(getattr(mr, "target_pos", None) or getattr(mr, "dst_cube_pos", None))
    if base_goal is not None and mr_goal is not None:
        delta = mr_goal - base_goal
        if float(np.linalg.norm(delta[:2])) > 1e-12:
            return delta
    return None


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
    shifted_followup_points: List[np.ndarray],
    source_points: List[np.ndarray],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    sample_count = min(int(sample_points), len(shifted_followup_points), len(source_points))
    if sample_count < SEMP1_MIN_PUSH_POINTS:
        return None

    followup_resampled = _resample_polyline(shifted_followup_points, sample_count)
    source_resampled = _resample_polyline(source_points, sample_count)
    if followup_resampled is None or source_resampled is None:
        return None

    pointwise_delta = followup_resampled - source_resampled
    pointwise_dist = np.linalg.norm(pointwise_delta, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())
    mse = float(np.mean(np.sum(np.square(pointwise_delta), axis=1)))

    followup_path_len = _path_length(shifted_followup_points)
    source_path_len = _path_length(source_points)
    path_scale = source_path_len if source_path_len is not None and source_path_len > 1e-12 else followup_path_len
    if path_scale is None or path_scale <= 1e-12:
        return None

    relative_deviation = float(mean_pointwise_deviation / path_scale)
    similarity_score = float(1.0 / (1.0 + relative_deviation))
    return {
        "mean_pointwise_deviation_m": mean_pointwise_deviation,
        "mse": mse,
        "shifted_followup_path_len_m": float(followup_path_len) if followup_path_len is not None else None,
        "source_path_len_m": float(source_path_len) if source_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "sample_points": float(sample_count),
    }


def _shift_points(points: List[np.ndarray], shift: np.ndarray) -> List[np.ndarray]:
    return [np.asarray(point, dtype=np.float64)[:3] - shift[:3] for point in points]


def _fallback_trajectory(record: Any) -> Dict[str, Any]:
    """
    Fallback trajectory used when a push segment cannot be extracted (e.g. cube never moved).

    We compare full EEF paths (after applying the expected translation to the followup),
    so we can still compute a trajectory similarity proxy instead of marking the episode
    as unanalyzable.
    """
    eef_points = _trajectory_points(record, "eef_path")
    return {"points": eef_points, "len": len(eef_points)}


@register_mr_rule("MR-SEMP1")
@register_mr_rule("MR-SEMP-1")
@register_mr_rule("mr_semp_1")
def analyze_mr_semp1_visual_translation_push_equivariance(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    similarity_threshold = float(
        kwargs.get("similarity_threshold", SEMP1_TRAJECTORY_SIMILARITY_THRESHOLD)
    )
    mse_threshold_m = float(kwargs.get("mse_threshold_m", SEMP1_MSE_THRESHOLD_M))
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", SEMP1_CONTACT_MOVE_THRESH_M))
    stop_move_thresh_m = float(kwargs.get("stop_move_thresh_m", SEMP1_STOP_MOVE_THRESH_M))
    sample_points = int(kwargs.get("sample_points", SEMP1_SAMPLE_POINTS))

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
        expected_translation = _expected_translation(base, mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if expected_translation is None:
            analyzable = False
            reasons.append("missing_expected_translation_delta")

        base_push_len = len(binfo["push_points"])
        mr_push_len = len(minfo["push_points"])
        use_fallback = base_push_len < SEMP1_MIN_PUSH_POINTS or mr_push_len < SEMP1_MIN_PUSH_POINTS

        base_points = binfo["push_points"]
        mr_points = minfo["push_points"]
        if use_fallback:
            base_points = _fallback_trajectory(base)["points"]
            mr_points = _fallback_trajectory(mr)["points"]
            reasons.append(
                "fallback_to_eef_path_due_to_insufficient_push_points"
                f"(base_push_len={base_push_len}, followup_push_len={mr_push_len})"
            )
            if base_push_len < SEMP1_MIN_PUSH_POINTS:
                reasons.append("insufficient_base_push_points")
            if mr_push_len < SEMP1_MIN_PUSH_POINTS:
                reasons.append("insufficient_followup_push_points")

        metrics = None
        similarity = None
        mse = None
        shifted_followup_points = None
        if analyzable and expected_translation is not None:
            shifted_followup_points = _shift_points(mr_points, expected_translation)
            metrics = _trajectory_similarity_metrics(
                shifted_followup_points,
                base_points,
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
                mse = float(metrics["mse"])
                if similarity < similarity_threshold or mse >= mse_threshold_m:
                    violated = True
                    reasons.append(
                        "VIOLATION: Push Translation Equivariance Breakdown "
                        f"(similarity={similarity:.6f} < {similarity_threshold:.6f}"
                        f" or mse={mse:.6f} >= {mse_threshold_m:.6f})"
                    )
                else:
                    reasons.append(
                        "trajectory_preserves_translation_equivariance" if use_fallback else "push_trajectory_preserves_translation_equivariance"
                    )

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
                "base_push_end_index": binfo["push_end_index"],
                "mr_push_end_index": minfo["push_end_index"],
                "base_push_len": len(binfo["push_points"]),
                "mr_push_len": len(minfo["push_points"]),
                "expected_translation": None if expected_translation is None else expected_translation.tolist(),
                "trajectory_similarity": similarity,
                "mse": mse,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "shifted_followup_path_len_m": None if metrics is None else metrics["shifted_followup_path_len_m"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "sample_points": None if metrics is None else metrics["sample_points"],
                "similarity_threshold": similarity_threshold,
                "mse_threshold_m": mse_threshold_m,
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
        "mr_id": "MR-SEMP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "similarity_threshold": similarity_threshold,
            "mse_threshold_m": mse_threshold_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "stop_move_thresh_m": stop_move_thresh_m,
            "sample_points": sample_points,
            "invariance_proxy": "push_phase_trajectory_similarity_after_inverse_translation_alignment",
            "attack_type": "visual_2d_continuous_translation_of_cube_and_goal_region",
        },
        "details": details,
    }
