from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


CPTMP2_TAE_DELTA_TOL_M = 0.005
CPTMP2_TRAJECTORY_SIMILARITY_THRESHOLD = 0.95
CPTMP2_ALIGN_START_RATIO = 0.7
CPTMP2_MIN_ALIGNMENT_POINTS = 4
CPTMP2_SAMPLE_POINTS = 24


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


def _final_valid_xyz(seq: Any) -> Optional[np.ndarray]:
    items = seq or []
    for item in reversed(items):
        try:
            arr = np.asarray(item, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            return arr[:3]
    return None


def _final_red_pos(record: Any) -> Optional[np.ndarray]:
    return _final_valid_xyz(
        _first_not_none(
            _nested_get(record, "trajectory", "src_cube_pos"),
            getattr(record, "src_cube_pos", None),
            _nested_get(record, "trajectory", "cube_pos"),
            getattr(record, "cube_pos", None),
        )
    )


def _final_green_pos(record: Any) -> Optional[np.ndarray]:
    return _final_valid_xyz(
        _first_not_none(
            _nested_get(record, "trajectory", "dst_cube_pos"),
            getattr(record, "dst_cube_pos", None),
        )
    )


def _distance_xy(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]))


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


def _trajectory_similarity_metrics(
    base_points: List[List[float]],
    mr_points: List[List[float]],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    if base_points is None or mr_points is None:
        return None

    sample_count = min(int(sample_points), len(base_points), len(mr_points))
    if sample_count < CPTMP2_MIN_ALIGNMENT_POINTS:
        return None

    base_resampled = _resample_polyline(base_points, sample_count)
    mr_resampled = _resample_polyline(mr_points, sample_count)
    if base_resampled is None or mr_resampled is None:
        return None

    pointwise_dist = np.linalg.norm(base_resampled - mr_resampled, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())

    base_path_len = _path_length(base_points)
    mr_path_len = _path_length(mr_points)
    path_scale = base_path_len if base_path_len is not None and base_path_len > 1e-12 else mr_path_len
    if path_scale is None or path_scale <= 1e-12:
        return None

    relative_deviation = float(mean_pointwise_deviation / path_scale)
    similarity_score = float(1.0 / (1.0 + relative_deviation))
    endpoint_deviation = float(np.linalg.norm(base_resampled[-1] - mr_resampled[-1]))

    return {
        "mean_pointwise_deviation_m": mean_pointwise_deviation,
        "base_path_len_m": float(base_path_len) if base_path_len is not None else None,
        "mr_path_len_m": float(mr_path_len) if mr_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "endpoint_deviation_m": endpoint_deviation,
        "sample_points": float(sample_count),
    }


def _align_start_index(record: Any, traj_len: int, *, align_start_ratio: float) -> Dict[str, Any]:
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    if grasp_idx is None or traj_len <= 0:
        return {
            "align_start_index": None,
            "grasp_frame_index": grasp_idx,
            "grasp_frame_index_source": grasp_idx_source,
        }

    tail_len = max(1, traj_len - int(grasp_idx))
    align_start = int(grasp_idx) + int(round(tail_len * align_start_ratio))
    align_start = max(int(grasp_idx), min(traj_len - 1, align_start))
    return {
        "align_start_index": align_start,
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
    }


def _tail_segment(points: List[List[float]], start_idx: Optional[int]) -> List[List[float]]:
    if start_idx is None or not points:
        return []
    start_idx = max(0, min(len(points), int(start_idx)))
    return points[start_idx:]


def _analyze_cptmp2_critical_stacking_blindness(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    tae_delta_tol_m: float,
    similarity_threshold: float,
    align_start_ratio: float,
    sample_points: int,
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
        base_align = _align_start_index(base, len(base_traj), align_start_ratio=align_start_ratio)
        mr_align = _align_start_index(mr, len(mr_traj), align_start_ratio=align_start_ratio)
        base_tail = _tail_segment(base_traj, base_align["align_start_index"])
        mr_tail = _tail_segment(mr_traj, mr_align["align_start_index"])

        base_red_final = _final_red_pos(base)
        base_green_final = _final_green_pos(base)
        mr_red_final = _final_red_pos(mr)
        mr_green_final = _final_green_pos(mr)

        tae_base = _distance_xy(base_red_final, base_green_final)
        tae_mr = _distance_xy(mr_red_final, mr_green_final)

        analyzable = True
        violated = False
        reasons = []
        metrics = None

        if tae_base is None:
            analyzable = False
            reasons.append("missing_base_terminal_alignment_error")
        if tae_mr is None:
            analyzable = False
            reasons.append("missing_mr_terminal_alignment_error")
        if not base_traj:
            analyzable = False
            reasons.append("missing_base_eef_trajectory")
        if not mr_traj:
            analyzable = False
            reasons.append("missing_mr_eef_trajectory")
        if base_align["align_start_index"] is None:
            analyzable = False
            reasons.append("missing_base_align_start")
        if mr_align["align_start_index"] is None:
            analyzable = False
            reasons.append("missing_mr_align_start")

        tae_delta_m = None
        if analyzable:
            tae_delta_m = float(abs(float(tae_mr) - float(tae_base)))
            if bool(mr_success) and tae_delta_m < tae_delta_tol_m:
                violated = True
                reasons.append(
                    "VIOLATION: Pseudo-Visual Servoing (Open-loop coordinate prediction detected)."
                )

            metrics = _trajectory_similarity_metrics(
                mr_tail,
                base_tail,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("insufficient_alignment_tail_points")
            else:
                similarity = float(metrics["trajectory_similarity"])
                if similarity > similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Open-loop Trajectory (Model ignored visual feedback for alignment)."
                    )

        if analyzable and not violated:
            reasons.append("alignment_blindness_induced_expected_degradation")

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
                "base_grasp_frame_index": base_align["grasp_frame_index"],
                "mr_grasp_frame_index": mr_align["grasp_frame_index"],
                "base_grasp_frame_index_source": base_align["grasp_frame_index_source"],
                "mr_grasp_frame_index_source": mr_align["grasp_frame_index_source"],
                "base_align_start_index": base_align["align_start_index"],
                "mr_align_start_index": mr_align["align_start_index"],
                "base_tail_point_count": len(base_tail),
                "mr_tail_point_count": len(mr_tail),
                "base_terminal_alignment_error_xy_m": tae_base,
                "mr_terminal_alignment_error_xy_m": tae_mr,
                "tae_delta_m": tae_delta_m,
                "tae_delta_tol_m": tae_delta_tol_m,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "base_tail_path_len_m": None if metrics is None else metrics["base_path_len_m"],
                "mr_tail_path_len_m": None if metrics is None else metrics["mr_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "trajectory_similarity": None if metrics is None else metrics["trajectory_similarity"],
                "endpoint_deviation_m": None if metrics is None else metrics["endpoint_deviation_m"],
                "sample_points": sample_points,
                "similarity_threshold": similarity_threshold,
                "align_start_ratio": align_start_ratio,
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
            "tae_delta_tol_m": tae_delta_tol_m,
            "similarity_threshold": similarity_threshold,
            "align_start_ratio": align_start_ratio,
            "sample_points": sample_points,
            "metric": "terminal_alignment_error_and_alignment_tail_similarity",
            "violation_rule": "mr_success AND tae_delta < tol OR trajectory_similarity(t_align:) > threshold",
            "invariance_expectation": "critical_alignment_blindness_should_break_visual_servoing_precision",
        },
        "details": details,
    }


@register_mr_rule("MR-CPTMP2")
@register_mr_rule("MR-CPTMP-2")
@register_mr_rule("mr_cptmp_2")
def analyze_mr_cptmp2_critical_stacking_blindness(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_cptmp2_critical_stacking_blindness(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-CPTMP2-CRITICAL-STACKING-BLINDNESS"),
        tae_delta_tol_m=float(kwargs.get("tae_delta_tol_m", CPTMP2_TAE_DELTA_TOL_M)),
        similarity_threshold=float(
            kwargs.get("similarity_threshold", CPTMP2_TRAJECTORY_SIMILARITY_THRESHOLD)
        ),
        align_start_ratio=float(kwargs.get("align_start_ratio", CPTMP2_ALIGN_START_RATIO)),
        sample_points=int(kwargs.get("sample_points", CPTMP2_SAMPLE_POINTS)),
    )
