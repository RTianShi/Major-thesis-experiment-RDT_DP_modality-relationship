from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


FPDP1_TAE_DELTA_THRESHOLD_M = 0.005
FPDP1_TRAJECTORY_SIMILARITY_THRESHOLD = 0.98
FPDP1_MIN_ALIGNMENT_POINTS = 4
FPDP1_SAMPLE_POINTS = 32


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


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = getattr(record, "eef_path", None) or []
    points: List[List[float]] = []
    for point in traj:
        arr = _xyz(point)
        if arr is not None:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
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


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _terminal_alignment_error(record: Any) -> Optional[float]:
    cube_end = _last_valid_xyz(getattr(record, "cube_pos", None) or getattr(record, "src_cube_pos", None))
    goal_end = _last_valid_xyz(getattr(record, "target_pos", None) or getattr(record, "dst_cube_pos", None))
    return _distance(cube_end, goal_end)


def _path_length(points: List[List[float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _resample_polyline(points: List[List[float]], sample_count: int) -> Optional[np.ndarray]:
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
    source_points: List[List[float]],
    followup_points: List[List[float]],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    sample_count = min(int(sample_points), len(source_points), len(followup_points))
    if sample_count < FPDP1_MIN_ALIGNMENT_POINTS:
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


@register_mr_rule("MR-FPDP1")
@register_mr_rule("MR-FPDP-1")
@register_mr_rule("FPDP-High-Frequency-Edge-Deprivation")
def analyze_mr_fpdp1_edge_deprivation_precision_bias(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    tae_delta_threshold_m = float(kwargs.get("tae_delta_threshold_m", FPDP1_TAE_DELTA_THRESHOLD_M))
    similarity_threshold = float(
        kwargs.get("similarity_threshold", FPDP1_TRAJECTORY_SIMILARITY_THRESHOLD)
    )
    sample_points = int(kwargs.get("sample_points", FPDP1_SAMPLE_POINTS))

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
        base_points = _trajectory_points(base)
        mr_points = _trajectory_points(mr)
        tae_s = _terminal_alignment_error(base)
        tae_f = _terminal_alignment_error(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if tae_s is None:
            analyzable = False
            reasons.append("missing_base_terminal_alignment_error")
        if tae_f is None:
            analyzable = False
            reasons.append("missing_followup_terminal_alignment_error")

        tae_delta = None
        if analyzable and tae_s is not None and tae_f is not None:
            tae_delta = abs(float(tae_f - tae_s))
            if tae_delta < tae_delta_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Shortcut Learning "
                    f"(abs_tae_delta={tae_delta:.6f}m < {tae_delta_threshold_m:.6f}m)"
                )
            else:
                reasons.append("terminal_alignment_precision_degraded_as_expected")

        similarity = None
        metrics = None
        if base_points and mr_points:
            metrics = _trajectory_similarity_metrics(
                base_points,
                mr_points,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("cannot_compute_trajectory_similarity")
            else:
                similarity = float(metrics["trajectory_similarity"])
                if similarity > similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Open-loop Execution "
                        f"(similarity={similarity:.6f} > {similarity_threshold:.6f})"
                    )
                else:
                    reasons.append("trajectory_changed_under_edge_blurring")
        else:
            analyzable = False
            reasons.append("missing_base_or_followup_eef_path")

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
                "tae_s": tae_s,
                "tae_f": tae_f,
                "abs_tae_delta_m": tae_delta,
                "tae_delta_threshold_m": tae_delta_threshold_m,
                "trajectory_similarity": similarity,
                "similarity_threshold": similarity_threshold,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "followup_path_len_m": None if metrics is None else metrics["followup_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "sample_points": None if metrics is None else metrics["sample_points"],
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-FPDP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "tae_delta_threshold_m": tae_delta_threshold_m,
            "similarity_threshold": similarity_threshold,
            "sample_points": sample_points,
            "invariance_proxy": "terminal_alignment_degradation_and_trajectory_flexibility_under_edge_blur",
            "attack_type": "high_frequency_edge_deprivation",
        },
        "details": details,
    }
