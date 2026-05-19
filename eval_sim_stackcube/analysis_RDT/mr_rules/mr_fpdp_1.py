from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _nested_get
from .registry import register_mr_rule


FPDP1_TRAJECTORY_SIMILARITY_THRESHOLD = 0.95
FPDP1_MIN_ALIGNMENT_POINTS = 4
FPDP1_SAMPLE_POINTS = 24


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
    blurred_points: List[List[float]],
    base_points: List[List[float]],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    if blurred_points is None or base_points is None:
        return None

    sample_count = min(int(sample_points), len(blurred_points), len(base_points))
    if sample_count < FPDP1_MIN_ALIGNMENT_POINTS:
        return None

    blurred_resampled = _resample_polyline(blurred_points, sample_count)
    base_resampled = _resample_polyline(base_points, sample_count)
    if blurred_resampled is None or base_resampled is None:
        return None

    pointwise_dist = np.linalg.norm(blurred_resampled - base_resampled, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())

    blurred_path_len = _path_length(blurred_points)
    base_path_len = _path_length(base_points)
    path_scale = base_path_len if base_path_len is not None and base_path_len > 1e-12 else blurred_path_len
    if path_scale is None or path_scale <= 1e-12:
        return None

    relative_deviation = float(mean_pointwise_deviation / path_scale)
    similarity_score = float(1.0 / (1.0 + relative_deviation))
    endpoint_deviation = float(np.linalg.norm(blurred_resampled[-1] - base_resampled[-1]))

    return {
        "mean_pointwise_deviation_m": mean_pointwise_deviation,
        "blurred_path_len_m": float(blurred_path_len) if blurred_path_len is not None else None,
        "base_path_len_m": float(base_path_len) if base_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "endpoint_deviation_m": endpoint_deviation,
        "sample_points": float(sample_count),
    }


def _analyze_fpdp1_high_frequency_edge_blurring(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    similarity_threshold: float,
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

        analyzable = True
        violated = False
        reasons = []
        metrics = None

        if not base_traj:
            analyzable = False
            reasons.append("missing_base_eef_trajectory")
        if not mr_traj:
            analyzable = False
            reasons.append("missing_mr_eef_trajectory")

        if analyzable:
            metrics = _trajectory_similarity_metrics(
                mr_traj,
                base_traj,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("insufficient_trajectory_alignment_points")
            else:
                similarity = float(metrics["trajectory_similarity"])
                if similarity > similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Feature-Indifferent Execution (Trajectory identical despite visual blur)."
                    )
                else:
                    reasons.append("expected_degradation_under_edge_blurring")

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
                "base_point_count": len(base_traj),
                "mr_point_count": len(mr_traj),
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "blurred_path_len_m": None if metrics is None else metrics["blurred_path_len_m"],
                "base_path_len_m": None if metrics is None else metrics["base_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "trajectory_similarity": None if metrics is None else metrics["trajectory_similarity"],
                "endpoint_deviation_m": None if metrics is None else metrics["endpoint_deviation_m"],
                "sample_points": sample_points,
                "similarity_threshold": similarity_threshold,
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
            "sample_points": sample_points,
            "similarity_threshold": similarity_threshold,
            "metric": "eef_trajectory_similarity_under_visual_blur",
            "violation_rule": "trajectory_similarity > similarity_threshold",
            "invariance_expectation": "edge_blur_should_change_precision_dependent_execution",
        },
        "details": details,
    }


@register_mr_rule("MR-FPDP1")
@register_mr_rule("MR-FPDP-1")
@register_mr_rule("mr_fpdp_1")
def analyze_mr_fpdp1_high_frequency_edge_blurring(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_fpdp1_high_frequency_edge_blurring(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-FPDP1-HIGH-FREQUENCY-EDGE-BLURRING"),
        similarity_threshold=float(
            kwargs.get("similarity_threshold", FPDP1_TRAJECTORY_SIMILARITY_THRESHOLD)
        ),
        sample_points=int(kwargs.get("sample_points", FPDP1_SAMPLE_POINTS)),
    )
