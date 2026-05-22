from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


JDCP1_SIMILARITY_THRESHOLD = 0.85
JDCP1_MIN_ALIGNMENT_POINTS = 4
JDCP1_SAMPLE_POINTS = 32


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
    if sample_count < JDCP1_MIN_ALIGNMENT_POINTS:
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


@register_mr_rule("MR-JDCP1")
@register_mr_rule("MR-JDCP-1")
@register_mr_rule("mr_jdcp_1")
def analyze_mr_jdcp1_equivalent_language_mutation(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    similarity_threshold = float(kwargs.get("similarity_threshold", JDCP1_SIMILARITY_THRESHOLD))
    sample_points = int(kwargs.get("sample_points", JDCP1_SAMPLE_POINTS))

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0
    success_drop_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        base_success = getattr(base, "success", None)
        mr_success = getattr(mr, "success", None)
        base_points = _trajectory_points(base)
        mr_points = _trajectory_points(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not bool(base_success):
            analyzable = False
            reasons.append("base_not_successful")
        if mr_success is False:
            violated = True
            reasons.append("VIOLATION: Followup failed under equivalent language mutation")
            success_drop_count += 1
        if not base_points:
            analyzable = False
            reasons.append("missing_base_eef_path")
        if not mr_points:
            analyzable = False
            reasons.append("missing_mr_eef_path")

        similarity = None
        metrics = None
        if analyzable and base_points and mr_points:
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
                if similarity < similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Trajectory Consistency Breakdown "
                        f"(similarity={similarity:.6f} < {similarity_threshold:.6f})"
                    )
                else:
                    reasons.append("trajectory_consistent_under_equivalent_language")

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
        "mr_id": "MR-JDCP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "success_drop_count": success_drop_count,
        "config": {
            "similarity_threshold": similarity_threshold,
            "sample_points": sample_points,
            "invariance_proxy": "strict_trajectory_similarity_under_equivalent_language_mutation",
        },
        "details": details,
    }
