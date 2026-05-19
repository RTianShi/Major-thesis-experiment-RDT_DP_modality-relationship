from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _nested_get
from .registry import register_mr_rule


GDIP1_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85
GDIP1_MIN_ALIGNMENT_POINTS = 4
GDIP1_SAMPLE_POINTS = 24


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
    blind_points: List[List[float]],
    sighted_points: List[List[float]],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    if blind_points is None or sighted_points is None:
        return None

    sample_count = min(int(sample_points), len(blind_points), len(sighted_points))
    if sample_count < GDIP1_MIN_ALIGNMENT_POINTS:
        return None

    blind_resampled = _resample_polyline(blind_points, sample_count)
    sighted_resampled = _resample_polyline(sighted_points, sample_count)
    if blind_resampled is None or sighted_resampled is None:
        return None

    pointwise_dist = np.linalg.norm(blind_resampled - sighted_resampled, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())

    blind_path_len = _path_length(blind_points)
    sighted_path_len = _path_length(sighted_points)
    path_scale = sighted_path_len if sighted_path_len is not None and sighted_path_len > 1e-12 else blind_path_len
    if path_scale is None or path_scale <= 1e-12:
        return None

    relative_deviation = float(mean_pointwise_deviation / path_scale)
    similarity_score = float(1.0 / (1.0 + relative_deviation))
    endpoint_deviation = float(np.linalg.norm(blind_resampled[-1] - sighted_resampled[-1]))

    return {
        "mean_pointwise_deviation_m": mean_pointwise_deviation,
        "blind_path_len_m": float(blind_path_len) if blind_path_len is not None else None,
        "sighted_path_len_m": float(sighted_path_len) if sighted_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "endpoint_deviation_m": endpoint_deviation,
        "sample_points": float(sample_count),
    }


def _analyze_gdip1_full_episode_visual_blindness(
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

        sighted_success = getattr(base, "success", None)
        blind_success = getattr(mr, "success", None)

        sighted_traj = _trajectory_points(base)
        blind_traj = _trajectory_points(mr)

        analyzable = True
        violated = False
        reasons = []
        metrics = None

        if blind_success is True:
            violated = True
            reasons.append(
                "CRITICAL VIOLATION: X-Ray Vision! Model succeeded while blind (Spatial Overfitting)."
            )

        if not sighted_traj:
            analyzable = False
            reasons.append("missing_sighted_eef_trajectory")
        if not blind_traj:
            analyzable = False
            reasons.append("missing_blind_eef_trajectory")

        if analyzable:
            metrics = _trajectory_similarity_metrics(
                blind_traj,
                sighted_traj,
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
                        "VIOLATION: Blind trajectory mimicked sighted trajectory (Memorized Path)."
                    )

        if analyzable and not violated:
            reasons.append("expected_failure_without_visual_grounding")

        if not analyzable:
            unavailable_count += 1

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "sighted_success": sighted_success,
                "blind_success": blind_success,
                "sighted_point_count": len(sighted_traj),
                "blind_point_count": len(blind_traj),
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "blind_path_len_m": None if metrics is None else metrics["blind_path_len_m"],
                "sighted_path_len_m": None if metrics is None else metrics["sighted_path_len_m"],
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
            "metric": "blind_success_and_trajectory_mimicry",
            "violation_rule": "blind_success == True OR trajectory_similarity > similarity_threshold",
            "expected_followup_success_rate_percent": 0.0,
            "invariance_expectation": "full_episode_visual_blindness_should_prevent_grounded_object_reaching",
        },
        "details": details,
    }


@register_mr_rule("MR-GDIP1")
@register_mr_rule("MR-GDIP-1")
@register_mr_rule("mr_gdip_1")
def analyze_mr_gdip1_full_episode_visual_blindness(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_gdip1_full_episode_visual_blindness(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-GDIP1-FULL-EPISODE-VISUAL-BLINDNESS"),
        similarity_threshold=float(
            kwargs.get("similarity_threshold", GDIP1_TRAJECTORY_SIMILARITY_THRESHOLD)
        ),
        sample_points=int(kwargs.get("sample_points", GDIP1_SAMPLE_POINTS)),
    )
