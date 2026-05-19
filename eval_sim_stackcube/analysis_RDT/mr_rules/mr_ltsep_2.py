from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_utils import _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


LTSEP2_SIMILARITY_THRESHOLD = 0.82
LTSEP2_MIN_ALIGNMENT_POINTS = 4
LTSEP2_SAMPLE_POINTS = 24


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _nested_get(record, "trajectory", "eef_path") or getattr(record, "eef_path", None) or []
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


def _post_grasp_segment(record: Any) -> Tuple[Optional[List[List[float]]], Optional[int], Optional[str]]:
    traj = _trajectory_points(record)
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    if grasp_idx is None or grasp_idx < 0:
        return None, None, grasp_idx_source
    if grasp_idx >= len(traj) - 1:
        return None, grasp_idx, grasp_idx_source
    return traj[grasp_idx:], grasp_idx, grasp_idx_source


def _trajectory_similarity_metrics(
    base_points: List[List[float]],
    mr_points: List[List[float]],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    if base_points is None or mr_points is None:
        return None

    sample_count = min(int(sample_points), len(base_points), len(mr_points))
    if sample_count < LTSEP2_MIN_ALIGNMENT_POINTS:
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
        "base_post_grasp_path_len_m": float(base_path_len) if base_path_len is not None else None,
        "mr_post_grasp_path_len_m": float(mr_path_len) if mr_path_len is not None else None,
        "relative_deviation": relative_deviation,
        "trajectory_similarity": similarity_score,
        "endpoint_deviation_m": endpoint_deviation,
        "sample_points": float(sample_count),
    }


def _analyze_ltsep2_collision_torque(
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

        base_grasp_idx, base_grasp_idx_source = _grasp_index_with_source(base)
        mr_grasp_idx, mr_grasp_idx_source = _grasp_index_with_source(mr)

        base_segment, _, _ = _post_grasp_segment(base)
        mr_segment, _, _ = _post_grasp_segment(mr)

        analyzable = True
        violated = False
        reasons = []
        metrics = None

        if base_grasp_idx is None:
            analyzable = False
            reasons.append("missing_base_grasp_frame_index")
        if mr_grasp_idx is None:
            analyzable = False
            reasons.append("missing_mr_grasp_frame_index")
        if base_segment is None:
            analyzable = False
            reasons.append("missing_base_post_grasp_trajectory")
        if mr_segment is None:
            analyzable = False
            reasons.append("missing_mr_post_grasp_trajectory")

        if analyzable:
            metrics = _trajectory_similarity_metrics(
                base_segment,
                mr_segment,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("insufficient_post_grasp_alignment_points")
            else:
                similarity = float(metrics["trajectory_similarity"])
                if similarity >= similarity_threshold:
                    violated = True
                    reasons.append(
                        f"trajectory_too_similar({similarity:.4f} >= {similarity_threshold:.4f})"
                    )
                else:
                    reasons.append("collision_torque_induced_divergence")

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
                "base_grasp_frame_index": base_grasp_idx,
                "mr_grasp_frame_index": mr_grasp_idx,
                "base_grasp_frame_index_source": base_grasp_idx_source,
                "mr_grasp_frame_index_source": mr_grasp_idx_source,
                "base_post_grasp_point_count": len(base_segment) if base_segment is not None else None,
                "mr_post_grasp_point_count": len(mr_segment) if mr_segment is not None else None,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "base_post_grasp_path_len_m": None if metrics is None else metrics["base_post_grasp_path_len_m"],
                "mr_post_grasp_path_len_m": None if metrics is None else metrics["mr_post_grasp_path_len_m"],
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
            "metric": "post_grasp_eef_trajectory_similarity",
            "violation_rule": "trajectory_similarity >= similarity_threshold",
            "invariance_proxy": "collision_torque_should_force_post_grasp_divergence",
        },
        "details": details,
    }


@register_mr_rule("MR-LTSEP-2")
@register_mr_rule("MR-LTSEP2")
@register_mr_rule("mr_ltsep_2")
def analyze_mr_ltsep2_invisible_obstacle(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_ltsep2_collision_torque(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-LTSEP-2-INVISIBLE-OBSTACLE"),
        similarity_threshold=float(kwargs.get("similarity_threshold", LTSEP2_SIMILARITY_THRESHOLD)),
        sample_points=int(kwargs.get("sample_points", LTSEP2_SAMPLE_POINTS)),
    )