from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


SADP2_TRAJECTORY_SIMILARITY_THRESHOLD = 0.85
SADP2_MIN_ALIGNMENT_POINTS = 4
SADP2_SAMPLE_POINTS = 32


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

    base_goal = _first_valid_xyz(getattr(base, "target_pos", None) or getattr(base, "dst_cube_pos", None))
    mr_goal = _first_valid_xyz(getattr(mr, "target_pos", None) or getattr(mr, "dst_cube_pos", None))
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
    if sample_count < SADP2_MIN_ALIGNMENT_POINTS:
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


@register_mr_rule("MR-SADP2")
@register_mr_rule("MR-SADP-2")
@register_mr_rule("SADP-Intra-Modal-Background-Noise")
def analyze_mr_sadp2_background_noise_trajectory_equivariance(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    similarity_threshold = float(kwargs.get("similarity_threshold", SADP2_TRAJECTORY_SIMILARITY_THRESHOLD))
    sample_points = int(kwargs.get("sample_points", SADP2_SAMPLE_POINTS))

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
        expected_translation = _expected_translation(base, mr)
        base_points = _trajectory_points(base, "eef_path")
        mr_points = _trajectory_points(mr, "eef_path")

        analyzable = True
        violated = False
        reasons: List[str] = []

        if expected_translation is None:
            analyzable = False
            reasons.append("missing_expected_translation_delta")

        metrics = None
        similarity = None
        if analyzable and expected_translation is not None:
            shifted_followup_points = _shift_points(mr_points, expected_translation)
            metrics = _trajectory_similarity_metrics(
                shifted_followup_points,
                base_points,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("cannot_compute_shifted_trajectory_similarity")
            else:
                similarity = float(metrics["trajectory_similarity"])
                if similarity < similarity_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Background Noise Trajectory Equivariance Breakdown "
                        f"(similarity={similarity:.6f} < {similarity_threshold:.6f})"
                    )
                else:
                    reasons.append("trajectory_preserves_background_noise_invariance")

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
                "expected_translation": None if expected_translation is None else expected_translation.tolist(),
                "trajectory_similarity": similarity,
                "mean_pointwise_deviation_m": None if metrics is None else metrics["mean_pointwise_deviation_m"],
                "mse": None if metrics is None else metrics["mse"],
                "shifted_followup_path_len_m": None if metrics is None else metrics["shifted_followup_path_len_m"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "relative_deviation": None if metrics is None else metrics["relative_deviation"],
                "sample_points": None if metrics is None else metrics["sample_points"],
                "similarity_threshold": similarity_threshold,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SADP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "similarity_threshold": similarity_threshold,
            "sample_points": sample_points,
            "invariance_proxy": "translated_push_trajectory_similarity_under_background_material_and_proprio_noise",
            "attack_type": "visual_and_proprioceptive_background_noise",
        },
        "details": details,
    }
