from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


DRP1_TRAJECTORY_DIVERGENCE_THRESHOLD_M = 0.08
DRP1_APPROACH_POINT_DIVERGENCE_THRESHOLD_M = 0.05
DRP1_CONTACT_MOVE_THRESH_M = 0.002
DRP1_MIN_ALIGNMENT_POINTS = 8
DRP1_SAMPLE_POINTS = 32


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
        if float(np.linalg.norm(point - start)) > move_thresh_m:
            return idx, "cube_motion_onset"
    return None, "cube_never_moved"


def _path_length(points: List[np.ndarray]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


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


def _trajectory_divergence_metrics(
    source_points: List[np.ndarray],
    followup_points: List[np.ndarray],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    sample_count = min(int(sample_points), len(source_points), len(followup_points))
    if sample_count < DRP1_MIN_ALIGNMENT_POINTS:
        return None

    source_resampled = _resample_polyline(source_points, sample_count)
    followup_resampled = _resample_polyline(followup_points, sample_count)
    if source_resampled is None or followup_resampled is None:
        return None

    pointwise_dist = np.linalg.norm(followup_resampled - source_resampled, axis=1)
    mean_pointwise_deviation = float(pointwise_dist.mean())
    max_pointwise_deviation = float(pointwise_dist.max())
    source_path_len = _path_length(source_points)
    followup_path_len = _path_length(followup_points)

    return {
        "trajectory_divergence_m": mean_pointwise_deviation,
        "max_pointwise_deviation_m": max_pointwise_deviation,
        "source_path_len_m": None if source_path_len is None else float(source_path_len),
        "followup_path_len_m": None if followup_path_len is None else float(followup_path_len),
        "sample_points": float(sample_count),
    }


def _approach_anchor(record: Any, move_thresh_m: float) -> Dict[str, Any]:
    eef_points = _trajectory_points(record, "eef_path")
    contact_idx, contact_idx_source = _contact_index_from_cube_motion(record, move_thresh_m)
    if contact_idx is None or not eef_points:
        return {
            "contact_frame_index": contact_idx,
            "contact_frame_index_source": contact_idx_source,
            "approach_frame_index": None,
            "approach_point": None,
        }

    if contact_idx <= 0:
        approach_idx = 0
    else:
        approach_idx = max(0, contact_idx - 1)
    if approach_idx >= len(eef_points):
        return {
            "contact_frame_index": contact_idx,
            "contact_frame_index_source": "approach_index_out_of_range",
            "approach_frame_index": None,
            "approach_point": None,
        }
    return {
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "approach_frame_index": approach_idx,
        "approach_point": eef_points[approach_idx],
    }


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


@register_mr_rule("MR-DRP1")
@register_mr_rule("MR-DRP-1")
@register_mr_rule("mr_drp_1")
def analyze_mr_drp1_extreme_diagonal_perturbation(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    trajectory_divergence_threshold_m = float(
        kwargs.get("trajectory_divergence_threshold_m", DRP1_TRAJECTORY_DIVERGENCE_THRESHOLD_M)
    )
    approach_point_divergence_threshold_m = float(
        kwargs.get("approach_point_divergence_threshold_m", DRP1_APPROACH_POINT_DIVERGENCE_THRESHOLD_M)
    )
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", DRP1_CONTACT_MOVE_THRESH_M))
    sample_points = int(kwargs.get("sample_points", DRP1_SAMPLE_POINTS))

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
        base_points = _trajectory_points(base, "eef_path")
        mr_points = _trajectory_points(mr, "eef_path")
        binfo = _approach_anchor(base, contact_move_thresh_m)
        minfo = _approach_anchor(mr, contact_move_thresh_m)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not bool(base_success):
            analyzable = False
            reasons.append("base_not_successful")
        if not base_points:
            analyzable = False
            reasons.append("missing_base_eef_path")
        if not mr_points:
            analyzable = False
            reasons.append("missing_mr_eef_path")
        if binfo["approach_point"] is None:
            analyzable = False
            reasons.append("missing_base_approach_point")
        if minfo["approach_point"] is None:
            analyzable = False
            reasons.append("missing_followup_approach_point")

        trajectory_divergence_m = None
        approach_point_divergence_m = None
        metrics = None

        if analyzable:
            metrics = _trajectory_divergence_metrics(
                base_points,
                mr_points,
                sample_points=sample_points,
            )
            if metrics is None:
                analyzable = False
                reasons.append("cannot_compute_trajectory_divergence")
            else:
                trajectory_divergence_m = float(metrics["trajectory_divergence_m"])
                if trajectory_divergence_m <= trajectory_divergence_threshold_m:
                    violated = True
                    reasons.append(
                        "VIOLATION: Blind Executor Trajectory Invariance "
                        f"(trajectory_divergence={trajectory_divergence_m:.6f}m <= "
                        f"{trajectory_divergence_threshold_m:.6f}m)"
                    )

            approach_point_divergence_m = _distance(
                binfo["approach_point"],
                minfo["approach_point"],
            )
            if approach_point_divergence_m is None:
                analyzable = False
                reasons.append("cannot_compute_approach_point_divergence")
            elif approach_point_divergence_m <= approach_point_divergence_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Approach Phase Landing Point Did Not Shift "
                    f"(approach_divergence={approach_point_divergence_m:.6f}m <= "
                    f"{approach_point_divergence_threshold_m:.6f}m)"
                )

            if not violated:
                reasons.append("trajectory_responds_to_diagonal_cube_shift")

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
                "base_approach_frame_index": binfo["approach_frame_index"],
                "mr_approach_frame_index": minfo["approach_frame_index"],
                "base_approach_point": None if binfo["approach_point"] is None else binfo["approach_point"].tolist(),
                "mr_approach_point": None if minfo["approach_point"] is None else minfo["approach_point"].tolist(),
                "trajectory_divergence_m": trajectory_divergence_m,
                "approach_point_divergence_m": approach_point_divergence_m,
                "max_pointwise_deviation_m": None if metrics is None else metrics["max_pointwise_deviation_m"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "followup_path_len_m": None if metrics is None else metrics["followup_path_len_m"],
                "sample_points": None if metrics is None else metrics["sample_points"],
                "trajectory_divergence_threshold_m": trajectory_divergence_threshold_m,
                "approach_point_divergence_threshold_m": approach_point_divergence_threshold_m,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-DRP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "trajectory_divergence_threshold_m": trajectory_divergence_threshold_m,
            "approach_point_divergence_threshold_m": approach_point_divergence_threshold_m,
            "contact_move_thresh_m": contact_move_thresh_m,
            "sample_points": sample_points,
            "metamorphic_expectation": "followup trajectory must diverge significantly from source",
            "violation_rule": (
                "trajectory_divergence <= threshold OR approach_point_divergence <= threshold"
            ),
        },
        "details": details,
    }
