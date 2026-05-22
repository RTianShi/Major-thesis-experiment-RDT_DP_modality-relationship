from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SADP1_TRANSLATION_ERROR_THRESHOLD_M = 0.025
SADP1_PATH_LENGTH_RATIO_THRESHOLD = 1.35
SADP1_CONTACT_MOVE_THRESH_M = 0.002
SADP1_STOP_MOVE_THRESH_M = 0.001
SADP1_MIN_PUSH_POINTS = 4
SADP1_SAMPLE_POINTS = 24


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
        if float(np.linalg.norm(point - start)) > move_thresh_m:
            return idx, "cube_motion_onset"
    return None, "cube_never_moved"


def _push_segment(record: Any, move_thresh_m: float, stop_move_thresh_m: float) -> Dict[str, Any]:
    eef_points = _trajectory_points(record, "eef_path")
    cube_points = _trajectory_points(record, "cube_pos") or _trajectory_points(record, "src_cube_pos")
    contact_idx, contact_idx_source = _contact_index_from_cube_motion(record, move_thresh_m)

    if contact_idx is None:
        return {
            "contact_frame_index": None,
            "contact_frame_index_source": contact_idx_source,
            "push_end_index": None,
            "push_points": [],
        }

    if contact_idx >= len(eef_points) or contact_idx >= len(cube_points):
        return {
            "contact_frame_index": contact_idx,
            "contact_frame_index_source": "contact_index_out_of_range",
            "push_end_index": None,
            "push_points": [],
        }

    last_motion_idx = contact_idx
    prev = cube_points[contact_idx]
    for idx in range(contact_idx + 1, len(cube_points)):
        cur = cube_points[idx]
        if float(np.linalg.norm(cur - prev)) > stop_move_thresh_m:
            last_motion_idx = idx
        prev = cur

    upper = min(len(eef_points), last_motion_idx + 1)
    return {
        "contact_frame_index": contact_idx,
        "contact_frame_index_source": contact_idx_source,
        "push_end_index": last_motion_idx,
        "push_points": eef_points[contact_idx:upper],
    }


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


def _trajectory_similarity_metrics(
    source_points: List[np.ndarray],
    followup_points: List[np.ndarray],
    *,
    sample_points: int,
) -> Optional[Dict[str, float]]:
    sample_count = min(int(sample_points), len(source_points), len(followup_points))
    if sample_count < SADP1_MIN_PUSH_POINTS:
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


def _distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


@register_mr_rule("MR-SADP1")
@register_mr_rule("MR-SADP-1")
@register_mr_rule("SADP-Cross-Modal-Semantic-Noise")
@register_mr_rule("mr_sadp_1")
def analyze_mr_sadp1_cross_modal_semantic_noise(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    translation_error_threshold_m = float(
        kwargs.get("translation_error_threshold_m", SADP1_TRANSLATION_ERROR_THRESHOLD_M)
    )
    path_length_ratio_threshold = float(kwargs.get("path_length_ratio_threshold", SADP1_PATH_LENGTH_RATIO_THRESHOLD))
    contact_move_thresh_m = float(kwargs.get("contact_move_thresh_m", SADP1_CONTACT_MOVE_THRESH_M))
    stop_move_thresh_m = float(kwargs.get("stop_move_thresh_m", SADP1_STOP_MOVE_THRESH_M))
    sample_points = int(kwargs.get("sample_points", SADP1_SAMPLE_POINTS))

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
        base_push = _push_segment(base, contact_move_thresh_m, stop_move_thresh_m)
        mr_push = _push_segment(mr, contact_move_thresh_m, stop_move_thresh_m)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not base_points:
            analyzable = False
            reasons.append("missing_base_eef_path")
        if not mr_points:
            analyzable = False
            reasons.append("missing_mr_eef_path")
        if base_push["contact_frame_index"] is None:
            analyzable = False
            reasons.append("missing_base_contact_frame_index")
        if mr_push["contact_frame_index"] is None:
            analyzable = False
            reasons.append("missing_mr_contact_frame_index")

        base_src = _first_valid_xyz(getattr(base, "src_cube_pos", None) or getattr(base, "cube_pos", None))
        mr_src = _first_valid_xyz(getattr(mr, "src_cube_pos", None) or getattr(mr, "cube_pos", None))
        base_contact_point = base_push["push_points"][0] if base_push["push_points"] else None
        mr_contact_point = mr_push["push_points"][0] if mr_push["push_points"] else None
        base_final_point = _last_valid_xyz(base_points)
        mr_final_point = _last_valid_xyz(mr_points)

        translation_error_m = None
        path_length_ratio = None
        metrics = None

        if analyzable:
            if base_src is None or mr_src is None:
                analyzable = False
                reasons.append("missing_cube_translation_reference")
            else:
                delta_visual = mr_src - base_src
                if mr_contact_point is None or base_contact_point is None:
                    analyzable = False
                    reasons.append("cannot_compute_grasp_delta")
                else:
                    translation_error_m = float(np.linalg.norm((mr_contact_point - base_contact_point) - delta_visual))
                    if translation_error_m > translation_error_threshold_m:
                        violated = True
                        reasons.append(
                            "VIOLATION: Visual Decisiveness Violation: Spatial anchor drifted due to language noise."
                        )

            metrics = _trajectory_similarity_metrics(base_points, mr_points, sample_points=sample_points)
            if metrics is None:
                analyzable = False
                reasons.append("cannot_compute_trajectory_metrics")
            else:
                path_length_ratio = float(metrics["followup_path_len_m"] / max(metrics["source_path_len_m"], 1e-12))
                if path_length_ratio > path_length_ratio_threshold:
                    violated = True
                    reasons.append("VIOLATION: Trajectory Distortion: Language noise caused inefficient or warped path.")

            if not violated:
                reasons.append("cross_modal_semantic_noise_pass")

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
                "base_contact_frame_index": base_push["contact_frame_index"],
                "mr_contact_frame_index": mr_push["contact_frame_index"],
                "base_contact_point": None if base_contact_point is None else base_contact_point.tolist(),
                "mr_contact_point": None if mr_contact_point is None else mr_contact_point.tolist(),
                "base_final_point": None if base_final_point is None else base_final_point.tolist(),
                "mr_final_point": None if mr_final_point is None else mr_final_point.tolist(),
                "translation_error_m": translation_error_m,
                "path_length_ratio": path_length_ratio,
                "path_length_ratio_threshold": path_length_ratio_threshold,
                "translation_error_threshold_m": translation_error_threshold_m,
                "trajectory_similarity": None if metrics is None else metrics["trajectory_similarity"],
                "source_path_len_m": None if metrics is None else metrics["source_path_len_m"],
                "followup_path_len_m": None if metrics is None else metrics["followup_path_len_m"],
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SADP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "translation_error_threshold_m": translation_error_threshold_m,
            "path_length_ratio_threshold": path_length_ratio_threshold,
            "contact_move_thresh_m": contact_move_thresh_m,
            "stop_move_thresh_m": stop_move_thresh_m,
            "sample_points": sample_points,
            "violation_rule": "translation_error > threshold OR path_length_ratio > threshold",
        },
        "details": details,
    }
