from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


CPTMP1_BLIND_START_RATIO = 0.1
CPTMP1_BLIND_END_RATIO = 0.4
CPTMP1_JITTER_RATIO_THRESHOLD = 4.0
CPTMP1_MAX_BLIND_ANGLE_DEG = 60.0


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


def _goal_point(record: Any) -> Optional[List[float]]:
    goal = _first_not_none(
        _nested_get(record, "goal_point"),
        _nested_get(record, "mr_eval", "goal_point"),
        _nested_get(record, "trajectory", "goal_point"),
    )
    if goal is None:
        return None
    try:
        arr = np.asarray(goal, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _blind_interval_indices(
    record: Any,
    traj_len: int,
    *,
    blind_start_ratio: float,
    blind_end_ratio: float,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    if grasp_idx is None or traj_len <= 0:
        return None, None, grasp_idx, grasp_idx_source

    tgrasp = max(1, min(int(grasp_idx), traj_len - 1))
    blind_start = max(0, min(traj_len - 1, int(round(tgrasp * blind_start_ratio))))
    blind_end_exclusive = max(blind_start + 1, int(round(tgrasp * blind_end_ratio)))
    blind_end_exclusive = min(traj_len, blind_end_exclusive)
    return blind_start, blind_end_exclusive, grasp_idx, grasp_idx_source


def _slice_interval(points: List[List[float]], start: Optional[int], end_exclusive: Optional[int]) -> List[List[float]]:
    if start is None or end_exclusive is None:
        return []
    if not points:
        return []
    start = max(0, min(len(points), int(start)))
    end_exclusive = max(start, min(len(points), int(end_exclusive)))
    return points[start:end_exclusive]


def _jitter_std(points: List[List[float]]) -> Optional[float]:
    if points is None or len(points) < 3:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    step_norms = np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1)
    if step_norms.size == 0:
        return None
    return float(np.std(step_norms))


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cosine = float(np.dot(a, b) / (na * nb))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _blind_interval_heading_angle(
    points: List[List[float]],
    goal_point: Optional[List[float]],
    start: Optional[int],
    end_exclusive: Optional[int],
) -> Optional[float]:
    interval = _slice_interval(points, start, end_exclusive)
    if len(interval) < 2 or goal_point is None:
        return None

    interval_arr = np.asarray(interval, dtype=np.float64)
    goal_arr = np.asarray(goal_point, dtype=np.float64)[:3]
    motion_vec = interval_arr[-1, :3] - interval_arr[0, :3]
    target_vec = goal_arr - interval_arr[0, :3]
    return _angle_between_deg(motion_vec, target_vec)


def _analyze_cptmp1_early_phase_approaching_blindness(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    blind_start_ratio: float,
    blind_end_ratio: float,
    jitter_ratio_threshold: float,
    max_blind_angle_deg: float,
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
        goal_point = _first_not_none(_goal_point(base), _goal_point(mr))

        base_start, base_end, base_grasp_idx, base_grasp_source = _blind_interval_indices(
            base,
            len(base_traj),
            blind_start_ratio=blind_start_ratio,
            blind_end_ratio=blind_end_ratio,
        )
        mr_start, mr_end, mr_grasp_idx, mr_grasp_source = _blind_interval_indices(
            mr,
            len(mr_traj),
            blind_start_ratio=blind_start_ratio,
            blind_end_ratio=blind_end_ratio,
        )

        analyzable = True
        violated = False
        reasons = []

        if not base_traj:
            analyzable = False
            reasons.append("missing_base_eef_trajectory")
        if not mr_traj:
            analyzable = False
            reasons.append("missing_mr_eef_trajectory")
        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if base_start is None or base_end is None:
            analyzable = False
            reasons.append("missing_base_blind_interval")
        if mr_start is None or mr_end is None:
            analyzable = False
            reasons.append("missing_mr_blind_interval")

        base_interval = _slice_interval(base_traj, base_start, base_end)
        mr_interval = _slice_interval(mr_traj, mr_start, mr_end)
        base_jitter = None
        mr_jitter = None
        jitter_ratio = None
        blind_angle_deg = None

        if analyzable:
            if len(base_interval) < 3:
                analyzable = False
                reasons.append("base_blind_interval_too_short")
            if len(mr_interval) < 3:
                analyzable = False
                reasons.append("mr_blind_interval_too_short")

        if analyzable:
            base_jitter = _jitter_std(base_interval)
            mr_jitter = _jitter_std(mr_interval)
            blind_angle_deg = _blind_interval_heading_angle(mr_traj, goal_point, mr_start, mr_end)

            if base_jitter is None:
                analyzable = False
                reasons.append("missing_base_blind_jitter")
            if mr_jitter is None:
                analyzable = False
                reasons.append("missing_mr_blind_jitter")
            if blind_angle_deg is None:
                analyzable = False
                reasons.append("missing_blind_heading_angle")

        if analyzable:
            jitter_ratio = float(mr_jitter / max(base_jitter, 1e-12))
            if jitter_ratio > jitter_ratio_threshold:
                violated = True
                reasons.append("VIOLATION: Temporal collapse (Violent twitching during visual blink).")
            if blind_angle_deg > max_blind_angle_deg:
                violated = True
                reasons.append("VIOLATION: Intent loss (Robot strayed or stopped during blink).")
            if not violated:
                reasons.append("temporal_state_stability_pass")

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
                "goal_point": goal_point,
                "base_grasp_frame_index": base_grasp_idx,
                "mr_grasp_frame_index": mr_grasp_idx,
                "base_grasp_frame_index_source": base_grasp_source,
                "mr_grasp_frame_index_source": mr_grasp_source,
                "base_blind_interval_start": base_start,
                "base_blind_interval_end_exclusive": base_end,
                "mr_blind_interval_start": mr_start,
                "mr_blind_interval_end_exclusive": mr_end,
                "base_blind_interval_len": len(base_interval),
                "mr_blind_interval_len": len(mr_interval),
                "base_blind_jitter": base_jitter,
                "mr_blind_jitter": mr_jitter,
                "jitter_ratio": jitter_ratio,
                "blind_heading_angle_deg": blind_angle_deg,
                "blind_start_ratio": blind_start_ratio,
                "blind_end_ratio": blind_end_ratio,
                "jitter_ratio_threshold": jitter_ratio_threshold,
                "max_blind_angle_deg": max_blind_angle_deg,
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
            "blind_start_ratio": blind_start_ratio,
            "blind_end_ratio": blind_end_ratio,
            "jitter_ratio_threshold": jitter_ratio_threshold,
            "max_blind_angle_deg": max_blind_angle_deg,
            "metric": "blink_interval_jitter_and_heading_retention",
            "violation_rule": "blind_jitter_ratio > threshold OR blind_heading_angle_deg > max_blind_angle_deg",
            "invariance_expectation": "midflight_visual_blink_should_preserve_temporal_intent_and_stability",
        },
        "details": details,
    }


@register_mr_rule("MR-CPTMP1")
@register_mr_rule("MR-CPTMP-1")
@register_mr_rule("mr_cptmp_1")
def analyze_mr_cptmp1_early_phase_approaching_blindness(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_cptmp1_early_phase_approaching_blindness(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-CPTMP1-EARLY-PHASE-APPROACHING-BLINDNESS"),
        blind_start_ratio=float(kwargs.get("blind_start_ratio", CPTMP1_BLIND_START_RATIO)),
        blind_end_ratio=float(kwargs.get("blind_end_ratio", CPTMP1_BLIND_END_RATIO)),
        jitter_ratio_threshold=float(kwargs.get("jitter_ratio_threshold", CPTMP1_JITTER_RATIO_THRESHOLD)),
        max_blind_angle_deg=float(kwargs.get("max_blind_angle_deg", CPTMP1_MAX_BLIND_ANGLE_DEG)),
    )
