from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


def _pair_key(record: Any):
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _grasp_index(record: Any) -> Optional[int]:
    idx = _first_not_none(
        getattr(record, "derived_grasp_frame_index", None),
        getattr(record, "mr_eval_grasp_frame_index", None),
    )
    if idx is None:
        return None
    return int(idx)


def _to_xyz(point: Any) -> Optional[List[float]]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64)
    if arr.ndim == 0:
        return None
    arr = arr.reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _point_at(path: Any, idx: Optional[int]) -> Optional[List[float]]:
    if idx is None:
        return None
    seq = path or []
    if not (0 <= idx < len(seq)):
        return None
    return _to_xyz(seq[idx])


def _l2_distance(point_a: Optional[List[float]], point_b: Optional[List[float]]) -> Optional[float]:
    if point_a is None or point_b is None:
        return None
    arr_a = np.asarray(point_a, dtype=np.float64)
    arr_b = np.asarray(point_b, dtype=np.float64)
    return float(np.linalg.norm(arr_a[:3] - arr_b[:3]))


@register_mr_rule("MR-JSAP-1")
@register_mr_rule("MR-JSAP-2")
@register_mr_rule("MR-JSAP-3")
@register_mr_rule("MR-JSAP1")
def analyze_mr_jsap_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    grasp_error_threshold_m = float(kwargs.get("grasp_error_threshold_m", 0.04))

    bmap = {_pair_key(record): record for record in base_records}
    mmap = {_pair_key(record): record for record in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        followup_grasp_frame_index = _grasp_index(mr)
        followup_grasp_point = _point_at(getattr(mr, "eef_path", None), followup_grasp_frame_index)
        followup_blue_cylinder_pos = _point_at(getattr(mr, "cube_pos", None), followup_grasp_frame_index)
        grasp_error_m = _l2_distance(followup_grasp_point, followup_blue_cylinder_pos)

        base_success = getattr(base, "success", None)
        followup_success = getattr(mr, "success", None)

        analyzable = True
        violated = False
        reasons: List[str] = []
        forced_violation = base_success is True and followup_success is False

        if followup_grasp_frame_index is None:
            violated = True
            reasons.append("missing_followup_grasp_frame")
        elif followup_grasp_point is None:
            analyzable = False
            reasons.append("missing_followup_grasp_point")
        elif followup_blue_cylinder_pos is None:
            analyzable = False
            reasons.append("missing_followup_blue_cylinder_position")
        elif grasp_error_m is not None and grasp_error_m > grasp_error_threshold_m:
            violated = True
            reasons.append(
                f"Anchor Spatial Drift (Model failed to lock onto blue cylinder, error: {grasp_error_m:.4f}m)"
            )
        else:
            reasons.append("grasp_locked_onto_blue_cylinder")

        if forced_violation:
            violated = True
            reasons.append("Catastrophic_Failure: Base_Success_Followup_Failure")

        if not analyzable and not forced_violation:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "source_success": base_success,
            "followup_success": followup_success,
            "followup_grasp_frame_index": followup_grasp_frame_index,
            "followup_grasp_point": followup_grasp_point,
            "followup_blue_cylinder_pos": followup_blue_cylinder_pos,
            "grasp_error_m": grasp_error_m,
            "grasp_error_threshold_m": grasp_error_threshold_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-JSAP-1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "grasp_error_threshold_m": grasp_error_threshold_m,
        },
        "details": details,
    }
