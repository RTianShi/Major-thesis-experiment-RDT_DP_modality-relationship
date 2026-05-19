from typing import Any, Dict, List, Optional

import json
import os

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


def _close_frame_index(record: Any) -> Optional[int]:
    idx = _first_not_none(
        getattr(record, "mr_eval_gripper_fully_closed_frame_index", None),
        getattr(record, "gripper_fully_closed_frame_index", None),
    )
    if idx is None:
        return None
    return int(idx)


def _flatten_path(path: Any) -> List[Any]:
    if not path:
        return []
    if isinstance(path[0], list) and path[0] and isinstance(path[0][0], list):
        return [p[0] for p in path if p and isinstance(p[0], list)]
    return path


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
    seq = _flatten_path(path)
    if not (0 <= idx < len(seq)):
        return None
    return _to_xyz(seq[idx])


def _l2_distance(point_a: Optional[List[float]], point_b: Optional[List[float]]) -> Optional[float]:
    if point_a is None or point_b is None:
        return None
    arr_a = np.asarray(point_a, dtype=np.float64)
    arr_b = np.asarray(point_b, dtype=np.float64)
    return float(np.linalg.norm(arr_a[:3] - arr_b[:3]))


def _load_positions(record: Any) -> Dict[str, Any]:
    file_path = getattr(record, "file", None)
    if not file_path or not os.path.isfile(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    positions = obj.get("positions", {})
    return positions if isinstance(positions, dict) else {}


def _pick_center_from_positions(positions: Dict[str, Any], candidates: List[str]) -> Optional[List[float]]:
    for key in candidates:
        if key not in positions:
            continue
        value = positions.get(key)
        if isinstance(value, list) and value:
            point = _to_xyz(value[0])
            if point is not None:
                return point
        point = _to_xyz(value)
        if point is not None:
            return point
    return None


@register_mr_rule("MR-SDPP-1")
@register_mr_rule("MR-SDPP1")
def analyze_mr_sdpp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    grasp_error_threshold_m = float(kwargs.get("grasp_error_threshold_m", 0.04))

    bmap = {_pair_key(record): record for record in base_records}
    mmap = {_pair_key(record): record for record in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0
    red_closer_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]
        positions = _load_positions(mr)

        grasp_frame_index = _grasp_index(mr)
        close_frame_index = _close_frame_index(mr)
        probe_frame_index = _first_not_none(grasp_frame_index, close_frame_index)
        grasp_point = _point_at(getattr(mr, "eef_path", None), probe_frame_index)
        red_sphere_center = _pick_center_from_positions(
            positions,
            ["red_sphere_initial", "red_sphere", "red_sphere_pos", "red_sphere_pose"],
        )
        blue_cube_center = _pick_center_from_positions(
            positions,
            ["blue_cube_initial", "blue_cube", "blue_cube_pos", "blue_cube_pose"],
        )

        dist_to_red_sphere_m = _l2_distance(grasp_point, red_sphere_center)
        dist_to_blue_cube_m = _l2_distance(grasp_point, blue_cube_center)
        if (
            dist_to_red_sphere_m is not None
            and dist_to_blue_cube_m is not None
            and dist_to_red_sphere_m < dist_to_blue_cube_m
        ):
            red_closer_count += 1

        analyzable = True
        violated = False
        reasons: List[str] = []

        if probe_frame_index is None:
            analyzable = False
            reasons.append("missing_followup_grasp_or_close_frame")
        elif grasp_point is None:
            analyzable = False
            reasons.append("missing_followup_probe_point")
        elif red_sphere_center is None or blue_cube_center is None:
            analyzable = False
            reasons.append("missing_red_sphere_or_blue_cube_center")
        else:
            if dist_to_red_sphere_m is not None and dist_to_red_sphere_m < grasp_error_threshold_m:
                violated = True
                reasons.append(
                    f"VIOLATION: Color Bias (Model locked onto red sphere, error: {dist_to_red_sphere_m:.4f}m)"
                )
            if dist_to_blue_cube_m is not None and dist_to_blue_cube_m < grasp_error_threshold_m:
                violated = True
                reasons.append(
                    f"VIOLATION: Shape Bias (Model locked onto blue cube, error: {dist_to_blue_cube_m:.4f}m)"
                )
            if not violated:
                reasons.append("Synergy_Failure_Protection_Triggered")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "source_success": getattr(base, "success", None),
            "followup_success": getattr(mr, "success", None),
            "followup_grasp_frame_index": grasp_frame_index,
            "followup_close_frame_index": close_frame_index,
            "followup_probe_frame_index": probe_frame_index,
            "followup_grasp_point": grasp_point,
            "red_sphere_center": red_sphere_center,
            "blue_cube_center": blue_cube_center,
            "dist_to_red_sphere_m": dist_to_red_sphere_m,
            "dist_to_blue_cube_m": dist_to_blue_cube_m,
            "grasp_error_threshold_m": grasp_error_threshold_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SDPP-1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "red_closer_count": red_closer_count,
        "config": {
            "grasp_error_threshold_m": grasp_error_threshold_m,
        },
        "details": details,
    }
