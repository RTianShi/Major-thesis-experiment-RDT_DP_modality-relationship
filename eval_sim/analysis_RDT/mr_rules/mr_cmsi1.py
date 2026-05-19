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


@register_mr_rule("MR-CMSI1")
@register_mr_rule("MR-CMSI-1")
def analyze_mr_cmsi1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    blue_mismatch_threshold_m = float(kwargs.get("blue_mismatch_threshold_m", 0.04))
    red_drift_threshold_m = float(kwargs.get("red_drift_threshold_m", 0.025))

    bmap = {_pair_key(record): record for record in base_records}
    mmap = {_pair_key(record): record for record in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]
        positions = _load_positions(mr)

        grasp_frame_index = _grasp_index(mr)
        close_frame_index = _close_frame_index(mr)
        probe_frame_index = _first_not_none(grasp_frame_index, close_frame_index)
        grasp_point = _point_at(getattr(mr, "eef_path", None), probe_frame_index)

        red_cube_center = _pick_center_from_positions(
            positions,
            ["red_cube_initial", "red_cube", "red_cube_pos", "red_cube_final"],
        )
        blue_cube_center = _pick_center_from_positions(
            positions,
            [
                "cmsi1_blue_cube_initial",
                "cmsi1_blue_cube",
                "cmsi1_blue_cube_final",
                "blue_cube_initial",
                "blue_cube",
            ],
        )

        dist_to_red_cube_m = _l2_distance(grasp_point, red_cube_center)
        dist_to_blue_cube_m = _l2_distance(grasp_point, blue_cube_center)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if probe_frame_index is None:
            analyzable = False
            reasons.append("missing_followup_grasp_or_close_frame")
        elif grasp_point is None:
            analyzable = False
            reasons.append("missing_followup_probe_point")
        elif red_cube_center is None or blue_cube_center is None:
            analyzable = False
            reasons.append("missing_red_or_blue_cube_center")
        else:
            if dist_to_blue_cube_m is not None and dist_to_blue_cube_m < blue_mismatch_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Semantic Target Mismatch (Robot grasped the blue cube instead of red)"
                )
            if dist_to_red_cube_m is not None and dist_to_red_cube_m > red_drift_threshold_m:
                violated = True
                reasons.append(
                    "VIOLATION: Distractor-Induced Drift (Precision degraded due to visual noise)"
                )
            if not violated:
                reasons.append("Target_Locked_On_Red_Cube")

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
            "red_cube_center": red_cube_center,
            "blue_cube_center": blue_cube_center,
            "dist_to_red_cube_m": dist_to_red_cube_m,
            "dist_to_blue_cube_m": dist_to_blue_cube_m,
            "blue_mismatch_threshold_m": blue_mismatch_threshold_m,
            "red_drift_threshold_m": red_drift_threshold_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-CMSI1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "blue_mismatch_threshold_m": blue_mismatch_threshold_m,
            "red_drift_threshold_m": red_drift_threshold_m,
        },
        "details": details,
    }
