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
    return None if idx is None else int(idx)


def _close_frame_index(record: Any) -> Optional[int]:
    idx = _first_not_none(
        getattr(record, "mr_eval_gripper_fully_closed_frame_index", None),
        getattr(record, "gripper_fully_closed_frame_index", None),
    )
    return None if idx is None else int(idx)


def _flatten_path(path: Any) -> List[Any]:
    if not path:
        return []
    if isinstance(path[0], list) and path[0] and isinstance(path[0][0], list):
        return [p[0] for p in path if p and isinstance(p[0], list)]
    return path


def _to_xyz(point: Any) -> Optional[List[float]]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _point_at(path: Any, idx: Optional[int]) -> Optional[List[float]]:
    seq = _flatten_path(path)
    if idx is None or not (0 <= idx < len(seq)):
        return None
    return _to_xyz(seq[idx])


def _path_length(path: Any) -> Optional[float]:
    seq = _flatten_path(path)
    if len(seq) < 2:
        return 0.0 if seq else None
    points = []
    for point in seq:
        xyz = _to_xyz(point)
        if xyz is None:
            return None
        points.append(xyz)
    arr = np.asarray(points, dtype=np.float64)
    delta = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(delta, axis=1).sum())


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


@register_mr_rule("MR-GDIP2")
@register_mr_rule("MR-GDIP-2")
def analyze_mr_gdip_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    path_length_threshold_m = float(kwargs.get("path_length_threshold_m", 0.1))
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
        positions = _load_positions(mr)

        grasp_frame_index = _grasp_index(mr)
        close_frame_index = _close_frame_index(mr)
        probe_frame_index = _first_not_none(grasp_frame_index, close_frame_index)
        grasp_point = _point_at(getattr(mr, "eef_path", None), probe_frame_index)
        path_length_m = _path_length(getattr(mr, "eef_path", None))
        red_cube_center = _pick_center_from_positions(
            positions,
            ["red_cube_initial", "red_cube", "red_cube_pos", "red_cube_final"],
        )
        grasp_error_m = _l2_distance(grasp_point, red_cube_center)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if path_length_m is None:
            analyzable = False
            reasons.append("missing_followup_path_length")
        elif probe_frame_index is None:
            analyzable = False
            reasons.append("missing_followup_grasp_or_close_frame")
        elif grasp_point is None:
            analyzable = False
            reasons.append("missing_followup_probe_point")
        elif red_cube_center is None:
            analyzable = False
            reasons.append("missing_red_cube_center")
        elif path_length_m > path_length_threshold_m and grasp_error_m is not None and grasp_error_m < grasp_error_threshold_m:
            violated = True
            reasons.append(
                "VIOLATION: Task Intent Collapse (Robot executed grasp-and-place behavior under empty instruction)"
            )
        else:
            reasons.append("Language_Deprivation_Respected")

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
            "followup_path_length_m": path_length_m,
            "path_length_threshold_m": path_length_threshold_m,
            "grasp_error_m": grasp_error_m,
            "grasp_error_threshold_m": grasp_error_threshold_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-GDIP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "path_length_threshold_m": path_length_threshold_m,
            "grasp_error_threshold_m": grasp_error_threshold_m,
        },
        "details": details,
    }
