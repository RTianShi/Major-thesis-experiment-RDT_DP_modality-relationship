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


@register_mr_rule("MR-DCRB1")
@register_mr_rule("MR-DCRB-1")
def analyze_mr_dcrb1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    closer_margin_m = float(kwargs.get("closer_margin_m", 0.02))

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
        source_success = getattr(base, "success", None)
        followup_success = getattr(mr, "success", None)

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
                "dcrb1_blue_cube_initial",
                "dcrb1_blue_cube",
                "cmsi1_blue_cube_initial",
                "cmsi1_blue_cube",
                "cmsi1_blue_cube_final",
                "blue_cube_initial",
                "blue_cube",
                "blue_cube_pos",
            ],
        )

        dist_to_red_cube_m = _l2_distance(grasp_point, red_cube_center)
        dist_to_blue_cube_m = _l2_distance(grasp_point, blue_cube_center)

        analyzable = False
        violated = False
        reasons: List[str] = []

        if source_success is True and followup_success is True:
            violated = True
            analyzable = True
            reasons.append("VIOLATION: Both source and followup succeeded")

        if probe_frame_index is None:
            reasons.append("missing_followup_grasp_or_close_frame")
        elif grasp_point is None:
            reasons.append("missing_followup_probe_point")
        elif red_cube_center is None:
            reasons.append("missing_red_cube_center")
        elif blue_cube_center is None:
            reasons.append("missing_blue_cube_center")
        elif dist_to_red_cube_m is None or dist_to_blue_cube_m is None:
            reasons.append("missing_probe_distance")
        elif dist_to_red_cube_m + closer_margin_m < dist_to_blue_cube_m:
            analyzable = True
            violated = True
            reasons.append(
                "VIOLATION: Closer to red cube at probe frame "
                f"(red={dist_to_red_cube_m:.4f}m, blue={dist_to_blue_cube_m:.4f}m, margin={closer_margin_m:.4f}m)"
            )
        else:
            analyzable = True
            reasons.append("Blue_Cube_Selected_Or_Not_Closer_To_Red")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "source_success": source_success,
            "followup_success": followup_success,
            "followup_grasp_frame_index": grasp_frame_index,
            "followup_close_frame_index": close_frame_index,
            "followup_probe_frame_index": probe_frame_index,
            "followup_grasp_point": grasp_point,
            "red_cube_center": red_cube_center,
            "blue_cube_center": blue_cube_center,
            "dist_to_red_cube_m": dist_to_red_cube_m,
            "dist_to_blue_cube_m": dist_to_blue_cube_m,
            "closer_margin_m": closer_margin_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-DCRB1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "closer_margin_m": closer_margin_m,
            "dual_success_is_violation": True,
        },
        "details": details,
    }
