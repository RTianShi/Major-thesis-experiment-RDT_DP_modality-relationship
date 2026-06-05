from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


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


def _table_height(record: Any, traj: List[List[float]]) -> Optional[float]:
    table_candidates = (
        _nested_get(record, "trajectory", "table_height"),
        _nested_get(record, "trajectory", "table_z"),
        getattr(record, "table_height", None),
        getattr(record, "table_z", None),
        _nested_get(record, "mr_eval", "table_height"),
        _nested_get(record, "mr_eval", "table_z"),
    )
    for value in table_candidates:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue

    if traj:
        arr = np.asarray(traj, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return float(np.min(arr[:, 2]))
    return None


def _quat_to_release_angle_deg(quat: Any) -> Optional[float]:
    if quat is None:
        return None
    try:
        q = np.asarray(quat, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if q.size < 4:
        return None
    q = q[:4]
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return None
    w, x, y, z = q / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    local_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_normal = rot @ local_normal
    cos_theta = float(np.clip(abs(world_normal[2]), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


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


def _grasp_and_release_points(record: Any) -> Dict[str, Any]:
    traj = _trajectory_points(record)
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    eef_quat_seq = _first_not_none(
        _nested_get(record, "trajectory", "eef_quat"),
        getattr(record, "eef_quat", None),
    ) or []
    if grasp_idx is None or not traj:
        return {
            "grasp_frame_index": grasp_idx,
            "grasp_frame_index_source": grasp_idx_source,
            "grasp_point": None,
            "release_hover_point": None,
            "table_height": _table_height(record, traj),
            "delta_z_max_m": None,
            "release_frame_index": None,
            "release_angle_deg": None,
        }

    grasp_point = _xyz(traj[grasp_idx]) if 0 <= grasp_idx < len(traj) else None
    release_hover_point = None
    if grasp_idx + 1 < len(traj):
        post_grasp = [_xyz(p) for p in traj[grasp_idx + 1 :] if _xyz(p) is not None]
        if post_grasp:
            release_hover_point = max(post_grasp, key=lambda point: float(point[2]))
    table_height = _table_height(record, traj)
    delta_z_max = None
    if table_height is not None and traj:
        arr = np.asarray(traj, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            delta_z_max = float(np.max(arr[:, 2]) - table_height)
    release_frame_index = None
    release_angle_deg = None
    gripper_cmd = _first_not_none(
        _nested_get(record, "trajectory", "gripper_action_cmd"),
        getattr(record, "gripper_action_cmd", None),
    ) or []
    for idx in range(max(0, grasp_idx + 1), len(gripper_cmd)):
        try:
            value = float(gripper_cmd[idx])
        except Exception:
            continue
        if value >= 0.95:
            release_frame_index = idx
            break
    if release_frame_index is None and len(gripper_cmd) > grasp_idx + 1:
        release_frame_index = len(gripper_cmd) - 1
    if release_frame_index is not None and 0 <= release_frame_index < len(eef_quat_seq):
        release_angle_deg = _quat_to_release_angle_deg(eef_quat_seq[release_frame_index])

    return {
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
        "grasp_point": grasp_point,
        "release_hover_point": release_hover_point,
        "table_height": table_height,
        "delta_z_max_m": delta_z_max,
        "release_frame_index": release_frame_index,
        "release_angle_deg": release_angle_deg,
    }


@register_mr_rule("MR5")
@register_mr_rule("MR-5")
@register_mr_rule("Instruction-Specialization")
def analyze_mr5_instruction_specialization(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    max_release_angle_deg = float(kwargs.get("max_release_angle_deg", 10.0))

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]
        binfo = _grasp_and_release_points(base)
        minfo = _grasp_and_release_points(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if binfo["grasp_point"] is None:
            analyzable = False
            reasons.append("missing_base_grasp_point")
        if minfo["grasp_point"] is None:
            analyzable = False
            reasons.append("missing_mr_grasp_point")
        if minfo["release_angle_deg"] is None:
            analyzable = False
            reasons.append("missing_mr_release_angle_deg")
        if analyzable:
            if float(minfo["release_angle_deg"]) > max_release_angle_deg:
                violated = True
                reasons.append("VIOLATION: release angle to world Z exceeds 10 degrees")
            else:
                reasons.append("instruction_specialization_release_angle_pass")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "base_success": getattr(base, "success", None),
            "mr_success": getattr(mr, "success", None),
            "base_grasp_frame_index": binfo["grasp_frame_index"],
            "mr_grasp_frame_index": minfo["grasp_frame_index"],
            "base_grasp_frame_index_source": binfo["grasp_frame_index_source"],
            "mr_grasp_frame_index_source": minfo["grasp_frame_index_source"],
            "base_grasp_point": None if binfo["grasp_point"] is None else binfo["grasp_point"].tolist(),
            "mr_grasp_point": None if minfo["grasp_point"] is None else minfo["grasp_point"].tolist(),
            "mr_release_hover_point": None if minfo["release_hover_point"] is None else minfo["release_hover_point"].tolist(),
            "mr_release_frame_index": minfo["release_frame_index"],
            "mr_release_angle_deg": minfo["release_angle_deg"],
            "max_release_angle_deg": max_release_angle_deg,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": kwargs.get("mr_id", "MR5"),
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "metric": "release_angle_to_world_z_deg",
            "violation_rule": "theta > max_release_angle_deg",
            "max_release_angle_deg": max_release_angle_deg,
            "expectation": "specialized_instruction_should_keep_release_orientation_near_vertical",
        },
        "details": details,
    }
