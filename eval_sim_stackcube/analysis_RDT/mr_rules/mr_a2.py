from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_utils import _first_not_none, _nested_get
from .registry import register_mr_rule


MR_A2_FALSE_GRASP_DIST_THRESH = 0.1
MR_A2_CLOSE_CMD_THRESH = -0.95
MR_A2_OPEN_WIDTH_THRESH = 0.06
MR_A2_CLOSED_WIDTH_THRESH = 0.01
MR_A2_LIFT_Z_THRESH = 0.05


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _to_xyz(point: Any) -> Optional[List[float]]:
    if point is None:
        return None
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    xyz = [float(arr[0]), float(arr[1]), float(arr[2])]
    if not all(np.isfinite(v) for v in xyz):
        return None
    return xyz


def _eef_point_at(record: Any, idx: Optional[int]) -> Optional[List[float]]:
    eef_path = _first_not_none(
        getattr(record, "eef_path", None),
        _nested_get(record, "trajectory", "eef_path"),
    ) or []
    if idx is None or not (0 <= int(idx) < len(eef_path)):
        return None
    return _to_xyz(eef_path[int(idx)])


def _to_1d_float_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def _close_event_index(record: Any) -> Optional[int]:
    widths = _first_not_none(
        _nested_get(record, "trajectory", "gripper_width"),
        getattr(record, "gripper_width", None),
    )
    actions = _first_not_none(
        _nested_get(record, "trajectory", "gripper_action_cmd"),
        getattr(record, "gripper_action_cmd", None),
    )
    finger_qpos = _first_not_none(
        _nested_get(record, "trajectory", "gripper_finger_qpos"),
        getattr(record, "gripper_finger_qpos", None),
    )

    widths_arr = _to_1d_float_array(widths)
    actions_arr = _to_1d_float_array(actions)

    if widths_arr is None and finger_qpos is not None:
        q = np.asarray(finger_qpos, dtype=np.float64)
        if q.ndim >= 2 and q.shape[-1] >= 2:
            widths_arr = np.sum(q[..., :2], axis=-1).reshape(-1)

    if widths_arr is None and actions_arr is None:
        return None

    n = None
    if widths_arr is not None:
        n = widths_arr.shape[0]
    if actions_arr is not None:
        n = actions_arr.shape[0] if n is None else min(n, actions_arr.shape[0])

    if n is None or n <= 0:
        return None

    was_open = False
    for i in range(n):
        width = widths_arr[i] if widths_arr is not None else None
        cmd = actions_arr[i] if actions_arr is not None else None

        if width is not None and float(width) >= MR_A2_OPEN_WIDTH_THRESH:
            was_open = True

        closed_by_width = width is not None and float(width) <= MR_A2_CLOSED_WIDTH_THRESH
        closed_by_cmd = cmd is not None and float(cmd) <= MR_A2_CLOSE_CMD_THRESH
        if was_open and (closed_by_width or closed_by_cmd):
            return int(i)

    return None


def _goal_point(record: Any) -> Optional[List[float]]:
    return _first_not_none(
        _to_xyz(getattr(record, "goal_point", None)),
        _to_xyz(_nested_get(record, "mr_eval", "goal_point")),
        _to_xyz(_nested_get(record, "trajectory", "goal_point")),
        _to_xyz(_nested_get(record, "positions", "green_goal")),
        _to_xyz(_nested_get(record, "positions", "goal")),
    )


def _point_l2(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _analyze_a2_false_grasp(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    dist_thresh: float,
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

        close_idx = _close_event_index(mr)
        close_point = _eef_point_at(mr, close_idx)
        goal_point = _goal_point(mr) or _goal_point(base)
        lift_delta = None

        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")

        if close_idx is None:
            reasons.append("missing_close_event")
        elif close_point is None:
            reasons.append("missing_close_point")
        else:
            eef_path = _first_not_none(
                getattr(mr, "eef_path", None),
                _nested_get(mr, "trajectory", "eef_path"),
            ) or []
            if close_idx is not None and eef_path:
                after = [
                    _to_xyz(p)
                    for p in eef_path[int(close_idx) :]
                    if _to_xyz(p) is not None
                ]
                if after:
                    max_z = max(p[2] for p in after)
                    lift_delta = float(max_z - close_point[2])
                else:
                    reasons.append("missing_eef_path_after_close")

        dist = _point_l2(close_point, goal_point)
        if dist is None:
            reasons.append("cannot_compute_distance")

        if close_point is not None and dist is not None and lift_delta is not None:
            if dist > dist_thresh and lift_delta > MR_A2_LIFT_Z_THRESH:
                violated = True
                reasons.append(
                    f"blind_grasp_far_from_goal({dist:.4f} > {dist_thresh:.4f})"
                )
                reasons.append(
                    f"close_then_lift({lift_delta:.4f} > {MR_A2_LIFT_Z_THRESH:.4f})"
                )
            else:
                reasons.append("no_blind_grasp_detected")

        if (close_point is None and dist is None) or lift_delta is None:
            analyzable = False

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
                "distance_m": dist,
                "distance_thresh_m": dist_thresh,
                "close_frame_index": close_idx,
                "close_point": close_point,
                "eef_lift_delta_m": lift_delta,
                "eef_lift_thresh_m": MR_A2_LIFT_Z_THRESH,
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
            "distance_thresh_m": dist_thresh,
            "close_cmd_thresh": MR_A2_CLOSE_CMD_THRESH,
            "open_width_thresh": MR_A2_OPEN_WIDTH_THRESH,
            "closed_width_thresh": MR_A2_CLOSED_WIDTH_THRESH,
            "lift_z_thresh_m": MR_A2_LIFT_Z_THRESH,
            "metric": "dist(Pgrasp_f, Pgreen_sphere)",
            "violation_rule": "distance > distance_thresh_m and lift_delta > lift_z_thresh_m",
        },
        "details": details,
    }


@register_mr_rule("MR-A2")
@register_mr_rule("MR-A2-VIS-DEPRIV")
def analyze_mr_a2_false_grasp(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_a2_false_grasp(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-A2"),
        dist_thresh=float(kwargs.get("dist_thresh_m", MR_A2_FALSE_GRASP_DIST_THRESH)),
    )
