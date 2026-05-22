from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SESP2_TORQUE_RATIO_THRESHOLD = 1.15
SESP2_VELOCITY_RATIO_THRESHOLD = 0.95
SESP2_TORQUE_STD_RATIO_THRESHOLD = 2.0
SESP2_PATH_LENGTH_RATIO_THRESHOLD = 1.3


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _safe_xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return arr[:3]


def _safe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _safe_vec_list(seq: Any) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = []
    for item in seq or []:
        out.append(_safe_xyz(item))
    return out


def _safe_norm(vec: Any) -> Optional[float]:
    arr = _safe_xyz(vec)
    if arr is None:
        return None
    return float(np.linalg.norm(arr))


def _cube_points(record: Any) -> List[np.ndarray]:
    seq = getattr(record, "cube_pos", None) or getattr(record, "src_cube_pos", None) or []
    points: List[np.ndarray] = []
    for item in seq:
        arr = _safe_xyz(item)
        if arr is not None:
            points.append(arr)
    return points


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _window_slice(values: List[Any], start: Optional[int], end: Optional[int]) -> List[Any]:
    if start is None:
        start = 0
    if end is None:
        end = len(values)
    start = max(0, int(start))
    end = max(start, int(end))
    return list(values[start:end])


def _contact_frame_index(record: Any) -> Optional[int]:
    mr_eval = getattr(record, "mr_eval", None)
    if isinstance(mr_eval, dict):
        idx = mr_eval.get("contact_frame_index")
        if idx is not None:
            try:
                return int(idx)
            except Exception:
                pass
    idx = getattr(record, "mr_eval_contact_frame_index", None)
    try:
        if idx is not None:
            return int(idx)
    except Exception:
        pass

    is_contact = getattr(record, "is_contact", None) or []
    for idx, flag in enumerate(is_contact):
        if flag is True:
            return idx

    cube_points = _cube_points(record)
    if len(cube_points) < 2:
        return None
    start = cube_points[0]
    for idx, point in enumerate(cube_points[1:], start=1):
        if float(np.linalg.norm(point - start)) > 0.002:
            return idx
    return None


def _trajectory_total_steps(record: Any) -> Optional[int]:
    value = getattr(record, "total_steps", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _trajectory_total_path_length(record: Any) -> Optional[float]:
    value = getattr(record, "path_len", None)
    if value is None:
        value = getattr(record, "total_path_length_meters", None)
    return _safe_float(value)


def _torque_and_velocity_window(record: Any) -> Dict[str, Any]:
    contact_idx = _contact_frame_index(record)
    total_steps = _trajectory_total_steps(record)

    joint_torques = list(getattr(record, "joint_torques", None) or [])
    eef_vel = list(getattr(record, "eef_vel", None) or [])

    end = total_steps if total_steps is not None else max(len(joint_torques), len(eef_vel))
    torque_window = _window_slice(joint_torques, contact_idx, end)
    vel_window = _window_slice(eef_vel, contact_idx, end)

    torque_norms = [float(np.linalg.norm(v)) for v in (_safe_xyz(item) for item in torque_window) if v is not None]
    vel_norms = [float(np.linalg.norm(v)) for v in (_safe_xyz(item) for item in vel_window) if v is not None]

    return {
        "contact_frame_index": contact_idx,
        "total_steps": total_steps,
        "torque_norms": torque_norms,
        "velocity_norms": vel_norms,
        "mean_torque": _mean_std(torque_norms)[0],
        "std_torque": _mean_std(torque_norms)[1],
        "mean_velocity": _mean_std(vel_norms)[0],
        "std_velocity": _mean_std(vel_norms)[1],
    }


@register_mr_rule("MR-SESP2")
@register_mr_rule("MR-SESP-2")
@register_mr_rule("mr_sesp_2")
def analyze_mr_sesp2_physical_semantic_synergy(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    torque_ratio_threshold = float(kwargs.get("torque_ratio_threshold", SESP2_TORQUE_RATIO_THRESHOLD))
    velocity_ratio_threshold = float(kwargs.get("velocity_ratio_threshold", SESP2_VELOCITY_RATIO_THRESHOLD))
    torque_std_ratio_threshold = float(kwargs.get("torque_std_ratio_threshold", SESP2_TORQUE_STD_RATIO_THRESHOLD))
    path_length_ratio_threshold = float(kwargs.get("path_length_ratio_threshold", SESP2_PATH_LENGTH_RATIO_THRESHOLD))

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        analyzable = True
        violated = False
        reasons: List[str] = []

        base_win = _torque_and_velocity_window(base)
        mr_win = _torque_and_velocity_window(mr)

        base_mean_torque = base_win["mean_torque"]
        mr_mean_torque = mr_win["mean_torque"]
        base_mean_velocity = base_win["mean_velocity"]
        mr_mean_velocity = mr_win["mean_velocity"]
        base_std_torque = base_win["std_torque"]
        mr_std_torque = mr_win["std_torque"]

        base_path_len = _trajectory_total_path_length(base)
        mr_path_len = _trajectory_total_path_length(mr)

        if base_win["contact_frame_index"] is None:
            analyzable = False
            reasons.append("missing_base_contact_frame_index")
        if mr_win["contact_frame_index"] is None:
            analyzable = False
            reasons.append("missing_followup_contact_frame_index")

        if not getattr(base, "joint_torques", None):
            analyzable = False
            reasons.append("missing_base_joint_torques")
        if not getattr(mr, "joint_torques", None):
            analyzable = False
            reasons.append("missing_followup_joint_torques")
        if not getattr(base, "eef_vel", None):
            analyzable = False
            reasons.append("missing_base_eef_vel")
        if not getattr(mr, "eef_vel", None):
            analyzable = False
            reasons.append("missing_followup_eef_vel")
        if base_path_len is None:
            analyzable = False
            reasons.append("missing_base_total_path_length_meters")
        if mr_path_len is None:
            analyzable = False
            reasons.append("missing_followup_total_path_length_meters")

        torque_ratio = None
        velocity_ratio = None
        torque_std_ratio = None
        path_length_ratio = None

        if analyzable:
            if base_mean_torque is None or mr_mean_torque is None:
                analyzable = False
                reasons.append("cannot_compute_mean_torque")
            else:
                torque_ratio = float(mr_mean_torque / max(base_mean_torque, 1e-12))
              

            if base_mean_velocity is None or mr_mean_velocity is None:
                analyzable = False
                reasons.append("cannot_compute_mean_velocity")
            else:
                velocity_ratio = float(mr_mean_velocity / max(base_mean_velocity, 1e-12))
                if velocity_ratio > velocity_ratio_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Velocity Adaptation Failure: Pushing speed too high for a heavy object."
                    )

            if base_std_torque is None or mr_std_torque is None:
                analyzable = False
                reasons.append("cannot_compute_torque_std")
            else:
                torque_std_ratio = float(mr_std_torque / max(base_std_torque, 1e-12))
                if torque_std_ratio > torque_std_ratio_threshold:
                    violated = True
                    reasons.append(
                        "VIOLATION: Dynamic Instability: Significant torque oscillation detected during heavy push."
                    )

            path_length_ratio = float(mr_path_len / max(base_path_len, 1e-12))
            if path_length_ratio > path_length_ratio_threshold:
                violated = True
                reasons.append(
                    "VIOLATION: Path Length Violation: Inefficient trajectory due to physical property change."
                )

            if not violated:
                reasons.append("physical_semantic_synergy_pass")

        if not analyzable:
            unavailable_count += 1
            violated = False
        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "contact_frame_index_base": base_win["contact_frame_index"],
                "contact_frame_index_followup": mr_win["contact_frame_index"],
                "mean_torque_base": base_mean_torque,
                "mean_torque_followup": mr_mean_torque,
                "mean_velocity_base": base_mean_velocity,
                "mean_velocity_followup": mr_mean_velocity,
                "std_torque_base": base_std_torque,
                "std_torque_followup": mr_std_torque,
                "path_length_base_m": base_path_len,
                "path_length_followup_m": mr_path_len,
                "torque_ratio": torque_ratio,
                "velocity_ratio": velocity_ratio,
                "torque_std_ratio": torque_std_ratio,
                "path_length_ratio": path_length_ratio,
                "torque_ratio_threshold": torque_ratio_threshold,
                "velocity_ratio_threshold": velocity_ratio_threshold,
                "torque_std_ratio_threshold": torque_std_ratio_threshold,
                "path_length_ratio_threshold": path_length_ratio_threshold,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SESP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "torque_ratio_threshold": torque_ratio_threshold,
            "velocity_ratio_threshold": velocity_ratio_threshold,
            "torque_std_ratio_threshold": torque_std_ratio_threshold,
            "path_length_ratio_threshold": path_length_ratio_threshold,
            "violation_rule": (
                "mean_torque_ratio < threshold OR mean_velocity_ratio > threshold OR "
                "torque_std_ratio > threshold OR path_length_ratio > threshold"
            ),
        },
        "details": details,
    }
