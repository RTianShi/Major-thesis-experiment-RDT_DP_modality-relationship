from typing import List, Optional, Sequence, Tuple, Any

import numpy as np


def _xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def detect_grasp_event_index_with_reason(
    gripper_cmd: Sequence[Optional[float]],
    gripper_width: Sequence[Optional[float]],
    eef_path: Sequence[Any],
    cube_pos: Sequence[Any],
    *,
    open_width_thresh: float = 0.075,
    close_width_thresh: float = 0.055,
    min_width_drop: float = 0.015,
    cmd_close_thresh: float = -0.02,
    proximity_thresh: float = 0.05,
    lookahead_frames: int = 5,
    min_obj_lift: float = 0.01,
    grasp_match_window: int = 3,
    grasp_mode: str = "strict",
    contact_proximity_thresh: float = 0.08,
) -> Tuple[Optional[int], Optional[str]]:
    if not gripper_cmd or not gripper_width or not eef_path or not cube_pos:
        return None, "insufficient_trajectory_data"

    contact_only = str(grasp_mode).strip().lower() in {"contact", "contact_only", "contact-only"}
    prev_width = None
    was_open = False
    first_intent_idx = None
    saw_valid_attempt = False
    n = min(len(gripper_cmd), len(gripper_width), len(eef_path), len(cube_pos))

    for i in range(n):
        cmd = gripper_cmd[i]
        width = gripper_width[i]
        if width is None:
            prev_width = width
            continue

        width = float(width)
        cmd_val = None if cmd is None else float(cmd)

        if width >= open_width_thresh:
            was_open = True

        eef_xyz = _xyz(eef_path[i])
        cube_xyz = _xyz(cube_pos[i])
        close_limit = contact_proximity_thresh if contact_only else proximity_thresh
        close_enough = (
            eef_xyz is not None
            and cube_xyz is not None
            and float(np.linalg.norm(eef_xyz - cube_xyz)) <= close_limit
        )

        if contact_only:
            if was_open and cmd_val is not None and cmd_val <= cmd_close_thresh and close_enough:
                return int(i), None
            prev_width = width
            continue

        # 在短窗口内寻找“有效近距离闭合”
        if was_open:
            window_start = max(0, i - max(1, int(grasp_match_window)) + 1)
            window_prev_widths = [float(w) for w in gripper_width[window_start:i] if w is not None]
            window_best_prev_width = max(window_prev_widths) if window_prev_widths else None

            accumulated_drop = (window_best_prev_width - width) if window_best_prev_width is not None else None
            is_valid_attempt = (
                cmd_val is not None
                and cmd_val <= cmd_close_thresh
                and width <= close_width_thresh
                and close_enough
                and accumulated_drop is not None
                and accumulated_drop >= min_width_drop
            )

            if is_valid_attempt:
                saw_valid_attempt = True
                if first_intent_idx is None:
                    first_intent_idx = int(i)

                future_idx = i + lookahead_frames
                if future_idx < n:
                    future_cube_xyz = _xyz(cube_pos[future_idx])
                    if future_cube_xyz is not None and float(future_cube_xyz[2] - cube_xyz[2]) > min_obj_lift:
                        return int(i), None

        prev_width = width

    if contact_only:
        if not was_open:
            return None, "never_opened_enough"
        return None, "no_contact_grasp_confirmed"

    if first_intent_idx is not None:
        return first_intent_idx, None
    if not was_open:
        return None, "never_opened_enough"
    if not saw_valid_attempt:
        return None, "no_valid_close_attempt"
    return None, "no_grasp_confirmed"


def detect_grasp_event_index(
    gripper_cmd: Sequence[Optional[float]],
    gripper_width: Sequence[Optional[float]],
    eef_path: Sequence[Any],
    cube_pos: Sequence[Any],
    **kwargs,
) -> Optional[int]:
    idx, _ = detect_grasp_event_index_with_reason(
        gripper_cmd,
        gripper_width,
        eef_path,
        cube_pos,
        **kwargs,
    )
    return idx


def derive_grasp_and_yaw_from_raw(
    gripper_cmd: Sequence[Optional[float]],
    gripper_width: Sequence[Optional[float]],
    eef_path: Sequence[Any],
    cube_pos: Sequence[Any],
    eef_yaw_deg: Sequence[Optional[float]],
    **kwargs,
) -> Tuple[Optional[int], Optional[float]]:
    idx = detect_grasp_event_index(
        gripper_cmd,
        gripper_width,
        eef_path,
        cube_pos,
        **kwargs,
    )
    yaw = None
    if idx is not None and 0 <= idx < len(eef_yaw_deg):
        yaw = eef_yaw_deg[idx]
    return idx, yaw
