from typing import List, Optional, Sequence, Tuple, Any

import numpy as np


def _xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def detect_grasp_event_index(
    gripper_cmd: Sequence[Optional[float]],
    gripper_width: Sequence[Optional[float]],
    eef_path: Sequence[Any],
    cube_pos: Sequence[Any],
    *,
    open_width_thresh: float = 0.075,
    close_width_thresh: float = 0.050,
    min_width_drop: float = 0.015,
    cmd_close_thresh: float = -0.02,
    proximity_thresh: float = 0.05,
    lookahead_frames: int = 5,
    min_obj_lift: float = 0.01,
) -> Optional[int]:
    prev_width = None
    was_open = False
    first_intent_idx = None
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

        if was_open and prev_width is not None:
            eef_xyz = _xyz(eef_path[i])
            cube_xyz = _xyz(cube_pos[i])
            is_valid_attempt = (
                cmd_val is not None
                and cmd_val <= cmd_close_thresh
                and (float(prev_width) - width) >= min_width_drop
                and width <= close_width_thresh
                and eef_xyz is not None
                and cube_xyz is not None
                and float(np.linalg.norm(eef_xyz - cube_xyz)) <= proximity_thresh
            )
            if is_valid_attempt:
                if first_intent_idx is None:
                    first_intent_idx = int(i)
                future_idx = i + lookahead_frames
                if future_idx < n:
                    future_cube_xyz = _xyz(cube_pos[future_idx])
                    if future_cube_xyz is not None and float(future_cube_xyz[2] - cube_xyz[2]) > min_obj_lift:
                        return int(i)

        prev_width = width

    return first_intent_idx


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
