from typing import Any, Optional, Tuple

import numpy as np


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _nested_get(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return cur


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


def _infer_grasp_index_from_signals(
    record: Any,
    *,
    close_cmd_thresh: float = -0.95,
    closed_width_thresh: float = 0.003,
    lookahead: int = 6,
) -> Optional[int]:
    gripper_width = _first_not_none(
        _nested_get(record, "gripper_width"),
        _nested_get(record, "trajectory", "gripper_width"),
    )
    gripper_action_cmd = _first_not_none(
        _nested_get(record, "gripper_action_cmd"),
        _nested_get(record, "trajectory", "gripper_action_cmd"),
    )
    gripper_finger_qpos = _first_not_none(
        _nested_get(record, "gripper_finger_qpos"),
        _nested_get(record, "trajectory", "gripper_finger_qpos"),
    )

    widths = _to_1d_float_array(gripper_width)
    actions = _to_1d_float_array(gripper_action_cmd)

    if widths is None and gripper_finger_qpos is not None:
        q = np.asarray(gripper_finger_qpos, dtype=np.float64)
        if q.ndim >= 2 and q.shape[-1] >= 2:
            widths = np.sum(q[..., :2], axis=-1).reshape(-1)

    close_mask = None if widths is None else (widths <= closed_width_thresh)
    cmd_close_mask = None if actions is None else (actions <= close_cmd_thresh)

    if cmd_close_mask is not None and close_mask is not None:
        n = min(cmd_close_mask.shape[0], close_mask.shape[0])
        for i in range(n):
            if not cmd_close_mask[i]:
                continue
            j = min(i + lookahead, n - 1)
            if bool(np.any(close_mask[i : j + 1])):
                return int(i)

    if cmd_close_mask is not None:
        idx = np.where(cmd_close_mask)[0]
        if idx.size > 0:
            return int(idx[0])

    if close_mask is not None:
        idx = np.where(close_mask)[0]
        if idx.size > 0:
            return int(idx[0])

    return None


def _grasp_index_with_source(record: Any) -> Tuple[Optional[int], str]:
    annotated = _first_not_none(
        getattr(record, "derived_grasp_frame_index", None),
        getattr(record, "mr_eval_grasp_frame_index", None),
        _nested_get(record, "mr_eval", "grasp_frame_index"),
    )
    if annotated is not None:
        return int(annotated), "annotated"

    inferred = _infer_grasp_index_from_signals(record)
    if inferred is not None:
        return inferred, "inferred_from_gripper_signal"

    return None, "missing"
