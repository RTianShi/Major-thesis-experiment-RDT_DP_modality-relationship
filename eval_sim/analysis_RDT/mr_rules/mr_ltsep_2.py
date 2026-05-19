from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_sadp_1 import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


LTSEP2_STOP_RATIO = 0.3


def _goal_point(record: Any) -> Optional[List[float]]:
    goal = _first_not_none(
        _nested_get(record, "goal_point"),
        _nested_get(record, "mr_eval", "goal_point"),
        _nested_get(record, "trajectory", "goal_point"),
    )
    if goal is None:
        return None
    try:
        arr = np.asarray(goal, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory", "eef_path"),
        _nested_get(record, "trajectory_after_phantom_grasp"),
        _nested_get(record, "mr_eval", "trajectory_after_phantom_grasp"),
        getattr(record, "eef_path", None),
        _nested_get(record, "trajectory"),
    ) or []
    if isinstance(traj, dict):
        traj = traj.get("eef_path") or []
    points: List[List[float]] = []
    for p in traj:
        try:
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _disappear_frame_with_source(record: Any) -> Tuple[Optional[int], str]:
    raw = _first_not_none(
        _nested_get(record, "cube_disappear_frame"),
        _nested_get(record, "empty_grasp_trigger_frame"),
        _nested_get(record, "object_disappear_frame"),
        _nested_get(record, "mr_eval", "cube_disappear_frame"),
        _nested_get(record, "mr_eval", "empty_grasp_trigger_frame"),
        getattr(record, "cube_disappear_frame", None),
        getattr(record, "empty_grasp_trigger_frame", None),
    )
    if raw is not None:
        try:
            return int(raw), "annotated_disappear_event"
        except Exception:
            pass

    # 兼容场景脚本未显式记录消失帧的情况：退回到抓取建立帧。
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    if grasp_idx is not None:
        return int(grasp_idx), f"fallback_{grasp_idx_source}"

    return None, "missing"


def _point_l2(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _analyze_ltsep2(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    stop_ratio: float,
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

        src_success = getattr(base, "success", None)
        dst_success = getattr(mr, "success", None)

        goal_point = _goal_point(base) or _goal_point(mr)
        traj = _trajectory_points(mr)
        disappear_frame, disappear_frame_source = _disappear_frame_with_source(mr)

        disappear_point = None
        if disappear_frame is not None and 0 <= disappear_frame < len(traj):
            disappear_point = traj[disappear_frame]

        final_point = traj[-1] if traj else None

        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if disappear_frame is None:
            analyzable = False
            reasons.append("missing_disappear_frame")
        if disappear_point is None:
            analyzable = False
            reasons.append("missing_disappear_point")
        if final_point is None:
            analyzable = False
            reasons.append("missing_trajectory")

        dist_rem = _point_l2(goal_point, disappear_point)
        dist_actual = _point_l2(final_point, disappear_point)

        response_ratio = None
        is_halted = None
        if analyzable:
            if dist_rem is None or dist_rem <= 1e-8 or dist_actual is None:
                analyzable = False
                reasons.append("cannot_compute_response_ratio")
            else:
                response_ratio = float(dist_actual / dist_rem)
                is_halted = response_ratio < stop_ratio
                if not is_halted:
                    violated = True
                    reasons.append(f"continues_transport({response_ratio:.4f} >= {stop_ratio:.4f})")
                else:
                    reasons.append("deviated_or_stopped_after_disappearance")

        if not analyzable:
            unavailable_count += 1
            violated = False

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "src_success": src_success,
                "dst_success": dst_success,
                "disappear_frame": disappear_frame,
                "disappear_frame_source": disappear_frame_source,
                "disappear_point": disappear_point,
                "goal_point": goal_point,
                "final_point": final_point,
                "dist_rem_m": dist_rem,
                "dist_actual_m": dist_actual,
                "response_ratio": response_ratio,
                "stop_ratio": stop_ratio,
                "is_halted": is_halted,
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
            "stop_ratio": stop_ratio,
            "invariance_proxy": "post_disappearance_stop_response",
        },
        "details": details,
    }


@register_mr_rule("MR-LTSEP-2")
def analyze_mr_ltsep_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_ltsep2(
        base_records,
        mr_records,
        mr_id="MR-LTSEP-2",
        stop_ratio=float(kwargs.get("stop_ratio", LTSEP2_STOP_RATIO)),
    )
