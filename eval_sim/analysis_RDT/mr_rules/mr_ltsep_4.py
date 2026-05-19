from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_sadp_1 import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


LTSEP4_CONTINUE_RATIO = 0.3
LTSEP4_LIFT_Z_THRESHOLD = 0.05


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


def _cube_positions(record: Any) -> List[List[float]]:
    cube_pos = _first_not_none(
        _nested_get(record, "cube_pos"),
        _nested_get(record, "trajectory", "cube_pos"),
        _nested_get(record, "mr_eval", "cube_pos"),
        getattr(record, "cube_pos", None),
    ) or []
    points: List[List[float]] = []
    for p in cube_pos:
        try:
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _hide_frame_with_source(record: Any, *, lift_z_threshold: float) -> Tuple[Optional[int], str]:
    raw = _first_not_none(
        _nested_get(record, "cube_invisible_frame"),
        _nested_get(record, "invisible_trigger_frame"),
        _nested_get(record, "hide_frame"),
        _nested_get(record, "object_invisible_frame"),
        _nested_get(record, "mr_eval", "cube_invisible_frame"),
        _nested_get(record, "mr_eval", "invisible_trigger_frame"),
        getattr(record, "cube_invisible_frame", None),
        getattr(record, "invisible_trigger_frame", None),
    )
    if raw is not None:
        try:
            return int(raw), "annotated_invisible_event"
        except Exception:
            pass

    cube_pos = _cube_positions(record)
    for idx, p in enumerate(cube_pos):
        if len(p) >= 3 and p[2] > lift_z_threshold:
            return idx, "derived_cube_lift_above_threshold"

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


def _analyze_ltsep4(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    continue_ratio: float,
    lift_z_threshold: float,
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
        cube_pos_base = _cube_positions(base)
        hide_frame, hide_frame_source = _hide_frame_with_source(base, lift_z_threshold=lift_z_threshold)

        hide_point = None
        if hide_frame is not None and 0 <= hide_frame < len(cube_pos_base):
            hide_point = cube_pos_base[hide_frame]

        final_point = traj[-1] if traj else None

        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if hide_frame is None:
            analyzable = False
            reasons.append("missing_invisible_trigger_frame")
        if hide_point is None:
            analyzable = False
            reasons.append("missing_hide_point")
        if final_point is None:
            analyzable = False
            reasons.append("missing_trajectory")

        dist_rem = _point_l2(goal_point, hide_point)
        dist_actual = _point_l2(final_point, hide_point)

        response_ratio = None
        is_continued = None
        if analyzable:
            if dist_rem is None or dist_rem <= 1e-8 or dist_actual is None:
                analyzable = False
                reasons.append("cannot_compute_response_ratio")
            else:
                response_ratio = float(dist_actual / dist_rem)
                is_continued = response_ratio > continue_ratio
                if not is_continued:
                    violated = True
                    reasons.append(f"stops_or_diverges_after_invisible({response_ratio:.4f} < {continue_ratio:.4f})")
                else:
                    reasons.append("continues_transport_after_invisible")

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
                "invisible_frame": hide_frame,
                "invisible_frame_source": hide_frame_source,
                "invisible_point": hide_point,
                "goal_point": goal_point,
                "final_point": final_point,
                "dist_rem_m": dist_rem,
                "dist_actual_m": dist_actual,
                "response_ratio": response_ratio,
                "continue_ratio": continue_ratio,
                "lift_z_threshold_m": lift_z_threshold,
                "is_continued": is_continued,
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
            "continue_ratio": continue_ratio,
            "lift_z_threshold_m": lift_z_threshold,
            "invariance_proxy": "post_invisible_continue_transport",
        },
        "details": details,
    }


@register_mr_rule("MR-LTSEP-4")
def analyze_mr_ltsep_4(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_ltsep4(
        base_records,
        mr_records,
        mr_id="MR-LTSEP-4",
        continue_ratio=float(kwargs.get("continue_ratio", LTSEP4_CONTINUE_RATIO)),
        lift_z_threshold=float(kwargs.get("lift_z_threshold", LTSEP4_LIFT_Z_THRESHOLD)),
    )
