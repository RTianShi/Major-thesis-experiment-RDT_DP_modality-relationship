from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_sadp_1 import _first_not_none, _nested_get
from .registry import register_mr_rule


FPDP2_TRIGGER_DISTANCE_M = 0.05
FPDP2_SUCCESS_GOAL_THRESH_M = 0.025
FPDP2_MIN_POST_TRIGGER_STD_M = 0.005
FPDP2_MIN_POST_TRIGGER_RANGE_M = 0.015
FPDP2_MAX_FINAL_GOAL_DIST_M = 0.10


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
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _goal_point(record: Any) -> Optional[List[float]]:
    return _to_xyz(
        _first_not_none(
            _nested_get(record, "goal_point"),
            _nested_get(record, "mr_eval", "goal_point"),
            _nested_get(record, "trajectory", "goal_point"),
        )
    )


def _cube_positions(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "cube_pos"),
        _nested_get(record, "trajectory", "cube_pos"),
        _nested_get(record, "mr_eval", "cube_pos"),
        getattr(record, "cube_pos", None),
    ) or []
    points: List[List[float]] = []
    for p in traj:
        xyz = _to_xyz(p)
        if xyz is not None:
            points.append(xyz)
    return points


def _point_l2(a: Any, b: Any) -> Optional[float]:
    pa = _to_xyz(a)
    pb = _to_xyz(b)
    if pa is None or pb is None:
        return None
    return float(np.linalg.norm(np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64)))


def _distance_series(cube_positions: List[List[float]], goal_point: Optional[List[float]]) -> List[float]:
    if goal_point is None:
        return []
    out: List[float] = []
    for p in cube_positions:
        dist = _point_l2(p, goal_point)
        if dist is not None:
            out.append(float(dist))
    return out


def _first_trigger(distances: List[float], trigger_distance_m: float) -> Tuple[Optional[int], Optional[float], str]:
    for idx, dist in enumerate(distances):
        if float(dist) < trigger_distance_m:
            return idx, float(dist), "derived_from_cube_goal_distance"
    return None, None, "not_triggered"


def _analyze_fpdp2(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    trigger_distance_m: float,
    success_goal_thresh_m: float,
    min_post_trigger_std_m: float,
    min_post_trigger_range_m: float,
    max_final_goal_dist_m: float,
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

        goal_point = _first_not_none(_goal_point(base), _goal_point(mr))
        base_cube_pos = _cube_positions(base)
        mr_cube_pos = _cube_positions(mr)
        base_distances = _distance_series(base_cube_pos, goal_point)
        mr_distances = _distance_series(mr_cube_pos, goal_point)

        base_trigger_frame, base_trigger_dist, base_trigger_source = _first_trigger(
            base_distances, trigger_distance_m
        )
        mr_trigger_frame, mr_trigger_dist, mr_trigger_source = _first_trigger(
            mr_distances, trigger_distance_m
        )

        mr_post_trigger_distances = mr_distances[mr_trigger_frame:] if mr_trigger_frame is not None else []
        mr_final_goal_dist_m = mr_distances[-1] if mr_distances else None
        base_final_goal_dist_m = base_distances[-1] if base_distances else None

        post_trigger_std_m = None
        post_trigger_range_m = None
        if mr_post_trigger_distances:
            arr = np.asarray(mr_post_trigger_distances, dtype=np.float64)
            post_trigger_std_m = float(arr.std())
            post_trigger_range_m = float(arr.max() - arr.min())

        analyzable = True
        violated = False
        reasons = []

      
        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if base_trigger_frame is None:
            analyzable = False
            reasons.append("baseline_terminal_region_not_reached")

        if analyzable:
            if mr_trigger_frame is None:
                violated = True
                reasons.append("followup_never_reaches_terminal_region")
            else:
                if dst_success is True:
                    violated = True
                    reasons.append("terminal_state_bias_should_reduce_success_rate")

                if mr_final_goal_dist_m is not None and mr_final_goal_dist_m <= success_goal_thresh_m:
                    violated = True
                    reasons.append(
                        f"still_stabilizes_within_success_threshold({mr_final_goal_dist_m:.4f}m <= {success_goal_thresh_m:.4f}m)"
                    )

                if mr_final_goal_dist_m is not None and mr_final_goal_dist_m > max_final_goal_dist_m:
                    violated = True
                    reasons.append(
                        f"failure_mode_not_near_goal_oscillation({mr_final_goal_dist_m:.4f}m > {max_final_goal_dist_m:.4f}m)"
                    )

                if post_trigger_std_m is None or post_trigger_std_m < min_post_trigger_std_m:
                    violated = True
                    reasons.append(
                        f"post_trigger_distance_std_too_small({post_trigger_std_m if post_trigger_std_m is not None else 'None'} < {min_post_trigger_std_m:.4f}m)"
                    )

                if post_trigger_range_m is None or post_trigger_range_m < min_post_trigger_range_m:
                    violated = True
                    reasons.append(
                        f"post_trigger_distance_range_too_small({post_trigger_range_m if post_trigger_range_m is not None else 'None'} < {min_post_trigger_range_m:.4f}m)"
                    )

                if not violated:
                    reasons.append("near_goal_overcorrection_and_failure_to_settle")

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
                "goal_point": goal_point,
                "base_trigger_frame": base_trigger_frame,
                "base_trigger_distance_m": base_trigger_dist,
                "base_trigger_source": base_trigger_source,
                "mr_trigger_frame": mr_trigger_frame,
                "mr_trigger_distance_m": mr_trigger_dist,
                "mr_trigger_source": mr_trigger_source,
                "base_final_goal_dist_m": base_final_goal_dist_m,
                "mr_final_goal_dist_m": mr_final_goal_dist_m,
                "post_trigger_std_m": post_trigger_std_m,
                "post_trigger_range_m": post_trigger_range_m,
                "trigger_distance_m": trigger_distance_m,
                "success_goal_thresh_m": success_goal_thresh_m,
                "min_post_trigger_std_m": min_post_trigger_std_m,
                "min_post_trigger_range_m": min_post_trigger_range_m,
                "max_final_goal_dist_m": max_final_goal_dist_m,
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
            "trigger_distance_m": trigger_distance_m,
            "success_goal_thresh_m": success_goal_thresh_m,
            "min_post_trigger_std_m": min_post_trigger_std_m,
            "min_post_trigger_range_m": min_post_trigger_range_m,
            "max_final_goal_dist_m": max_final_goal_dist_m,
            "invariance_proxy": "terminal_proprio_bias_should_cause_near_goal_oscillatory_failure",
        },
        "details": details,
    }


@register_mr_rule("MR-FPDP-2")
@register_mr_rule("MR-FPDP2")
def analyze_mr_fpdp_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_fpdp2(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-FPDP-2"),
        trigger_distance_m=float(kwargs.get("trigger_distance_m", FPDP2_TRIGGER_DISTANCE_M)),
        success_goal_thresh_m=float(kwargs.get("success_goal_thresh_m", FPDP2_SUCCESS_GOAL_THRESH_M)),
        min_post_trigger_std_m=float(kwargs.get("min_post_trigger_std_m", FPDP2_MIN_POST_TRIGGER_STD_M)),
        min_post_trigger_range_m=float(kwargs.get("min_post_trigger_range_m", FPDP2_MIN_POST_TRIGGER_RANGE_M)),
        max_final_goal_dist_m=float(kwargs.get("max_final_goal_dist_m", FPDP2_MAX_FINAL_GOAL_DIST_M)),
    )
