from typing import Any, Dict, List, Optional

import numpy as np

from .mr_sadp_1 import _first_not_none, _nested_get
from .registry import register_mr_rule


FPDP1_SUCCESS_GOAL_THRESH_M = 0.025
FPDP1_MIN_TERMINAL_ALIGNMENT_ERROR_M = 0.04
FPDP1_MIN_ERROR_INCREASE_M = 0.02
FPDP1_NEAR_GOAL_UPPER_BOUND_M = 0.15


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


def _analyze_fpdp1(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    success_goal_thresh_m: float,
    min_terminal_alignment_error_m: float,
    min_error_increase_m: float,
    near_goal_upper_bound_m: float,
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

        base_final_cube = base_cube_pos[-1] if base_cube_pos else None
        mr_final_cube = mr_cube_pos[-1] if mr_cube_pos else None
        base_final_goal_dist_m = _point_l2(base_final_cube, goal_point)
        mr_final_goal_dist_m = _point_l2(mr_final_cube, goal_point)

        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if base_final_cube is None:
            analyzable = False
            reasons.append("missing_base_final_cube")
        if mr_final_cube is None:
            analyzable = False
            reasons.append("missing_mr_final_cube")

        error_increase_m = None
        if analyzable:
            if base_final_goal_dist_m is None or mr_final_goal_dist_m is None:
                analyzable = False
                reasons.append("missing_terminal_alignment_error")
            else:
                error_increase_m = float(mr_final_goal_dist_m - base_final_goal_dist_m)

                if dst_success is True:
                    violated = True
                    reasons.append("high_frequency_visual_degradation_should_break_success_threshold")

                if mr_final_goal_dist_m < min_terminal_alignment_error_m:
                    violated = True
                    reasons.append(
                        f"terminal_alignment_error_not_large_enough({mr_final_goal_dist_m:.4f}m < {min_terminal_alignment_error_m:.4f}m)"
                    )

                if error_increase_m < min_error_increase_m:
                    violated = True
                    reasons.append(
                        f"terminal_error_increase_too_small({error_increase_m:.4f}m < {min_error_increase_m:.4f}m)"
                    )

                if mr_final_goal_dist_m > near_goal_upper_bound_m:
                    violated = True
                    reasons.append(
                        f"degradation_exceeds_near_goal_band({mr_final_goal_dist_m:.4f}m > {near_goal_upper_bound_m:.4f}m)"
                    )

                if mr_final_goal_dist_m <= success_goal_thresh_m:
                    violated = True
                    reasons.append(
                        f"still_within_success_threshold({mr_final_goal_dist_m:.4f}m <= {success_goal_thresh_m:.4f}m)"
                    )

                if not violated:
                    reasons.append("coarse_localization_preserved_but_terminal_alignment_broken")

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
                "base_final_cube": base_final_cube,
                "mr_final_cube": mr_final_cube,
                "base_final_goal_dist_m": base_final_goal_dist_m,
                "mr_final_goal_dist_m": mr_final_goal_dist_m,
                "error_increase_m": error_increase_m,
                "success_goal_thresh_m": success_goal_thresh_m,
                "min_terminal_alignment_error_m": min_terminal_alignment_error_m,
                "min_error_increase_m": min_error_increase_m,
                "near_goal_upper_bound_m": near_goal_upper_bound_m,
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
            "success_goal_thresh_m": success_goal_thresh_m,
            "min_terminal_alignment_error_m": min_terminal_alignment_error_m,
            "min_error_increase_m": min_error_increase_m,
            "near_goal_upper_bound_m": near_goal_upper_bound_m,
            "invariance_proxy": "high_frequency_visual_loss_should_preserve_coarse_goaling_but_break_terminal_alignment",
        },
        "details": details,
    }


@register_mr_rule("MR-FPDP-1")
@register_mr_rule("MR-FPDP1")
def analyze_mr_fpdp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_fpdp1(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-FPDP-1"),
        success_goal_thresh_m=float(kwargs.get("success_goal_thresh_m", FPDP1_SUCCESS_GOAL_THRESH_M)),
        min_terminal_alignment_error_m=float(
            kwargs.get("min_terminal_alignment_error_m", FPDP1_MIN_TERMINAL_ALIGNMENT_ERROR_M)
        ),
        min_error_increase_m=float(kwargs.get("min_error_increase_m", FPDP1_MIN_ERROR_INCREASE_M)),
        near_goal_upper_bound_m=float(kwargs.get("near_goal_upper_bound_m", FPDP1_NEAR_GOAL_UPPER_BOUND_M)),
    )
