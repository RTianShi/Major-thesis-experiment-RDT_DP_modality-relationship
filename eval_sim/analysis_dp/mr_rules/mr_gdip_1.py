from typing import Any, Dict, List, Optional

import numpy as np

from .mr_sadp_1 import _first_not_none, _nested_get
from .registry import register_mr_rule


GDIP1_NEAR_GOAL_TOL_M = 0.05
GDIP1_CUBE_MOVE_MIN_M = 0.02


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


def _trajectory_points(record: Any, *field_paths: str) -> List[List[float]]:
    candidates = []
    for field_path in field_paths:
        candidates.append(_nested_get(record, *field_path.split(".")))
    traj = _first_not_none(*candidates, getattr(record, field_paths[-1].split(".")[-1], None)) or []

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


def _path_len(points: List[List[float]]) -> Optional[float]:
    if not points:
        return None
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return float(np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1).sum())


def _analyze_gdip1(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    near_goal_tol_m: float,
    cube_move_min_m: float,
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
        mr_cube_path = _trajectory_points(mr, "cube_pos", "trajectory.cube_pos", "mr_eval.cube_pos")
        mr_eef_path = _trajectory_points(mr, "eef_path", "trajectory.eef_path")

        mr_cube_start = mr_cube_path[0] if mr_cube_path else None
        mr_cube_final = mr_cube_path[-1] if mr_cube_path else None
        mr_eef_final = mr_eef_path[-1] if mr_eef_path else None

        mr_cube_goal_dist_m = _point_l2(mr_cube_final, goal_point)
        mr_eef_goal_dist_m = _point_l2(mr_eef_final, goal_point)
        mr_cube_net_disp_m = _point_l2(mr_cube_start, mr_cube_final)
        mr_eef_path_len_m = _path_len(mr_eef_path)

        analyzable = True
        violated = False
        reasons = []

        
        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")

        suspicious_goal_reached = False
        if analyzable:
            suspicious_goal_reached = (
                dst_success is True
                or (mr_cube_goal_dist_m is not None and mr_cube_goal_dist_m <= near_goal_tol_m)
            )

            if suspicious_goal_reached:
                violated = True
                if dst_success is True:
                    reasons.append("vision_deprivation_should_not_succeed")
                if mr_cube_goal_dist_m is not None and mr_cube_goal_dist_m <= near_goal_tol_m:
                    reasons.append(
                        f"cube_still_reaches_goal(cube_goal_dist={mr_cube_goal_dist_m:.4f}m <= {near_goal_tol_m:.4f}m)"
                    )
            else:
                reasons.append("expected_failure_under_global_visual_deprivation")

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
                "mr_cube_start": mr_cube_start,
                "mr_cube_final": mr_cube_final,
                "mr_eef_final": mr_eef_final,
                "mr_cube_goal_dist_m": mr_cube_goal_dist_m,
                "mr_eef_goal_dist_m": mr_eef_goal_dist_m,
                "mr_cube_net_disp_m": mr_cube_net_disp_m,
                "mr_eef_path_len_m": mr_eef_path_len_m,
                "near_goal_tol_m": near_goal_tol_m,
                "cube_move_min_m": cube_move_min_m,
                "cube_moved": (
                    None if mr_cube_net_disp_m is None else bool(mr_cube_net_disp_m >= cube_move_min_m)
                ),
                "suspicious_goal_reached": suspicious_goal_reached,
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
            "near_goal_tol_m": near_goal_tol_m,
            "cube_move_min_m": cube_move_min_m,
            "expected_followup_success_rate_percent": 0.0,
            "invariance_proxy": "global_vision_deprivation_should_block_object_localization",
        },
        "details": details,
    }


@register_mr_rule("MR-GDIP-1")
@register_mr_rule("MR-GDIP1")
def analyze_mr_gdip_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_gdip1(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-GDIP-1"),
        near_goal_tol_m=float(kwargs.get("near_goal_tol_m", GDIP1_NEAR_GOAL_TOL_M)),
        cube_move_min_m=float(kwargs.get("cube_move_min_m", GDIP1_CUBE_MOVE_MIN_M)),
    )
