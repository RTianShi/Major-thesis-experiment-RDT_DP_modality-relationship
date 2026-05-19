from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


MR_C1_GRASP_DIST_THRESH = 0.04


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


def _first_valid_point(points: Any) -> Optional[List[float]]:
    if points is None:
        return None
    if isinstance(points, list):
        for item in points:
            xyz = _to_xyz(item)
            if xyz is not None:
                return xyz
    return _to_xyz(points)


def _trajectory_points(record: Any, key: str) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory", key),
        getattr(record, key, None),
    ) or []
    points: List[List[float]] = []
    for p in traj:
        xyz = _to_xyz(p)
        if xyz is not None:
            points.append(xyz)
    return points


def _eef_point_at(record: Any, idx: Optional[int]) -> Optional[List[float]]:
    traj = _trajectory_points(record, "eef_path")
    if idx is None or not (0 <= int(idx) < len(traj)):
        return None
    return traj[int(idx)]


def _blue_cube_at(record: Any, idx: Optional[int]) -> Optional[List[float]]:
    src_traj = _trajectory_points(record, "src_cube_pos")
    if idx is not None and 0 <= int(idx) < len(src_traj):
        return src_traj[int(idx)]
    if src_traj:
        return src_traj[0]

    cube_traj = _trajectory_points(record, "cube_pos")
    if idx is not None and 0 <= int(idx) < len(cube_traj):
        return cube_traj[int(idx)]
    if cube_traj:
        return cube_traj[0]

    return _first_valid_point(_nested_get(record, "positions", "blue_cube_initial"))


def _yellow_cube_at(record: Any, idx: Optional[int]) -> Optional[List[float]]:
    dst_traj = _trajectory_points(record, "dst_cube_pos")
    if idx is not None and 0 <= int(idx) < len(dst_traj):
        return dst_traj[int(idx)]
    if dst_traj:
        return dst_traj[0]

    return _first_valid_point(_nested_get(record, "positions", "yellow_cube_initial"))


def _point_l2(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _analyze_c1_sdpp_conflict(
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

        grasp_idx, grasp_idx_source = _grasp_index_with_source(mr)
        grasp_point = _eef_point_at(mr, grasp_idx)
        blue_cube = _blue_cube_at(mr, grasp_idx)
        yellow_cube = _yellow_cube_at(mr, grasp_idx)

        analyzable = True
        violated = False
        reasons = []

        if grasp_point is None:
            analyzable = False
            reasons.append("missing_grasp_point")
        if blue_cube is None and yellow_cube is None:
            analyzable = False
            reasons.append("missing_color_cube_positions")

        dist_blue = _point_l2(grasp_point, blue_cube)
        dist_yellow = _point_l2(grasp_point, yellow_cube)

        if analyzable:
            if dist_blue is None and dist_yellow is None:
                analyzable = False
                reasons.append("cannot_compute_distances")
            else:
                if dist_blue is not None and dist_blue < dist_thresh:
                    violated = True
                    reasons.append(f"grasp_near_blue_cube({dist_blue:.4f} < {dist_thresh:.4f})")
                if dist_yellow is not None and dist_yellow < dist_thresh:
                    violated = True
                    reasons.append(f"grasp_near_yellow_cube({dist_yellow:.4f} < {dist_thresh:.4f})")
                if not violated:
                    reasons.append("no_conflict_grasp_detected")

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
                "grasp_frame_index": grasp_idx,
                "grasp_frame_index_source": grasp_idx_source,
                "grasp_point": grasp_point,
                "blue_cube_pos": blue_cube,
                "yellow_cube_pos": yellow_cube,
                "dist_to_blue_m": dist_blue,
                "dist_to_yellow_m": dist_yellow,
                "distance_thresh_m": dist_thresh,
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
            "metric": "dist(Pgrasp_f, Pblue_cube) or dist(Pgrasp_f, Pyellow_cube)",
            "violation_rule": "distance < distance_thresh_m for blue or yellow cube",
        },
        "details": details,
    }


@register_mr_rule("MR-C1")
@register_mr_rule("MR-C1-SDPP-CONFLICT")
def analyze_mr_c1_sdpp_conflict(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_c1_sdpp_conflict(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-C1"),
        dist_thresh=float(kwargs.get("dist_thresh_m", MR_C1_GRASP_DIST_THRESH)),
    )
