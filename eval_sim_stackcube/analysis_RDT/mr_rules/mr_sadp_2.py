from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


SADP2_POSITION_TOL_M = 0.025


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory", "eef_path"),
        getattr(record, "eef_path", None),
    ) or []
    points: List[List[float]] = []
    for point in traj:
        try:
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    try:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return arr[:3]


def _first_valid_xyz(seq: Any) -> Optional[np.ndarray]:
    items = seq or []
    for item in items:
        arr = _xyz(item)
        if arr is not None:
            return arr
    return None


def _grasp_and_final_points(record: Any) -> Dict[str, Any]:
    traj = _trajectory_points(record)
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    grasp_point = _xyz(traj[grasp_idx]) if grasp_idx is not None and 0 <= grasp_idx < len(traj) else None
    final_point = _xyz(traj[-1]) if traj else None
    return {
        "traj_len": len(traj),
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
        "grasp_point": grasp_point,
        "final_point": final_point,
    }


def _red_translation_delta(base: Any, mr: Any) -> Optional[np.ndarray]:
    base_red = _first_valid_xyz(
        _first_not_none(
            _nested_get(base, "trajectory", "src_cube_pos"),
            getattr(base, "src_cube_pos", None),
            _nested_get(base, "trajectory", "cube_pos"),
            getattr(base, "cube_pos", None),
        )
    )
    mr_red = _first_valid_xyz(
        _first_not_none(
            _nested_get(mr, "trajectory", "src_cube_pos"),
            getattr(mr, "src_cube_pos", None),
            _nested_get(mr, "trajectory", "cube_pos"),
            getattr(mr, "cube_pos", None),
        )
    )
    if base_red is None or mr_red is None:
        return None
    return mr_red - base_red


def _green_translation_delta(base: Any, mr: Any) -> Optional[np.ndarray]:
    base_green = _first_valid_xyz(
        _first_not_none(
            _nested_get(base, "trajectory", "dst_cube_pos"),
            getattr(base, "dst_cube_pos", None),
        )
    )
    mr_green = _first_valid_xyz(
        _first_not_none(
            _nested_get(mr, "trajectory", "dst_cube_pos"),
            getattr(mr, "dst_cube_pos", None),
        )
    )
    if base_green is None or mr_green is None:
        return None
    return mr_green - base_green


def _anchor_drift(
    base_point: Optional[np.ndarray],
    mr_point: Optional[np.ndarray],
    physical_delta: Optional[np.ndarray],
) -> Optional[float]:
    if base_point is None or mr_point is None or physical_delta is None:
        return None
    actual_delta = np.asarray(mr_point, dtype=np.float64)[:3] - np.asarray(base_point, dtype=np.float64)[:3]
    expected_delta = np.asarray(physical_delta, dtype=np.float64)[:3]
    return float(np.linalg.norm(actual_delta - expected_delta))


def _analyze_sadp2_visual_redundancy_immunity(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    position_tol_m: float,
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

        binfo = _grasp_and_final_points(base)
        minfo = _grasp_and_final_points(mr)
        delta_red = _red_translation_delta(base, mr)
        delta_green = _green_translation_delta(base, mr)

        analyzable = True
        violated = False
        reasons = []

        if binfo["grasp_point"] is None or minfo["grasp_point"] is None:
            analyzable = False
            if binfo["grasp_point"] is None:
                reasons.append("missing_base_grasp_point")
            if minfo["grasp_point"] is None:
                reasons.append("missing_mr_grasp_point")

        if binfo["final_point"] is None or minfo["final_point"] is None:
            analyzable = False
            if binfo["final_point"] is None:
                reasons.append("missing_base_final_point")
            if minfo["final_point"] is None:
                reasons.append("missing_mr_final_point")

        if delta_red is None:
            analyzable = False
            reasons.append("missing_red_translation_delta")
        if delta_green is None:
            analyzable = False
            reasons.append("missing_green_translation_delta")

        grasp_drift = None
        end_drift = None
        if analyzable:
            grasp_drift = _anchor_drift(
                binfo["grasp_point"],
                minfo["grasp_point"],
                delta_red,
            )
            end_drift = _anchor_drift(
                binfo["final_point"],
                minfo["final_point"],
                delta_green,
            )

            if grasp_drift is None or end_drift is None:
                analyzable = False
                reasons.append("cannot_compute_anchor_spatial_drift")
            else:
                if grasp_drift > position_tol_m:
                    violated = True
                    reasons.append(f"VIOLATION: Anchor Spatial Drift (Grasp point slipped by {grasp_drift}m)")
                if end_drift > position_tol_m:
                    violated = True
                    reasons.append(f"VIOLATION: Anchor Spatial Drift (End point slipped by {end_drift}m)")
                if not violated:
                    reasons.append("visual_redundancy_immunity_pass")

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
                "base_grasp_frame_index": binfo["grasp_frame_index"],
                "mr_grasp_frame_index": minfo["grasp_frame_index"],
                "base_grasp_frame_index_source": binfo["grasp_frame_index_source"],
                "mr_grasp_frame_index_source": minfo["grasp_frame_index_source"],
                "base_grasp_point": None if binfo["grasp_point"] is None else binfo["grasp_point"].tolist(),
                "mr_grasp_point": None if minfo["grasp_point"] is None else minfo["grasp_point"].tolist(),
                "base_final_point": None if binfo["final_point"] is None else binfo["final_point"].tolist(),
                "mr_final_point": None if minfo["final_point"] is None else minfo["final_point"].tolist(),
                "delta_d_red": None if delta_red is None else delta_red.tolist(),
                "delta_d_green": None if delta_green is None else delta_green.tolist(),
                "grasp_anchor_drift_m": grasp_drift,
                "end_anchor_drift_m": end_drift,
                "position_tol_m": position_tol_m,
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
            "position_tol_m": position_tol_m,
            "metric": "red_and_green_anchor_spatial_drift",
            "violation_rule": "grasp_anchor_drift > tol OR end_anchor_drift > tol",
            "invariance_expectation": "core_red_green_anchors_should_dominate_over_visual_redundancy",
        },
        "details": details,
    }


@register_mr_rule("MR-SADP2")
@register_mr_rule("MR-SADP-2")
@register_mr_rule("mr_sadp_2")
def analyze_mr_sadp2_visual_redundancy_immunity(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_sadp2_visual_redundancy_immunity(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-SADP2-VISUAL-REDUNDANCY-IMMUNITY"),
        position_tol_m=float(kwargs.get("position_tol_m", SADP2_POSITION_TOL_M)),
    )
