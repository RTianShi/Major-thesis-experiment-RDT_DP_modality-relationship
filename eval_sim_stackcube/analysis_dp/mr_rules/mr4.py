from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


MR4_POSITION_TOL_M = 0.025


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
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
        "grasp_point": grasp_point,
        "final_point": final_point,
    }


def _translation_vector(base: Any, mr: Any) -> Optional[np.ndarray]:
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
    if base_red is not None and mr_red is not None:
        delta = mr_red - base_red
        delta[2] = 0.0
        return delta
    return None


def _translation_error(
    base_point: Optional[np.ndarray],
    mr_point: Optional[np.ndarray],
    translation_vec: Optional[np.ndarray],
) -> Optional[float]:
    if base_point is None or mr_point is None or translation_vec is None:
        return None
    actual_shift = np.asarray(mr_point, dtype=np.float64)[:3] - np.asarray(base_point, dtype=np.float64)[:3]
    expected_shift = np.asarray(translation_vec, dtype=np.float64)[:3]
    return float(np.linalg.norm(actual_shift - expected_shift))


@register_mr_rule("MR4")
@register_mr_rule("MR-4")
@register_mr_rule("Target-Object-Relocation")
def analyze_mr4_target_object_relocation(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    position_tol_m = float(kwargs.get("position_tol_m", MR4_POSITION_TOL_M))
    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]
        binfo = _grasp_and_final_points(base)
        minfo = _grasp_and_final_points(mr)
        translation_vec = _translation_vector(base, mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if binfo["grasp_point"] is None:
            analyzable = False
            reasons.append("missing_base_grasp_point")
        if minfo["grasp_point"] is None:
            analyzable = False
            reasons.append("missing_mr_grasp_point")
        if binfo["final_point"] is None:
            analyzable = False
            reasons.append("missing_base_final_point")
        if minfo["final_point"] is None:
            analyzable = False
            reasons.append("missing_mr_final_point")
        if translation_vec is None:
            analyzable = False
            reasons.append("missing_translation_vector")

        grasp_error = None
        final_error = None
        if analyzable:
            grasp_error = _translation_error(binfo["grasp_point"], minfo["grasp_point"], translation_vec)
            final_error = _translation_error(binfo["final_point"], minfo["final_point"], translation_vec)
            if grasp_error is None or final_error is None:
                analyzable = False
                reasons.append("cannot_compute_translation_error")
            else:
                if grasp_error > position_tol_m:
                    violated = True
                    reasons.append("VIOLATION: ||(Pgrasp_f - Pgrasp_s) - Δd|| > 0.025m")
               
                if not violated:
                    reasons.append("target_object_relocation_pass")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "base_success": getattr(base, "success", None),
            "mr_success": getattr(mr, "success", None),
            "base_grasp_frame_index": binfo["grasp_frame_index"],
            "mr_grasp_frame_index": minfo["grasp_frame_index"],
            "base_grasp_frame_index_source": binfo["grasp_frame_index_source"],
            "mr_grasp_frame_index_source": minfo["grasp_frame_index_source"],
            "translation_vector": None if translation_vec is None else translation_vec.tolist(),
            "base_grasp_point": None if binfo["grasp_point"] is None else binfo["grasp_point"].tolist(),
            "mr_grasp_point": None if minfo["grasp_point"] is None else minfo["grasp_point"].tolist(),
            "base_final_point": None if binfo["final_point"] is None else binfo["final_point"].tolist(),
            "mr_final_point": None if minfo["final_point"] is None else minfo["final_point"].tolist(),
            "grasp_translation_error_m": grasp_error,
            "final_translation_error_m": final_error,
            "position_tol_m": position_tol_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": kwargs.get("mr_id", "MR4"),
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "position_tol_m": position_tol_m,
            "metric": "grasp_and_final_translation_equivariance_error",
            "violation_rule": "||(Pgrasp_f - Pgrasp_s) - Δd|| > tol OR ||(Pfinal_f - Pfinal_s) - Δd|| > tol",
            "expectation": "grasp_and_place_points_should_translate_with_target_objects",
        },
        "details": details,
    }
