from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _to_xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64)
    if arr.ndim == 0:
        return None
    arr = arr.reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _eef_point_at(record: Any, idx: Optional[int]) -> Optional[List[float]]:
    if idx is None:
        return None
    eef_path = getattr(record, "eef_path", None) or []
    if not (0 <= int(idx) < len(eef_path)):
        return None
    point = _to_xyz(eef_path[int(idx)])
    if point is None:
        return None
    return [float(point[0]), float(point[1]), float(point[2])]


def _eef_final_point(record: Any) -> Optional[List[float]]:
    eef_path = getattr(record, "eef_path", None) or []
    if not eef_path:
        return None
    point = _to_xyz(eef_path[-1])
    if point is None:
        return None
    return [float(point[0]), float(point[1]), float(point[2])]


def _grasp_index(record: Any) -> Optional[int]:
    return _first_not_none(
        getattr(record, "derived_grasp_frame_index", None),
        getattr(record, "mr_eval_grasp_frame_index", None),
    )


@register_mr_rule("MR4")
@register_mr_rule("MR-4")
@register_mr_rule("Target-Object-Relocation")
@register_mr_rule("mr4")
def analyze_mr4_target_object_relocation(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    delta_d = np.asarray(
        kwargs.get("translation_delta", [0.05, 0.05, 0.0]),
        dtype=np.float64,
    ).reshape(-1)
    if delta_d.size < 3:
        raise ValueError("translation_delta must contain at least 3 elements")
    delta_d = delta_d[:3]
    pos_tol = float(kwargs.get("position_tol", 0.025))

    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in (base_records or [])}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        b = bmap[k]
        m = mmap[k]

        src_success = getattr(b, "success", None)
        dst_success = getattr(m, "success", None)

        src_grasp_idx = _grasp_index(b)
        dst_grasp_idx = _grasp_index(m)
        src_grasp_point = _eef_point_at(b, src_grasp_idx)
        dst_grasp_point = _eef_point_at(m, dst_grasp_idx)
        src_end_point = _eef_final_point(b)
        dst_end_point = _eef_final_point(m)

        analyzable = True
        violated = False
        reasons = []
        unavailable_reasons = []

        grasp_shift = None
        end_shift = None
        grasp_error = None
        end_error = None
        spatial_error = None

        if src_grasp_point is not None and dst_grasp_point is None:
            analyzable = True
            violated = True
            reasons = ["Catastrophic_Failure: Lost_Grasp_Intent (Grasp action missing)"]
        elif src_grasp_point is None and dst_grasp_point is None:
            analyzable = False
            unavailable_reasons.append("Invalid_Pair: Both missing grasp")
        elif src_grasp_point is None:
            unavailable_reasons.append("missing_grasp_point")
        else:
            grasp_shift_arr = np.asarray(dst_grasp_point, dtype=np.float64) - np.asarray(src_grasp_point, dtype=np.float64)
            grasp_shift = [float(x) for x in grasp_shift_arr]
            grasp_error = float(np.linalg.norm(grasp_shift_arr - delta_d))

        if reasons:
            pass
        elif src_end_point is None or dst_end_point is None:
            unavailable_reasons.append("missing_end_point")
        else:
            end_shift_arr = np.asarray(dst_end_point, dtype=np.float64) - np.asarray(src_end_point, dtype=np.float64)
            end_shift = [float(x) for x in end_shift_arr]
            end_error = float(np.linalg.norm(end_shift_arr - delta_d))

        if reasons:
            analyzable = True
        elif unavailable_reasons:
            analyzable = False
        else:
            spatial_error = max(grasp_error, end_error)
            if (not bool(dst_success)) and grasp_error > pos_tol:
                violated = True
                reasons.append(
                    f"VIOLATION: Followup failed with grasp-anchor spatial drift (Error: {grasp_error:.3f}m > {pos_tol:.3f}m)"
                )
            if (not bool(dst_success)) and end_error > pos_tol:
                violated = True
                reasons.append(
                    f"VIOLATION: Followup failed with end-anchor spatial drift (Error: {end_error:.3f}m > {pos_tol:.3f}m)"
                )

        if not analyzable:
            unavailable_count += 1
            violated = False
            reasons = unavailable_reasons + reasons
        elif not reasons:
            reasons.append("Rigid_Translation_Equivariant_Pass")

        if violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": k,
            "src_success": src_success,
            "dst_success": dst_success,
            "src_grasp_frame_index": src_grasp_idx,
            "dst_grasp_frame_index": dst_grasp_idx,
            "src_grasp_point": src_grasp_point,
            "dst_grasp_point": dst_grasp_point,
            "expected_translation_delta": [float(x) for x in delta_d],
            "actual_grasp_translation_delta": grasp_shift,
            "grasp_translation_error_m": grasp_error,
            "src_end_point": src_end_point,
            "dst_end_point": dst_end_point,
            "actual_end_translation_delta": end_shift,
            "end_translation_error_m": end_error,
            "position_tol_m": pos_tol,
            "spatial_error_m": spatial_error,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR4",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "translation_delta": [float(x) for x in delta_d],
            "position_tol_m": pos_tol,
            "metric": "grasp_and_end_translation_error",
            "violation_rule": "(followup_success == False) AND (||(P_f - P_s) - Δd|| > position_tol_m)",
        },
        "details": details,
    }
