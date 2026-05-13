from typing import Any, Dict, List, Optional

import numpy as np

from .mr_utils import _first_not_none, _grasp_index_with_source, _nested_get
from .registry import register_mr_rule


SCDP1_DRIFT_TOL_M = 0.025


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory", "eef_path"),
        getattr(record, "eef_path", None),
    ) or []
    points: List[List[float]] = []
    for p in traj:
        try:
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _point_l2(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def _grasp_and_final_points(record: Any) -> Dict[str, Any]:
    traj = _trajectory_points(record)
    grasp_idx, grasp_idx_source = _grasp_index_with_source(record)
    grasp_point = traj[grasp_idx] if grasp_idx is not None and 0 <= grasp_idx < len(traj) else None
    final_point = traj[-1] if traj else None
    return {
        "traj_len": len(traj),
        "grasp_frame_index": grasp_idx,
        "grasp_frame_index_source": grasp_idx_source,
        "grasp_point": grasp_point,
        "final_point": final_point,
    }


def _analyze_scdp1_visual_distractor_debunking(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    drift_tol_m: float,
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

        grasp_drift_m = None
        final_drift_m = None
        if analyzable:
            grasp_drift_m = _point_l2(binfo["grasp_point"], minfo["grasp_point"])
            final_drift_m = _point_l2(binfo["final_point"], minfo["final_point"])

            if grasp_drift_m is None or final_drift_m is None:
                analyzable = False
                reasons.append("cannot_compute_drifts")
            else:
                if grasp_drift_m > drift_tol_m:
                    violated = True
                    reasons.append("VIOLATION: Grasp point spatial drift > 2.5cm")
                if final_drift_m > drift_tol_m:
                    violated = True
                    reasons.append("VIOLATION: Final placement drift > 2.5cm")
                if not violated:
                    reasons.append("absolute_invariance_pass")

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
                "base_traj_len": binfo["traj_len"],
                "mr_traj_len": minfo["traj_len"],
                "base_grasp_point": binfo["grasp_point"],
                "mr_grasp_point": minfo["grasp_point"],
                "base_final_point": binfo["final_point"],
                "mr_final_point": minfo["final_point"],
                "grasp_point_drift_m": grasp_drift_m,
                "final_point_drift_m": final_drift_m,
                "drift_tol_m": drift_tol_m,
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
            "drift_tol_m": drift_tol_m,
            "metric": "grasp_and_final_xyz_drift",
            "violation_rule": "distance(grasp_point_base, grasp_point_mr) > drift_tol_m OR "
            "distance(final_point_base, final_point_mr) > drift_tol_m",
            "invariance_expectation": "visual_only_change_should_not_move_grasp_or_place_points",
        },
        "details": details,
    }


@register_mr_rule("MR-SCDP1")
@register_mr_rule("MR-SCDP-1")
@register_mr_rule("mr_scdp_1")
def analyze_mr_scdp1_visual_distractor_debunking(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_scdp1_visual_distractor_debunking(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-SCDP1-VISUAL-DISTRACTOR-DEBUNKING"),
        drift_tol_m=float(kwargs.get("drift_tol_m", SCDP1_DRIFT_TOL_M)),
    )

