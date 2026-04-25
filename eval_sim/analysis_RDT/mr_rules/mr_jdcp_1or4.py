try:
    from eval_sim.analysis_RDT.mr_rules import register_mr_rule
except ImportError:
    from . import register_mr_rule

import numpy as np
from typing import List, Dict, Any, Optional


def _flatten_eef_path(eef_path):
    if not eef_path or not isinstance(eef_path[0], list):
        return []
    if isinstance(eef_path[0][0], list):
        return [point[0] for point in eef_path if point and isinstance(point[0], list)]
    return eef_path


def _point_at(eef_path, idx: Optional[int]):
    flat = _flatten_eef_path(eef_path)
    if idx is None or idx < 0 or idx >= len(flat):
        return None
    p = flat[idx]
    if p is None or len(p) < 3:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _end_point(eef_path):
    flat = _flatten_eef_path(eef_path)
    if not flat:
        return None
    p = flat[-1]
    if p is None or len(p) < 3:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _grasp_index(record: Any):
    derived = getattr(record, "derived_grasp_frame_index", None)
    if derived is not None:
        return int(derived)
    raw = getattr(record, "mr_eval_grasp_frame_index", None)
    if raw is not None:
        return int(raw)
    return None


def _l2_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return None
    return float(np.linalg.norm(a[:3] - b[:3]))


def _analyze_jdcp(base_records: List[Any], mr_records: List[Any], mr_id: str, **kwargs) -> Dict[str, Any]:
    pos_tol = float(kwargs.get("position_tol", 0.05))

    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        base = bmap[k]
        mr = mmap[k]

        base_success = getattr(base, "success", None)
        mr_success = getattr(mr, "success", None)

        base_grasp_idx = _grasp_index(base)
        mr_grasp_idx = _grasp_index(mr)
        base_grasp_point = _point_at(base.eef_path, base_grasp_idx)
        mr_grasp_point = _point_at(mr.eef_path, mr_grasp_idx)
        base_end_point = _end_point(base.eef_path)
        mr_end_point = _end_point(mr.eef_path)

        grasp_error = _l2_distance(base_grasp_point, mr_grasp_point)
        end_error = _l2_distance(base_end_point, mr_end_point)

        analyzable = True
        violated = False
        reasons = []

        if bool(base_success) and not bool(mr_success):
            violated = True
            reasons.append("Logic_Brittleness")

        if grasp_error is None and end_error is None:
            analyzable = False
            unavailable_count += 1
            reasons.append("missing_grasp_and_end_points")
        else:
            if grasp_error is not None and grasp_error > pos_tol:
                violated = True
                reasons.append(f"Grasp_Point_Drift (Error: {grasp_error:.3f}m)")
            if end_error is not None and end_error > pos_tol:
                violated = True
                reasons.append(f"End_Point_Drift (Error: {end_error:.3f}m)")

        if analyzable and not reasons:
            reasons.append("Equivalent_Behavior_Preserved")

        if violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": k,
            "source_success": base_success,
            "followup_success": mr_success,
            "src_grasp_frame_index": base_grasp_idx,
            "dst_grasp_frame_index": mr_grasp_idx,
            "src_grasp_point": base_grasp_point,
            "dst_grasp_point": mr_grasp_point,
            "src_end_point": base_end_point,
            "dst_end_point": mr_end_point,
            "grasp_error_m": grasp_error,
            "end_error_m": end_error,
            "position_tol_m": pos_tol,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

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
            "position_tol_m": pos_tol,
            "trajectory_similarity_proxy": "grasp_and_end_point_l2",
        },
        "details": details,
    }


@register_mr_rule("MR-JDCP-1")
def analyze_mr_jdcp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_jdcp(base_records, mr_records, mr_id="MR-JDCP-1", **kwargs)


@register_mr_rule("MR-JDCP-4")
def analyze_mr_jdcp_4(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_jdcp(base_records, mr_records, mr_id="MR-JDCP-4", **kwargs)
