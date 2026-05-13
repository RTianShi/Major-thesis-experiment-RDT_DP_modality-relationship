from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import register_mr_rule


SESP1_K_FACTOR = 1.5
SESP1_WIDTH_TOL = 0.01
SESP1_CONTACT_PROXIMITY_TOL = 0.08
SESP1_CONTACT_CMD_CLOSE_THRESH = -0.02


def _index_by_seed_or_episode(records: List[Any]) -> Dict[int, Any]:
    out = {}
    for r in records:
        key = r.seed if r.seed is not None else r.episode_id
        if key is not None:
            out[key] = r
    return out


def _xyz(point: Any) -> Optional[np.ndarray]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _contact_grasp_index(
    rec: Any,
    *,
    proximity_thresh: float = SESP1_CONTACT_PROXIMITY_TOL,
    cmd_close_thresh: float = SESP1_CONTACT_CMD_CLOSE_THRESH,
) -> Optional[int]:
    gripper_cmd = getattr(rec, "gripper_action_cmd", None) or []
    eef_path = getattr(rec, "eef_path", None) or []
    cube_pos = getattr(rec, "cube_pos", None) or []
    n = min(len(gripper_cmd), len(eef_path), len(cube_pos))
    for i in range(n):
        cmd = gripper_cmd[i]
        if cmd is None or float(cmd) > cmd_close_thresh:
            continue
        eef_xyz = _xyz(eef_path[i])
        cube_xyz = _xyz(cube_pos[i])
        if eef_xyz is None or cube_xyz is None:
            continue
        if float(np.linalg.norm(eef_xyz - cube_xyz)) <= float(proximity_thresh):
            return int(i)
    return None


def _grasp_index_with_source(
    rec: Any,
    *,
    contact_only: bool = False,
    contact_proximity_thresh: float = SESP1_CONTACT_PROXIMITY_TOL,
) -> Tuple[Optional[int], Optional[str]]:
    if rec is None:
        return None, None
    if contact_only:
        idx = _contact_grasp_index(rec, proximity_thresh=contact_proximity_thresh)
        if idx is not None:
            return int(idx), "contact"
    if getattr(rec, "derived_grasp_frame_index", None) is not None:
        return int(rec.derived_grasp_frame_index), "derived"
    if getattr(rec, "mr_eval_grasp_frame_index", None) is not None:
        return int(rec.mr_eval_grasp_frame_index), "mr_eval"
    return None, None


def _gripper_width_at(rec: Any, idx: Optional[int]) -> Optional[float]:
    widths = getattr(rec, "gripper_width", None) or []
    if idx is None or not (0 <= int(idx) < len(widths)):
        return None
    width = widths[int(idx)]
    if width is None:
        return None
    return float(width)


@register_mr_rule("MR-SESP1")
@register_mr_rule("MR-SESP-1")
@register_mr_rule("mr_sesp_1")
def analyze_mr_sesp_1(base_records, mr_records, **kwargs):
    k_factor = float(kwargs.get("k_factor", SESP1_K_FACTOR))
    width_tol = float(kwargs.get("width_tol", SESP1_WIDTH_TOL))

    bmap = _index_by_seed_or_episode(base_records)
    mmap = _index_by_seed_or_episode(mr_records)
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        b = bmap[k]
        m = mmap[k]

        src_grasp_idx, src_grasp_idx_source = _grasp_index_with_source(b)
        dst_grasp_idx, dst_grasp_idx_source = _grasp_index_with_source(
            m,
            contact_only=True,
            contact_proximity_thresh=float(kwargs.get("contact_proximity_thresh", SESP1_CONTACT_PROXIMITY_TOL)),
        )
        src_grasp_width = _gripper_width_at(b, src_grasp_idx)
        dst_grasp_width = _gripper_width_at(m, dst_grasp_idx)

        analyzable = True
        violated = False
        reasons = []

        if src_grasp_idx is None or dst_grasp_idx is None:
            analyzable = False
            reasons.append("missing_grasp_frame_index")
        if src_grasp_width is None or dst_grasp_width is None:
            analyzable = False
            reasons.append("missing_gripper_width_at_grasp")

        width_expected = None
        width_error = None
        if analyzable:
            if float(src_grasp_width) == 0.0 or float(dst_grasp_width) == 0.0:
                violated = True
                reasons.append("gripper_width_zero_at_grasp")
            width_expected = float(src_grasp_width) * float(k_factor)
            width_error = abs(float(dst_grasp_width) - width_expected)
            if width_error > width_tol:
                violated = True
                reasons.append(f"gripper_width_scaling_broken({width_error:.4f}m)")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif not reasons:
            reasons.append("scaling_pass")

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "base_success": bool(b.success) if b.success is not None else None,
                "mr_success": bool(m.success) if m.success is not None else None,
                "src_grasp_frame_index": src_grasp_idx,
                "dst_grasp_frame_index": dst_grasp_idx,
                "src_grasp_frame_index_source": src_grasp_idx_source,
                "dst_grasp_frame_index_source": dst_grasp_idx_source,
                "src_gripper_width_at_grasp_m": src_grasp_width,
                "dst_gripper_width_at_grasp_m": dst_grasp_width,
                "k_factor": k_factor,
                "expected_dst_gripper_width_m": width_expected,
                "gripper_width_error_m": width_error,
                "gripper_width_tol_m": width_tol,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
               
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None
    return {
        "mr_id": "MR-SESP1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "k_factor": k_factor,
            "width_tol_m": width_tol,
            "invariance_proxy": "gripper_width_at_grasp_scaled_by_k",
        },
        "details": details,
    }
