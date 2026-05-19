from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_sadp_1 import _eef_point_at, _first_not_none, _grasp_index_with_source, _nested_get, _to_1d_float_array
from .registry import register_mr_rule


SESP1_K_FACTOR = 1.5
SESP1_WIDTH_TOL = 0.01


def _gripper_width_sequence(record: Any) -> Optional[np.ndarray]:
    gripper_width = _first_not_none(
        _nested_get(record, "gripper_width"),
        _nested_get(record, "trajectory", "gripper_width"),
    )
    gripper_finger_qpos = _first_not_none(
        _nested_get(record, "gripper_finger_qpos"),
        _nested_get(record, "trajectory", "gripper_finger_qpos"),
    )

    widths = _to_1d_float_array(gripper_width)
    if widths is None and gripper_finger_qpos is not None:
        q = np.asarray(gripper_finger_qpos, dtype=np.float64)
        if q.ndim >= 2 and q.shape[-1] >= 2:
            widths = np.sum(q[..., :2], axis=-1).reshape(-1)

    return widths


def _gripper_width_at(record: Any, idx: Optional[int]) -> Optional[float]:
    widths = _gripper_width_sequence(record)
    if widths is None or idx is None or not (0 <= int(idx) < len(widths)):
        return None
    return float(widths[int(idx)])


def _analyze_sesp_scaling(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    k_factor: float,
    width_tol: float,
) -> Dict[str, Any]:
    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        base = bmap[k]
        mr = mmap[k]

        src_success = getattr(base, "success", None)
        dst_success = getattr(mr, "success", None)

        src_grasp_idx, src_grasp_idx_source = _grasp_index_with_source(base)
        dst_grasp_idx, dst_grasp_idx_source = _grasp_index_with_source(mr)

        src_grasp_width = _gripper_width_at(base, src_grasp_idx)
        dst_grasp_width = _gripper_width_at(mr, dst_grasp_idx)
        src_grasp_point = _eef_point_at(base, src_grasp_idx)
        dst_grasp_point = _eef_point_at(mr, dst_grasp_idx)

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
                "src_success": src_success,
                "dst_success": dst_success,
                "src_grasp_frame_index": src_grasp_idx,
                "dst_grasp_frame_index": dst_grasp_idx,
                "src_grasp_frame_index_source": src_grasp_idx_source,
                "dst_grasp_frame_index_source": dst_grasp_idx_source,
                "src_grasp_point": src_grasp_point,
                "dst_grasp_point": dst_grasp_point,
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
        "mr_id": mr_id,
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


@register_mr_rule("MR-SESP-1")
def analyze_mr_sesp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_sesp_scaling(
        base_records,
        mr_records,
        mr_id="MR-SESP-1",
        k_factor=float(kwargs.get("k_factor", SESP1_K_FACTOR)),
        width_tol=float(kwargs.get("width_tol", SESP1_WIDTH_TOL)),
    )
