from typing import Any, Dict, List, Optional

import numpy as np

from .mr_sadp_1 import _eef_final_point, _eef_point_at, _first_not_none, _grasp_index_with_source
from .registry import register_mr_rule


CPTMP2_MAX_PATH_LEN_RATIO = 1.3
CPTMP2_MAX_GRASP_POINT_DELTA_M = 0.04
CPTMP2_MAX_FINAL_POINT_DELTA_M = 0.04


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _path_len(record: Any) -> Optional[float]:
    value = getattr(record, "path_len", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _point_l2(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        pa = np.asarray(a, dtype=np.float64).reshape(-1)[:3]
        pb = np.asarray(b, dtype=np.float64).reshape(-1)[:3]
    except Exception:
        return None
    if pa.size < 3 or pb.size < 3:
        return None
    return float(np.linalg.norm(pa - pb))


def _analyze_cptmp2(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    max_path_len_ratio: float,
    max_grasp_point_delta_m: float,
    max_final_point_delta_m: float,
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

        src_grasp_idx, src_grasp_idx_source = _grasp_index_with_source(base)
        dst_grasp_idx, dst_grasp_idx_source = _grasp_index_with_source(mr)

        src_grasp_point = _eef_point_at(base, src_grasp_idx)
        dst_grasp_point = _eef_point_at(mr, dst_grasp_idx)
        src_end_point = _eef_final_point(base)
        dst_end_point = _eef_final_point(mr)

        base_path_len = _path_len(base)
        mr_path_len = _path_len(mr)

        grasp_point_delta_m = _point_l2(src_grasp_point, dst_grasp_point)
        final_point_delta_m = _point_l2(src_end_point, dst_end_point)

        analyzable = True
        violated = False
        reasons = []

       
        if bool(src_success) and not bool(dst_success):
            violated = True
            reasons.append("base_success_followup_fail")

        if src_grasp_point is None or dst_grasp_point is None:
            analyzable = False
            reasons.append("missing_grasp_point")
        if src_end_point is None or dst_end_point is None:
            analyzable = False
            reasons.append("missing_end_point")
        if base_path_len is None or mr_path_len is None:
            analyzable = False
            reasons.append("missing_path_length")

        path_len_ratio = None
        if analyzable:
            if base_path_len <= 1e-8:
                analyzable = False
                reasons.append("base_path_too_short")
            else:
                path_len_ratio = float(mr_path_len / base_path_len)
                if grasp_point_delta_m is not None and grasp_point_delta_m > max_grasp_point_delta_m:
                    violated = True
                    reasons.append(
                        f"grasp_point_drift(grasp_point_delta={grasp_point_delta_m:.4f}m > {max_grasp_point_delta_m:.4f}m)"
                    )
                if final_point_delta_m is not None and final_point_delta_m > max_final_point_delta_m:
                    violated = True
                    reasons.append(
                        f"final_point_drift(final_point_delta={final_point_delta_m:.4f}m > {max_final_point_delta_m:.4f}m)"
                    )
                if path_len_ratio > max_path_len_ratio:
                    violated = True
                    reasons.append(
                        f"excessive_path_oscillation(path_len_ratio={path_len_ratio:.4f} > {max_path_len_ratio:.4f})"
                    )
                if not violated:
                    reasons.append("only_minor_or_no_effect_before_grasp")

        if not analyzable:
            unavailable_count += 1
            violated = False if not bool(src_success) else violated

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
                "grasp_point_delta_m": grasp_point_delta_m,
                "src_end_point": src_end_point,
                "dst_end_point": dst_end_point,
                "final_point_delta_m": final_point_delta_m,
                "base_path_len_m": base_path_len,
                "mr_path_len_m": mr_path_len,
                "path_len_ratio": path_len_ratio,
                "max_path_len_ratio": max_path_len_ratio,
                "max_grasp_point_delta_m": max_grasp_point_delta_m,
                "max_final_point_delta_m": max_final_point_delta_m,
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
            "max_path_len_ratio": max_path_len_ratio,
            "max_grasp_point_delta_m": max_grasp_point_delta_m,
            "max_final_point_delta_m": max_final_point_delta_m,
            "invariance_proxy": "pre_grasp_proprio_noise_should_only_cause_minor_degradation",
        },
        "details": details,
    }


@register_mr_rule("MR-CPTMP-2")
@register_mr_rule("MR-CPTMP2")
def analyze_mr_cptmp_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_cptmp2(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-CPTMP-2"),
        max_path_len_ratio=float(kwargs.get("max_path_len_ratio", CPTMP2_MAX_PATH_LEN_RATIO)),
        max_grasp_point_delta_m=float(
            kwargs.get("max_grasp_point_delta_m", CPTMP2_MAX_GRASP_POINT_DELTA_M)
        ),
        max_final_point_delta_m=float(
            kwargs.get("max_final_point_delta_m", CPTMP2_MAX_FINAL_POINT_DELTA_M)
        ),
    )
