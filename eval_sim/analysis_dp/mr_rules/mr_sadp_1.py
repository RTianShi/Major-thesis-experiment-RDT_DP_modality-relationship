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
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _point_l2(p1: Any, p2: Any) -> Optional[float]:
    a = _to_xyz(p1)
    b = _to_xyz(p2)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _eef_point_at(record: Any, idx: Optional[int]) -> Optional[List[float]]:
    eef_path = getattr(record, "eef_path", None) or []
    if idx is None or not (0 <= int(idx) < len(eef_path)):
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


def _analyze_sadp_invariance(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    position_tol_m: float,
    final_tol_m: float,
) -> Dict[str, Any]:
    """
    分析SADP不变性违规与否。
    
    【违规逻辑 (violated = True)】：
    1. base_success_followup_fail: 原轨迹成功，MR轨迹失败。
    2. grasp_invariance_broken: 抓取点空间偏移距离 > position_tol_m。
    3. final_invariance_broken: 轨迹最终末端坐标偏移距离 > final_tol_m。
    
    【不可判定逻辑 (analyzable = False)】：
    若 base 或 mr 缺失抓取点(由于抓取索引缺失) 或 缺失终止点，则因距离无法计算标记为 False。并排除在统计之外。
    """
    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0
    success_drop_count = 0

    for k in keys:
        base = bmap[k]
        mr = mmap[k]

        src_success = getattr(base, "success", None)
        dst_success = getattr(mr, "success", None)

        src_grasp_idx = _grasp_index(base)
        dst_grasp_idx = _grasp_index(mr)
        src_grasp_point = _eef_point_at(base, src_grasp_idx)
        dst_grasp_point = _eef_point_at(mr, dst_grasp_idx)
        src_end_point = _eef_final_point(base)
        dst_end_point = _eef_final_point(mr)

        grasp_error = _point_l2(src_grasp_point, dst_grasp_point)
        end_error = _point_l2(src_end_point, dst_end_point)

        analyzable = True
        violated = False
        reasons = []

        if bool(src_success) and not bool(dst_success):
            violated = True
            success_drop_count += 1
            reasons.append("base_success_followup_fail")

        if src_grasp_point is None or dst_grasp_point is None:
            analyzable = False
            reasons.append("missing_grasp_point")
        if src_end_point is None or dst_end_point is None:
            analyzable = False
            reasons.append("missing_end_point")

        if analyzable:
            if grasp_error is not None and grasp_error > position_tol_m:
                violated = True
                reasons.append(f"grasp_invariance_broken({grasp_error:.4f}m)")
            if end_error is not None and end_error > final_tol_m:
                violated = True
                reasons.append(f"final_invariance_broken({end_error:.4f}m)")

        if not analyzable:
            unavailable_count += 1
            violated = False if not bool(src_success) else violated
        elif not reasons:
            reasons.append("invariance_pass")

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "src_success": src_success,
                "dst_success": dst_success,
                "src_grasp_frame_index": src_grasp_idx,
                "dst_grasp_frame_index": dst_grasp_idx,
                "src_grasp_point": src_grasp_point,
                "dst_grasp_point": dst_grasp_point,
                "grasp_position_error_m": grasp_error,
                "src_end_point": src_end_point,
                "dst_end_point": dst_end_point,
                "final_position_error_m": end_error,
                "grasp_position_tol_m": position_tol_m,
                "final_position_tol_m": final_tol_m,
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
        "success_drop_count": success_drop_count,
        "config": {
            "grasp_position_tol_m": position_tol_m,
            "final_position_tol_m": final_tol_m,
            "invariance_proxy": "eef_grasp_point_and_final_point_l2",
        },
        "details": details,
    }


@register_mr_rule("MR-SADP-1")
def analyze_mr_sadp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_sadp_invariance(
        base_records,
        mr_records,
        mr_id="MR-SADP-1",
        position_tol_m=float(kwargs.get("sadp_grasp_tol", kwargs.get("position_tol", 0.02))),
        final_tol_m=float(kwargs.get("sadp_final_tol", kwargs.get("position_tol", 0.02))),
    )
