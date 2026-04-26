from typing import Any, Dict, List

import numpy as np

from .registry import register_mr_rule


def _flatten_eef_path(eef_path):
    """
    展平 eef_path 数据，支持三层嵌套结构 [[[x, y, z]]] -> [[x, y, z]]。
    如果已经是二维结构，则直接返回。
    """
    if not eef_path or not isinstance(eef_path[0], list):
        return []
    if isinstance(eef_path[0][0], list):
        return [point[0] for point in eef_path if point and isinstance(point[0], list)]
    return eef_path


def _estimate_stop_index(eef_path, tail_window: int = 5, step_eps: float = 5e-4):
    """
    更鲁棒的停止点检测：
    - 如果轨迹末尾窗口确实静止，则返回静止窗口起点
    - 否则直接返回最后一个非空点
    """
    eef_path = _flatten_eef_path(eef_path)  # 展平 eef_path
    if not eef_path:
        return None

    arr = np.asarray(eef_path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
        return None
    if len(arr) == 1:
        return 0

    w = max(2, min(tail_window, len(arr)))
    # 检查轨迹末尾窗口是否静止
    seg = arr[-w:, :3]
    step = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    if np.all(step <= step_eps):
        return len(arr) - w  # 静止窗口起点
    # 否则直接取最后一个点
    return len(arr) - 1


def _zend_stop(record):
    """
    获取停止点的 Z 值。
    """
    eef_path = _flatten_eef_path(record.eef_path)  # 展平 eef_path
    idx = _estimate_stop_index(eef_path)
    if idx is None or not eef_path:
        return None
    p = eef_path[idx]
    return float(p[2]) if len(p) >= 3 else None


@register_mr_rule("MR-DRP-1")
def analyze_mr_drp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    expected_delta = float(kwargs.get("expected_delta_z", 0.23))
    tol = float(kwargs.get("delta_z_tol", 0.05))

    # 新参数名：low_stop_min/max；兼容旧参数名：low_release_min/max
    low_stop_min = float(kwargs.get("low_stop_min", kwargs.get("low_release_min", 0.05)))
    low_stop_max = float(kwargs.get("low_stop_max", kwargs.get("low_release_max", 0.15)))
    low_stop_band = (low_stop_min, low_stop_max)

    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        b = bmap[k]
        m = mmap[k]
        z_s = _zend_stop(b)
        z_f = _zend_stop(m)

        violated = False
        analyzable = True
        reasons = []

        if z_s is None or z_f is None:
            analyzable = False
            unavailable_count += 1
            reasons.append("cannot_estimate_end_z")
        else:
            dz = abs(z_f - z_s)
            if not (expected_delta - tol <= dz <= expected_delta + tol):
                violated = True
                reasons.append("delta_z_not_match_expected")

            # 衍生用例仍在低高度停止，可能是“Z轴盲目执行”
            if low_stop_band[0] <= z_f <= low_stop_band[1]:
                violated = True
                reasons.append("premature_low_stop_possible_z_blind")

        if violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": k,
            "zend_source": z_s,
            "zend_followup": z_f,
            "abs_delta_z": None if (z_s is None or z_f is None) else abs(z_f - z_s),
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
            "source_success": getattr(b, "success", None),
            "followup_success": getattr(m, "success", None),
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-DRP-1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "expected_delta_z": expected_delta,
            "delta_z_tol": tol,
            "low_stop_band": [low_stop_min, low_stop_max],
        },
        "details": details
    }
