try:
    from eval_sim.analysis.mr_rules import register_mr_rule
except ImportError:
    from . import register_mr_rule

import numpy as np
from typing import List, Dict, Any

def _flatten_eef_path(eef_path):
    if not eef_path or not isinstance(eef_path[0], list):
        return []
    if isinstance(eef_path[0][0], list):
        return [point[0] for point in eef_path if point and isinstance(point[0], list)]
    return eef_path

def _approach_vector(eef_path, n=10):
    eef_path = _flatten_eef_path(eef_path)
    if not eef_path or len(eef_path) < 2:
        return None
    n = min(n, len(eef_path) - 1)
    v = np.array(eef_path[n][:2]) - np.array(eef_path[0][:2])
    if np.linalg.norm(v) < 1e-6:
        return None
    return v / np.linalg.norm(v)

def _direction_vec(from_xy, to_xy):
    v = np.array(to_xy) - np.array(from_xy)
    if np.linalg.norm(v) < 1e-6:
        return None
    return v / np.linalg.norm(v)

def _angle_deg(v1, v2):
    if v1 is None or v2 is None:
        return None
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(dot) * 180.0 / np.pi)

@register_mr_rule("MR-DRP-2")
def analyze_mr_drp_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    # 阈值可调
    approach_n = kwargs.get("approach_n", 10)
    center_xy = np.array([0.0, 0.0])
    extreme_eps = 0.09  # 极点判定阈值
    center_bias_angle = 30.0  # 与中心方向夹角小于此角度判为中心偏置
    details = []
    violations = 0
    keys = set(r.seed or r.episode_id for r in base_records) & set(r.seed or r.episode_id for r in mr_records)
    base_dict = {r.seed or r.episode_id: r for r in base_records}
    mr_dict = {r.seed or r.episode_id: r for r in mr_records}

    for k in sorted(keys):
        base = base_dict[k]
        mr = mr_dict[k]
        # cube 初始位置
        base_cube = base.cube_pos[0][:2] if base.cube_pos and len(base.cube_pos[0]) >= 2 else None
        mr_cube = mr.cube_pos[0][:2] if mr.cube_pos and len(mr.cube_pos[0]) >= 2 else None
        # 判断衍生用例是否极点
        mr_is_extreme = mr_cube is not None and (abs(mr_cube[0]) > extreme_eps or abs(mr_cube[1]) > extreme_eps)
        # 轨迹接近向量
        base_vec = _approach_vector(base.eef_path, n=approach_n)
        mr_vec = _approach_vector(mr.eef_path, n=approach_n)
        # 展平 eef_path，确保拿到 [x, y]
        base_eef_flat = _flatten_eef_path(base.eef_path)
        if base_eef_flat and len(base_eef_flat[0]) >= 2:
            base_xy = base_eef_flat[0][:2]
        else:
            base_xy = None
        # 目标方向
        mr_target_vec = _direction_vec(base_xy, mr_cube) if (base_xy is not None and mr_cube is not None) else None
        center_vec = _direction_vec(base_xy, center_xy) if base_xy is not None else None
        # 夹角
        angle_to_target = _angle_deg(mr_vec, mr_target_vec)
        angle_to_center = _angle_deg(mr_vec, center_vec)
        violated = False
        reasons = []
        if mr_is_extreme:
            # 违例判据：接近向量与极点方向夹角过大，或与中心夹角过小
            if angle_to_target is not None and angle_to_target > 45.0:
                violated = True
                reasons.append("approach_not_point_to_extreme")
            if angle_to_center is not None and angle_to_center < center_bias_angle:
                violated = True
                reasons.append("center_bias")
        details.append({
            "key(seed_or_episode)": k,
            "base_cube_xy": base_cube,
            "mr_cube_xy": mr_cube,
            "mr_is_extreme": mr_is_extreme,
            "base_approach_vec": base_vec.tolist() if base_vec is not None else None,
            "mr_approach_vec": mr_vec.tolist() if mr_vec is not None else None,
            "mr_target_vec": mr_target_vec.tolist() if mr_target_vec is not None else None,
            "angle_to_target": angle_to_target,
            "angle_to_center": angle_to_center,
            "violated": violated,
            "reasons": reasons,
            "source_success": getattr(base, "success", None),
            "followup_success": getattr(mr, "success", None),
        })
        if violated:
            violations += 1

    return {
        "mr_id": "MR-DRP-2",
        "paired_episodes": len(keys),
        "violations": violations,
        "violation_rate_percent": (violations / len(keys) * 100.0) if keys else None,
        "details": details,
        "config": {
            "approach_n": approach_n,
            "extreme_eps": extreme_eps,
            "center_bias_angle": center_bias_angle,
        }
    }
