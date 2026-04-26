from typing import Any, Dict, List
from .registry import register_mr_rule

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _calculate_yaw_error(src_yaw: float, dst_yaw: float, expected_delta: float = 45.0) -> float:
    actual_delta = float(dst_yaw) - float(src_yaw)
    relative_diff = actual_delta - float(expected_delta)
    # 对正方形方块的 90 度对称性做折叠，将误差映射到 [-45, 45)。
    error = ((relative_diff + 45.0) % 90.0) - 45.0
    return abs(error)

@register_mr_rule("MR-SEMP-1")
def analyze_mr_semp_1(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    """
    MR-SEMP-1: 末端执行器抓取瞬间 yaw 角等变测试（考虑方块 90 度对称性）
    - 源用例抓取瞬间yaw为 Yaws
    - 衍生用例抓取瞬间yaw为 Yawf
    - 要求 (Yawf - Yaws) ≈ Δθ (Δθ=45°)
    - 对正方形方块的 90 度旋转对称性进行折叠
    - 若误差在容差内，判定为 Equivariant_Pass
    - 若误差超出容差但源/衍生用例都成功，判定为 Shortcut_Sloppy_Grasp
    - 否则判定为 Equivariance_Broken
    """
    angle_delta = float(kwargs.get("angle_delta", 45.0))
    angle_tol = float(kwargs.get("angle_tol", 15.0))

    # 按 seed 或 episode_id 配对
    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    passed = 0
    drp_only = 0
    unavailable = 0

    for k in keys:
        b = bmap[k]
        m = mmap[k]

        yaws = _first_not_none(
            getattr(b, "derived_eef_yaw_at_grasp", None),
            getattr(b, "mr_eval_eef_yaw_at_grasp", None),
        )
        yawf = _first_not_none(
            getattr(m, "derived_eef_yaw_at_grasp", None),
            getattr(m, "mr_eval_eef_yaw_at_grasp", None),
        )

        s_success = getattr(b, "success", None)
        f_success = getattr(m, "success", None)
        analyzable = (yaws is not None and yawf is not None)
        reasons = []
        error = None
        actual_delta = None
        relative_diff = None
        pass_flag = False
        drp_flag = False

        if analyzable:
            expected = yaws + angle_delta
            actual_delta = yawf - yaws
            relative_diff = actual_delta - angle_delta
            error = _calculate_yaw_error(yaws, yawf, angle_delta)

            if error <= angle_tol:
                pass_flag = True
                passed += 1
                reasons.append("Equivariant_Pass")
            elif s_success and f_success:
                drp_flag = True
                drp_only += 1
                reasons.append(f"Shortcut_Sloppy_Grasp (Error: {error:.1f})")
            else:
                reasons.append(f"Equivariance_Broken (Error: {error:.1f})")
        else:
            unavailable += 1
            reasons.append("missing_yaw_at_grasp_from_raw_and_mr_eval")

        details.append({
            "key(seed_or_episode)": k,
            "src_yaw_at_grasp": yaws,
            "dst_yaw_at_grasp": yawf,
            "expected_dst_yaw": None if yaws is None else yaws + angle_delta,
            "actual_delta_yaw": actual_delta,
            "relative_diff_before_symmetry": relative_diff,
            "yaw_error": error,
            "yaw_error_corrected": error,
            "src_success": s_success,
            "dst_success": f_success,
            "pass": pass_flag,
            "drp_only": drp_flag,
            "analyzable": analyzable,
            "reasons": reasons,
        })

    total = len(keys)
    analyzable_total = total - unavailable
    violations = analyzable_total - passed
    return {
        "mr_id": "MR-SEMP-1",
        "paired_episodes": total,
        "analyzable_episodes": analyzable_total,
        "unavailable_count": unavailable,
        "passed_count": passed,
        "pass_count": passed,
        "drp_only_count": drp_only,
        "violations": violations,
        "pass_rate_percent": (passed / analyzable_total * 100.0) if analyzable_total > 0 else None,
        "violation_rate_percent": (violations / analyzable_total * 100.0) if analyzable_total > 0 else None,
        "drp_only_rate_percent": (drp_only / analyzable_total * 100.0) if analyzable_total > 0 else None,
        "config": {
            "angle_delta": angle_delta,
            "angle_tol": angle_tol,
            "symmetry_period": 90.0,
        },
        "details": details
    }
