from typing import Any, Dict, List, Optional

from .mr_utils import _first_not_none
from .registry import register_mr_rule


SEMP2_ANGLE_TOL_DEG = 15.0
SEMP2_DEFAULT_DELTA_THETA_DEG = 45.0
SEMP2_POSE_BLINDNESS_YAW_EPS_DEG = 5.0
SEMP2_POSE_BLINDNESS_DELTA_MIN_DEG = 20.0


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _folded_yaw_error(actual_delta_deg: float, expected_delta_deg: float) -> float:
    relative_diff = float(actual_delta_deg) - float(expected_delta_deg)
    folded = ((relative_diff + 45.0) % 90.0) - 45.0
    return abs(float(folded))


def _yaw_at_grasp(record: Any) -> Optional[float]:
    value = _first_not_none(
        getattr(record, "derived_eef_yaw_at_grasp", None),
        getattr(record, "mr_eval_eef_yaw_at_grasp", None),
    )
    return None if value is None else float(value)


def _expected_delta_theta(base: Any, mr: Any, fallback_delta_theta: float) -> float:
    value = _first_not_none(
        getattr(mr, "mr_eval_expected_delta_yaw_deg", None),
        getattr(base, "mr_eval_expected_delta_yaw_deg", None),
        getattr(mr, "expected_delta_yaw_deg", None),
        getattr(base, "expected_delta_yaw_deg", None),
    )
    return float(fallback_delta_theta if value is None else value)


def _analyze_semp2_z_axis_rotation_equivariance(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    delta_theta_deg: float,
    angle_tol_deg: float,
    pose_blindness_yaw_eps_deg: float,
    pose_blindness_delta_min_deg: float,
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
        yaw_s = _yaw_at_grasp(base)
        yaw_f = _yaw_at_grasp(mr)
        expected_delta = _expected_delta_theta(base, mr, delta_theta_deg)

        analyzable = yaw_s is not None and yaw_f is not None
        violated = False
        reasons = []

        yaw_delta_actual = None
        yaw_error = None
        if analyzable:
            yaw_delta_actual = float(yaw_f - yaw_s)
            yaw_error = _folded_yaw_error(yaw_delta_actual, expected_delta)
            if abs(yaw_delta_actual) < pose_blindness_yaw_eps_deg and abs(expected_delta) > pose_blindness_delta_min_deg:
                violated = True
                reasons.append(
                    "VIOLATION: Pose Blindness (Model ignored rotation and used default orientation)"
                )
            elif yaw_error > angle_tol_deg:
                violated = True
                reasons.append("VIOLATION: Rotation equivariance failure (Gripper Yaw mismatch)")
            else:
                reasons.append("z_axis_rotation_equivariance_pass")
        else:
            unavailable_count += 1
            reasons.append("missing_eef_yaw_at_grasp")

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "base_success": base_success,
                "mr_success": mr_success,
                "src_yaw_at_grasp_deg": yaw_s,
                "dst_yaw_at_grasp_deg": yaw_f,
                "expected_delta_theta_deg": expected_delta,
                "actual_delta_theta_deg": yaw_delta_actual,
                "yaw_error_deg": yaw_error,
                "angle_tol_deg": angle_tol_deg,
                "pose_blindness_yaw_eps_deg": pose_blindness_yaw_eps_deg,
                "pose_blindness_delta_min_deg": pose_blindness_delta_min_deg,
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
            "delta_theta_deg": float(delta_theta_deg),
            "angle_tol_deg": float(angle_tol_deg),
            "pose_blindness_yaw_eps_deg": float(pose_blindness_yaw_eps_deg),
            "pose_blindness_delta_min_deg": float(pose_blindness_delta_min_deg),
            "symmetry_period_deg": 90.0,
            "metric": "gripper_yaw_delta_with_90deg_symmetry",
            "violation_rule": "abs(actual_delta_theta) < pose_blindness_yaw_eps_deg AND abs(expected_delta_theta) > pose_blindness_delta_min_deg OR folded_yaw_error > angle_tol_deg",
            "invariance_expectation": "gripper_yaw_change_should_match_cube_rotation_under_symmetry",
        },
        "details": details,
    }


@register_mr_rule("MR-SEMP2")
@register_mr_rule("MR-SEMP-2")
@register_mr_rule("mr_semp_2")
def analyze_mr_semp2_z_axis_rotation_equivariance(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    return _analyze_semp2_z_axis_rotation_equivariance(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-SEMP2-Z-AXIS-ROTATION-EQUIVARIANCE"),
        delta_theta_deg=float(kwargs.get("delta_theta_deg", SEMP2_DEFAULT_DELTA_THETA_DEG)),
        angle_tol_deg=float(kwargs.get("angle_tol_deg", SEMP2_ANGLE_TOL_DEG)),
        pose_blindness_yaw_eps_deg=float(
            kwargs.get("pose_blindness_yaw_eps_deg", SEMP2_POSE_BLINDNESS_YAW_EPS_DEG)
        ),
        pose_blindness_delta_min_deg=float(
            kwargs.get("pose_blindness_delta_min_deg", SEMP2_POSE_BLINDNESS_DELTA_MIN_DEG)
        ),
    )
