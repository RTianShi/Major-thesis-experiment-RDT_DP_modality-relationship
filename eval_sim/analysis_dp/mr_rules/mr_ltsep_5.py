from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mr_sadp_1 import _first_not_none, _nested_get
from .registry import register_mr_rule


LTSEP5_MAX_PATH_LEN_RATIO = 1.5
LTSEP5_MAX_FINAL_GOAL_DELTA_M = 0.05
LTSEP5_DEFAULT_DELAY_MS = 200.0
LTSEP5_DEFAULT_CONTROL_FREQ = 20.0


def _goal_point(record: Any) -> Optional[List[float]]:
    goal = _first_not_none(
        _nested_get(record, "goal_point"),
        _nested_get(record, "mr_eval", "goal_point"),
        _nested_get(record, "trajectory", "goal_point"),
    )
    if goal is None:
        return None
    try:
        arr = np.asarray(goal, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = _first_not_none(
        _nested_get(record, "trajectory"),
        _nested_get(record, "trajectory", "eef_path"),
        _nested_get(record, "mr_eval", "trajectory_after_phantom_grasp"),
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


def _path_len(points: List[List[float]]) -> Optional[float]:
    if not points:
        return None
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return float(np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1).sum())


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _delay_steps(record: Any, kwargs: Dict[str, Any]) -> Tuple[int, float, float, str]:
    raw_delay_steps = _first_not_none(
        _nested_get(record, "mr_eval", "vision_delay_steps"),
        _nested_get(record, "mr_eval", "delay_steps"),
        kwargs.get("delay_steps"),
    )
    if raw_delay_steps is not None:
        try:
            steps = max(0, int(raw_delay_steps))
            delay_ms = float(_first_not_none(kwargs.get("delay_ms"), LTSEP5_DEFAULT_DELAY_MS))
            control_freq = float(_first_not_none(kwargs.get("control_freq"), LTSEP5_DEFAULT_CONTROL_FREQ))
            return steps, delay_ms, control_freq, "annotated_delay_steps"
        except Exception:
            pass

    delay_ms = float(_first_not_none(
        _nested_get(record, "mr_eval", "vision_delay_ms"),
        _nested_get(record, "mr_eval", "delay_ms"),
        kwargs.get("delay_ms"),
        LTSEP5_DEFAULT_DELAY_MS,
    ))
    control_freq = float(_first_not_none(
        _nested_get(record, "mr_eval", "control_freq"),
        kwargs.get("control_freq"),
        LTSEP5_DEFAULT_CONTROL_FREQ,
    ))
    steps = max(0, int(round(delay_ms * control_freq / 1000.0)))
    return steps, delay_ms, control_freq, "derived_from_delay_ms_and_control_freq"


def _analyze_ltsep5(
    base_records: List[Any],
    mr_records: List[Any],
    *,
    mr_id: str,
    max_path_len_ratio: float,
    max_final_goal_delta_m: float,
    delay_ms: float,
    control_freq: float,
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

        goal_point = _first_not_none(_goal_point(base), _goal_point(mr))
        base_traj = _trajectory_points(base)
        mr_traj = _trajectory_points(mr)

        delay_steps, used_delay_ms, used_control_freq, delay_source = _delay_steps(
            mr,
            {
                "delay_ms": delay_ms,
                "control_freq": control_freq,
            },
        )

        base_path_len = _path_len(base_traj)
        mr_path_len = _path_len(mr_traj)

        base_final_point = base_traj[-1] if base_traj else None
        mr_final_point = mr_traj[-1] if mr_traj else None

        base_final_goal_dist = _point_l2(goal_point, base_final_point)
        mr_final_goal_dist = _point_l2(goal_point, mr_final_point)

        path_len_ratio = None
        final_goal_delta_m = None
        is_conservative = None
        analyzable = True
        violated = False
        reasons = []

        if goal_point is None:
            analyzable = False
            reasons.append("missing_goal_point")
        if base_path_len is None:
            analyzable = False
            reasons.append("missing_base_path")
        if mr_path_len is None:
            analyzable = False
            reasons.append("missing_mr_path")
        if base_final_goal_dist is None:
            analyzable = False
            reasons.append("missing_base_final_goal_distance")
        if mr_final_goal_dist is None:
            analyzable = False
            reasons.append("missing_mr_final_goal_distance")

        if analyzable:
            if base_path_len <= 1e-8:
                analyzable = False
                reasons.append("base_path_too_short")
            else:
                path_len_ratio = float(mr_path_len / base_path_len)
                final_goal_delta_m = float(mr_final_goal_dist - base_final_goal_dist)
                is_conservative = (path_len_ratio <= max_path_len_ratio) and (final_goal_delta_m <= max_final_goal_delta_m)

                if not is_conservative:
                    violated = True
                    if path_len_ratio > max_path_len_ratio:
                        reasons.append(f"excessive_degradation(path_len_ratio={path_len_ratio:.4f} > {max_path_len_ratio:.4f})")
                    if final_goal_delta_m > max_final_goal_delta_m:
                        reasons.append(f"goal_regression(final_goal_delta_m={final_goal_delta_m:.4f} > {max_final_goal_delta_m:.4f})")
                else:
                    reasons.append("conservative_degradation")

        if not analyzable:
            unavailable_count += 1
            violated = False

        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": k,
                "src_success": src_success,
                "dst_success": dst_success,
                "delay_steps": delay_steps,
                "delay_ms": used_delay_ms,
                "control_freq_hz": used_control_freq,
                "delay_source": delay_source,
                "goal_point": goal_point,
                "base_path_len_m": base_path_len,
                "mr_path_len_m": mr_path_len,
                "base_final_goal_dist_m": base_final_goal_dist,
                "mr_final_goal_dist_m": mr_final_goal_dist,
                "path_len_ratio": path_len_ratio,
                "final_goal_delta_m": final_goal_delta_m,
                "max_path_len_ratio": max_path_len_ratio,
                "max_final_goal_delta_m": max_final_goal_delta_m,
                "is_conservative": is_conservative,
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
            "delay_ms": delay_ms,
            "control_freq_hz": control_freq,
            "max_path_len_ratio": max_path_len_ratio,
            "max_final_goal_delta_m": max_final_goal_delta_m,
            "invariance_proxy": "temporal_misalignment_conservative_degradation",
        },
        "details": details,
    }


@register_mr_rule("MR-LTSEP-5")
@register_mr_rule("MR-LTSEP5")
def analyze_mr_ltsep_5(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_ltsep5(
        base_records,
        mr_records,
        mr_id=kwargs.get("mr_id", "MR-LTSEP-5"),
        max_path_len_ratio=float(kwargs.get("max_path_len_ratio", LTSEP5_MAX_PATH_LEN_RATIO)),
        max_final_goal_delta_m=float(kwargs.get("max_final_goal_delta_m", LTSEP5_MAX_FINAL_GOAL_DELTA_M)),
        delay_ms=float(kwargs.get("delay_ms", LTSEP5_DEFAULT_DELAY_MS)),
        control_freq=float(kwargs.get("control_freq", LTSEP5_DEFAULT_CONTROL_FREQ)),
    )
