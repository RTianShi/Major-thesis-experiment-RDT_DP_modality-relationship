from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[List[float]]:
    traj = getattr(record, "eef_path", None) or []
    points: List[List[float]] = []
    for point in traj:
        try:
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return points


def _path_length_m(record: Any) -> Optional[float]:
    path_len = getattr(record, "path_len", None)
    if path_len is not None:
        try:
            return float(path_len)
        except Exception:
            pass

    points = _trajectory_points(record)
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _average_velocity_m_per_step(record: Any) -> Optional[float]:
    path_len = _path_length_m(record)
    total_steps = getattr(record, "total_steps", None)
    if path_len is None or total_steps is None:
        return None
    try:
        total_steps = int(total_steps)
    except Exception:
        return None
    if total_steps <= 0:
        return None
    return float(path_len / total_steps)


@register_mr_rule("MR5")
@register_mr_rule("MR-5")
@register_mr_rule("Instruction-Specialization")
@register_mr_rule("mr5")
def analyze_mr5_instruction_specialization(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        base_avg_velocity = _average_velocity_m_per_step(base)
        mr_avg_velocity = _average_velocity_m_per_step(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if base_avg_velocity is None:
            analyzable = False
            reasons.append("missing_base_average_velocity")
        if mr_avg_velocity is None:
            analyzable = False
            reasons.append("missing_mr_average_velocity")

        if analyzable:
            if mr_avg_velocity >= base_avg_velocity:
                violated = True
                reasons.append(
                    "VIOLATION: Specialized 'slowly' instruction did not reduce average velocity "
                    f"(followup={mr_avg_velocity:.6f} >= base={base_avg_velocity:.6f})"
                )
            else:
                reasons.append("followup_average_velocity_reduced")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": getattr(base, "success", None),
                "mr_success": getattr(mr, "success", None),
                "base_total_steps": getattr(base, "total_steps", None),
                "mr_total_steps": getattr(mr, "total_steps", None),
                "base_path_len_m": _path_length_m(base),
                "mr_path_len_m": _path_length_m(mr),
                "base_avg_velocity_m_per_step": base_avg_velocity,
                "mr_avg_velocity_m_per_step": mr_avg_velocity,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR5",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "metric": "average_velocity_m_per_step",
            "violation_rule": "AvgVelocity_f >= AvgVelocity_s",
        },
        "details": details,
    }
