from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


CMSI1_PATH_LEN_RATIO_THRESHOLD = 1.25


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


def _path_length(points: List[List[float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


@register_mr_rule("MR-CMSI1")
@register_mr_rule("MR-CMSI-1")
@register_mr_rule("mr_cmsi_1")
def analyze_mr_cmsi1_saliency_distractor_injection(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    path_len_ratio_threshold = float(kwargs.get("path_len_ratio_threshold", CMSI1_PATH_LEN_RATIO_THRESHOLD))

    bmap = {_paired_key(r): r for r in (base_records or [])}
    mmap = {_paired_key(r): r for r in (mr_records or [])}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        base_success = getattr(base, "success", None)
        mr_success = getattr(mr, "success", None)
        base_points = _trajectory_points(base)
        mr_points = _trajectory_points(mr)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if not base_points:
            analyzable = False
            reasons.append("missing_base_eef_path")
        if not mr_points:
            analyzable = False
            reasons.append("missing_mr_eef_path")

        base_path_len = None
        mr_path_len = None
        path_len_ratio = None

        if analyzable:
            base_path_len = _path_length(base_points)
            mr_path_len = _path_length(mr_points)
            if base_path_len is None or base_path_len <= 1e-12:
                analyzable = False
                reasons.append("missing_base_path_length")
            if mr_path_len is None:
                analyzable = False
                reasons.append("missing_mr_path_length")

        if analyzable:
            path_len_ratio = float(mr_path_len / base_path_len)
            if path_len_ratio > path_len_ratio_threshold:
                violated = True
                reasons.append(
                    "VIOLATION: Significant Hesitation "
                    f"(path_len_ratio={path_len_ratio:.6f} > {path_len_ratio_threshold:.6f})"
                )
            else:
                reasons.append("trajectory_consistent_under_saliency_distractors")

        if not analyzable:
            unavailable_count += 1
            violated = False
        if violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": base_success,
                "mr_success": mr_success,
                "base_path_len_m": base_path_len,
                "mr_path_len_m": mr_path_len,
                "path_len_ratio": path_len_ratio,
                "path_len_ratio_threshold": path_len_ratio_threshold,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-CMSI1",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "path_len_ratio_threshold": path_len_ratio_threshold,
            "metric": "path_len_ratio = path_length_f / path_length_s",
            "violation_rule": "path_len_ratio > threshold",
        },
        "details": details,
    }
