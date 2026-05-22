from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


CPTMP2_JITTER_RATIO_THRESHOLD = 3.0
CPTMP2_PATH_LENGTH_RATIO_THRESHOLD = 1.3
CPTMP2_START_FRACTION = 0.10
CPTMP2_END_FRACTION = 0.30


def _paired_key(record: Any) -> Any:
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _trajectory_points(record: Any) -> List[np.ndarray]:
    traj = getattr(record, "eef_path", None) or []
    points: List[np.ndarray] = []
    for point in traj:
        try:
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if arr.size >= 3:
            points.append(arr[:3])
    return points


def _path_length(points: List[np.ndarray]) -> Optional[float]:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _mask_window(points: List[np.ndarray], start_fraction: float, end_fraction: float) -> List[np.ndarray]:
    if not points:
        return []
    total = len(points)
    start_idx = max(0, int(round(total * start_fraction)))
    end_idx = max(start_idx + 1, int(round(total * end_fraction)))
    end_idx = min(total, end_idx)
    return points[start_idx:end_idx]


def _window_jitter_std(points: List[np.ndarray]) -> Optional[float]:
    if len(points) < 3:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    deltas = np.diff(arr[:, :3], axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    if speeds.size < 2:
        return None
    return float(np.std(speeds))


@register_mr_rule("MR-CPTMP2")
@register_mr_rule("MR-CPTMP-2")
@register_mr_rule("CPTMP-Approach-Phase-Transient-Masking")
def analyze_mr_cptmp2_approach_phase_transient_masking(
    base_records: List[Any], mr_records: List[Any], **kwargs
) -> Dict[str, Any]:
    jitter_ratio_threshold = float(kwargs.get("jitter_ratio_threshold", CPTMP2_JITTER_RATIO_THRESHOLD))
    path_length_ratio_threshold = float(
        kwargs.get("path_length_ratio_threshold", CPTMP2_PATH_LENGTH_RATIO_THRESHOLD)
    )
    start_fraction = float(kwargs.get("start_fraction", CPTMP2_START_FRACTION))
    end_fraction = float(kwargs.get("end_fraction", CPTMP2_END_FRACTION))

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
            reasons.append("missing_followup_eef_path")

        base_window = _mask_window(base_points, start_fraction, end_fraction)
        mr_window = _mask_window(mr_points, start_fraction, end_fraction)
        base_jitter = _window_jitter_std(base_window)
        mr_jitter = _window_jitter_std(mr_window)
        jitter_ratio = None

        if base_jitter is None:
            analyzable = False
            reasons.append("cannot_compute_base_window_jitter")
        if mr_jitter is None:
            analyzable = False
            reasons.append("cannot_compute_followup_window_jitter")

        if analyzable and base_jitter is not None and mr_jitter is not None:
            if base_jitter <= 1e-12:
                jitter_ratio = float("inf") if mr_jitter > 1e-12 else 1.0
            else:
                jitter_ratio = float(mr_jitter / base_jitter)
            if jitter_ratio > jitter_ratio_threshold:
                violated = True
                reasons.append(
                    "VIOLATION: Instantaneous Collapse "
                    f"(jitter_ratio={jitter_ratio:.6f} > {jitter_ratio_threshold:.6f})"
                )
            else:
                reasons.append("no_violent_shaking_during_mask_window")

        base_path_len = _path_length(base_points)
        mr_path_len = _path_length(mr_points)
        path_length_ratio = None
        if base_path_len is None or base_path_len <= 1e-12:
            analyzable = False
            reasons.append("cannot_compute_base_path_length")
        elif mr_path_len is None:
            analyzable = False
            reasons.append("cannot_compute_followup_path_length")
        else:
            path_length_ratio = float(mr_path_len / base_path_len)
            if path_length_ratio > path_length_ratio_threshold:
                violated = True
                reasons.append(
                    "VIOLATION: Significant Hesitation "
                    f"(path_length_ratio={path_length_ratio:.6f} > {path_length_ratio_threshold:.6f})"
                )
            else:
                reasons.append("no_significant_recovery_detour")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append(
            {
                "key(seed_or_episode)": key,
                "base_success": base_success,
                "mr_success": mr_success,
                "mask_start_fraction": start_fraction,
                "mask_end_fraction": end_fraction,
                "base_mask_window_len": len(base_window),
                "mr_mask_window_len": len(mr_window),
                "base_jitter_std": base_jitter,
                "mr_jitter_std": mr_jitter,
                "jitter_ratio": jitter_ratio,
                "jitter_ratio_threshold": jitter_ratio_threshold,
                "base_path_length_m": base_path_len,
                "mr_path_length_m": mr_path_len,
                "path_length_ratio": path_length_ratio,
                "path_length_ratio_threshold": path_length_ratio_threshold,
                "analyzable": analyzable,
                "violated": violated,
                "reasons": reasons,
            }
        )

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-CPTMP2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {
            "jitter_ratio_threshold": jitter_ratio_threshold,
            "path_length_ratio_threshold": path_length_ratio_threshold,
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "invariance_proxy": "mask_window_jitter_and_post_recovery_efficiency",
            "attack_type": "approach_phase_transient_visual_masking",
        },
        "details": details,
    }
