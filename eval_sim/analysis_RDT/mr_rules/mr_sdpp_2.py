from typing import Any, Dict, List, Optional

import numpy as np

from .registry import register_mr_rule


def _pair_key(record: Any):
    return record.seed if getattr(record, "seed", None) is not None else getattr(record, "episode_id", None)


def _flatten_path(path: Any) -> List[Any]:
    if not path:
        return []
    if isinstance(path[0], list) and path[0] and isinstance(path[0][0], list):
        return [p[0] for p in path if p and isinstance(p[0], list)]
    return path


def _to_xyz(point: Any) -> Optional[List[float]]:
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64)
    if arr.ndim == 0:
        return None
    arr = arr.reshape(-1)
    if arr.size < 3:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _path_length(path: Any) -> Optional[float]:
    seq = _flatten_path(path)
    if len(seq) < 2:
        return 0.0 if seq else None
    pts = []
    for p in seq:
        xyz = _to_xyz(p)
        if xyz is None:
            return None
        pts.append(xyz)
    arr = np.asarray(pts, dtype=np.float64)
    delta = np.diff(arr[:, :3], axis=0)
    return float(np.linalg.norm(delta, axis=1).sum())


@register_mr_rule("MR-SDPP-2")
@register_mr_rule("MR-SDPP2")
def analyze_mr_sdpp_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    path_length_threshold_m = float(kwargs.get("path_length_threshold_m", 0.8))

    bmap = {_pair_key(record): record for record in base_records}
    mmap = {_pair_key(record): record for record in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for key in keys:
        base = bmap[key]
        mr = mmap[key]

        mr_path_length_m = _path_length(getattr(mr, "eef_path", None))
        source_success = getattr(base, "success", None)
        followup_success = getattr(mr, "success", None)

        analyzable = True
        violated = False
        reasons: List[str] = []

        if mr_path_length_m is None:
            analyzable = False
            reasons.append("missing_followup_path_length")
        elif mr_path_length_m > path_length_threshold_m:
            violated = True
            reasons.append("VIOLATION: Macro-Action Trigger (Robot executed full task sequence despite goal already met)")
        else:
            reasons.append("Stationary_After_Already_Satisfied")

        if not analyzable:
            unavailable_count += 1
            violated = False
        elif violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": key,
            "source_success": source_success,
            "followup_success": followup_success,
            "followup_path_length_m": mr_path_length_m,
            "path_length_threshold_m": path_length_threshold_m,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate_percent = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-SDPP-2",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate_percent,
        "config": {
            "path_length_threshold_m": path_length_threshold_m,
        },
        "details": details,
    }
