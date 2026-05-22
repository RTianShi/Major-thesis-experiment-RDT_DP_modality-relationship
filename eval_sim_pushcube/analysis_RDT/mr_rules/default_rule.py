from typing import Any, Dict, List

from .registry import register_mr_rule


def _index_by_seed_or_episode(records: List[Any]) -> Dict[int, Any]:
    out = {}
    for r in records:
        key = r.seed if r.seed is not None else r.episode_id
        if key is not None:
            out[key] = r
    return out


def compare_mr(
    base_records: List[Any],
    mr_records: List[Any],
    path_len_ratio_tol: float = 1.5,
) -> Dict[str, Any]:
    bmap = _index_by_seed_or_episode(base_records)
    mmap = _index_by_seed_or_episode(mr_records)
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    pairs = []
    violations = 0
    success_drop_count = 0

    for k in keys:
        b = bmap[k]
        m = mmap[k]

        b_succ = bool(b.success) if b.success is not None else False
        m_succ = bool(m.success) if m.success is not None else False

        violated = False
        reason = []

        if b_succ and (not m_succ):
            violated = True
            success_drop_count += 1
            reason.append("base_success_mr_fail")

        if b_succ and m_succ and b.path_len is not None and m.path_len is not None and b.path_len > 1e-9:
            ratio = m.path_len / b.path_len
            if ratio > path_len_ratio_tol:
                violated = True
                reason.append(f"path_len_ratio>{path_len_ratio_tol:.2f}")

        if violated:
            violations += 1

        pairs.append(
            {
                "key(seed_or_episode)": k,
                "base_success": b_succ,
                "mr_success": m_succ,
                "base_path_len_m": b.path_len,
                "mr_path_len_m": m.path_len,
                "violated": violated,
                "reason": reason,
                "base_file": b.file,
                "mr_file": m.file,
            }
        )

    violation_rate = (violations / len(keys) * 100.0) if keys else None

    return {
        "paired_episodes": len(keys),
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "success_drop_count": success_drop_count,
        "details": pairs,
    }


@register_mr_rule("default")
def analyze_mr_default(base_records, mr_records, **kwargs):
    path_len_ratio_tol = float(kwargs.get("path_len_ratio_tol", 1.5))
    out = compare_mr(base_records, mr_records, path_len_ratio_tol=path_len_ratio_tol)
    out["mr_id"] = "default"
    return out
