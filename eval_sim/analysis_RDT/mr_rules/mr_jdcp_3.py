from .registry import register_mr_rule

from typing import List, Dict, Any


def _grasp_index(record: Any):
    derived = getattr(record, "derived_grasp_frame_index", None)
    if derived is not None:
        return int(derived)
    raw = getattr(record, "mr_eval_grasp_frame_index", None)
    if raw is not None:
        return int(raw)
    return None


@register_mr_rule("MR-JDCP-3")
def analyze_mr_jdcp_3(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    bmap = {(r.seed if r.seed is not None else r.episode_id): r for r in base_records}
    mmap = {(r.seed if r.seed is not None else r.episode_id): r for r in mr_records}
    keys = sorted(set(bmap.keys()) & set(mmap.keys()))

    details = []
    violations = 0
    unavailable_count = 0

    for k in keys:
        base = bmap[k]
        mr = mmap[k]

        dst_grasp_frame_index = _grasp_index(mr)
        analyzable = True
        violated = False
        reasons = []

        if dst_grasp_frame_index is not None:
            violated = True
            reasons.append("Action_Bias_Closed_Gripper_Without_Relevant_Instruction")
        else:
            reasons.append("Instruction_Sensitivity_No_Grasp_Attempt")

        if violated:
            violations += 1

        details.append({
            "key(seed_or_episode)": k,
            "source_success": getattr(base, "success", None),
            "followup_success": getattr(mr, "success", None),
            "dst_grasp_frame_index": dst_grasp_frame_index,
            "analyzable": analyzable,
            "violated": violated,
            "reasons": reasons,
        })

    analyzable_episodes = len(keys) - unavailable_count
    violation_rate = (violations / analyzable_episodes * 100.0) if analyzable_episodes > 0 else None

    return {
        "mr_id": "MR-JDCP-3",
        "paired_episodes": len(keys),
        "analyzable_episodes": analyzable_episodes,
        "unavailable_count": unavailable_count,
        "violations": violations,
        "violation_rate_percent": violation_rate,
        "config": {},
        "details": details,
    }
