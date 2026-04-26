from typing import Any, Dict, List

from .mr_sadp_1 import _analyze_sadp_invariance
from .registry import register_mr_rule


@register_mr_rule("MR-SADP-2")
def analyze_mr_sadp_2(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_sadp_invariance(
        base_records,
        mr_records,
        mr_id="MR-SADP-2",
        position_tol_m=float(kwargs.get("sadp_grasp_tol", kwargs.get("position_tol", 0.02))),
        final_tol_m=float(kwargs.get("sadp_final_tol", kwargs.get("position_tol", 0.02))),
    )
