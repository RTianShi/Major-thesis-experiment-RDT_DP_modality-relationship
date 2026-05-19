from typing import Any, Dict, List

from .mr_sadp_1 import _analyze_sadp_invariance
from .registry import register_mr_rule


SADP3_GRASP_TOL_M = 0.05
SADP3_FINAL_TOL_M = 0.05


@register_mr_rule("MR-SADP-3")
def analyze_mr_sadp_3(base_records: List[Any], mr_records: List[Any], **kwargs) -> Dict[str, Any]:
    return _analyze_sadp_invariance(
        base_records,
        mr_records,
        mr_id="MR-SADP-3",
        position_tol_m=float(kwargs.get("sadp_grasp_tol", kwargs.get("position_tol", SADP3_GRASP_TOL_M))),
        final_tol_m=float(kwargs.get("sadp_final_tol", kwargs.get("position_tol", SADP3_FINAL_TOL_M))),
    )
