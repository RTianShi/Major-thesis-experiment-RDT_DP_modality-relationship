import importlib
from typing import Any, Callable, Dict

MR_RULE_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {}


def register_mr_rule(rule_id: str):
    def _decorator(fn: Callable[..., Dict[str, Any]]):
        MR_RULE_REGISTRY[rule_id] = fn
        return fn
    return _decorator


def _load_builtin_rules():
    # 导入即注册
    module_names = [
        "default_rule",
        "mr_sadp_2",
        "mr_cptmp_1",
        "mr_cptmp_2",
        "mr_drp_1",
        "mr_fpdp_1",
        "mr_fpdp_2",
        "mr_gdip_1",
        "mr_semp_1",
        "mr_semp_2",
        "mr_sesp_1",
        "mr_sesp_2",
        "mr_ltsep_1",
        "mr_ltsep_2",
        "mr_scdp_1",
        "mr_scdp_2",
        "mr_scdp_3",
        "mr2",
        "mr3",
        "mr4",
        "mr6",
    ]
    pkg = __name__.rsplit(".", 1)[0]
    for module_name in module_names:
        importlib.import_module(f"{pkg}.{module_name}")


_load_builtin_rules()
