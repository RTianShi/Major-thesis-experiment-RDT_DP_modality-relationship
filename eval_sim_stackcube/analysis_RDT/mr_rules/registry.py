from typing import Any, Callable, Dict

MR_RULE_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {}


def register_mr_rule(rule_id: str):
    def _decorator(fn: Callable[..., Dict[str, Any]]):
        MR_RULE_REGISTRY[rule_id] = fn
        return fn
    return _decorator


def _load_builtin_rules():
    # 导入即注册
    from . import default_rule  # noqa: F401
    from . import mr_cptmp_1  # noqa: F401
    from . import mr_cptmp_2  # noqa: F401
    from . import mr_drp_1  # noqa: F401
    from . import mr_fpdp_1  # noqa: F401
    from . import mr_fpdp_2  # noqa: F401
    from . import mr_gdip_1  # noqa: F401
    from . import mr_semp_1  # noqa: F401
    from . import mr_semp_2  # noqa: F401
    from . import mr_sesp_1  # noqa: F401
    from . import mr_sesp_2  # noqa: F401
    from . import mr_ltsep_1  # noqa: F401
    from . import mr_ltsep_2  # noqa: F401
    from . import mr_scdp_1  # noqa: F401
    from . import mr_scdp_2  # noqa: F401
    from . import mr_scdp_3  # noqa: F401
    from . import mr_a1  # noqa: F401
    from . import mr_a2  # noqa: F401
    from . import mr_b1  # noqa: F401
    from . import mr_c1  # noqa: F401
    from . import mr_d1  # noqa: F401
    from . import mr_d2  # noqa: F401
    from . import mr_d3  # noqa: F401
    from . import mr_e1  # noqa: F401
    from . import mr_sadp_1  # noqa: F401
    from . import mr1
    from . import mr2
    from . import mr3
    from . import mr4
    from . import mr5
    from . import mr6



_load_builtin_rules()
