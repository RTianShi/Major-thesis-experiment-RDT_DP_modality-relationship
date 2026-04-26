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
    from . import mr_drp_1  # noqa: F401
    from . import mr_drp_2  # noqa: F401
    from . import mr_semp_1  # noqa: F401
    from . import mr_semp_2  # noqa: F401
    from . import mr_sadp_1  # noqa: F401
    from . import mr_sadp_2  # noqa: F401


_load_builtin_rules()
