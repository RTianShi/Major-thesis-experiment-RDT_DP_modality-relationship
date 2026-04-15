from typing import Callable, Dict

ENV_REGISTRY: Dict[str, Callable] = {}


def register_env(name: str):
    def _wrap(fn: Callable):
        ENV_REGISTRY[name] = fn
        return fn
    return _wrap


def get_env(name: str) -> Callable:
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env MR: {name}")
    return ENV_REGISTRY[name]
