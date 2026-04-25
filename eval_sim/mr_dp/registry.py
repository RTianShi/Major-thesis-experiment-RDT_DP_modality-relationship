from typing import Callable, Dict

LANG_REGISTRY: Dict[str, Callable] = {}
VISION_REGISTRY: Dict[str, Callable] = {}
PROPRIO_REGISTRY: Dict[str, Callable] = {}


def register_language(name: str):
    def _wrap(fn: Callable):
        LANG_REGISTRY[name] = fn
        return fn
    return _wrap


def register_vision(name: str):
    def _wrap(fn: Callable):
        VISION_REGISTRY[name] = fn
        return fn
    return _wrap


def register_proprio(name: str):
    def _wrap(fn: Callable):
        PROPRIO_REGISTRY[name] = fn
        return fn
    return _wrap


def get_lang(name: str) -> Callable:
    if name not in LANG_REGISTRY:
        raise ValueError(f"Unknown language MR: {name}")
    return LANG_REGISTRY[name]


def get_vision(name: str) -> Callable:
    if name not in VISION_REGISTRY:
        raise ValueError(f"Unknown vision MR: {name}")
    return VISION_REGISTRY[name]


def get_proprio(name: str) -> Callable:
    if name not in PROPRIO_REGISTRY:
        raise ValueError(f"Unknown proprio MR: {name}")
    return PROPRIO_REGISTRY[name]
