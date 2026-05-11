"""注册装饰器（避免循环导入）。"""
from __future__ import annotations
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from app.preprocess.base import BasePreprocessor

_REGISTRY: dict[str, Type["BasePreprocessor"]] = {}


def register(extensions: tuple[str, ...]):
    """装饰器：把 Preprocessor 类绑定到扩展名（小写、带点）。"""
    def deco(cls: Type["BasePreprocessor"]) -> Type["BasePreprocessor"]:
        for ext in extensions:
            _REGISTRY[ext.lower()] = cls
        return cls
    return deco


def get_registry() -> dict[str, Type["BasePreprocessor"]]:
    """供 registry.py 查询。"""
    return _REGISTRY
