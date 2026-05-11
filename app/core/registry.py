"""文件扩展名 → Preprocessor 类 的注册表。"""
from __future__ import annotations
from typing import Type
from pathlib import Path

from app.core.exceptions import UnsupportedFileType
from app.preprocess.base import BasePreprocessor


_REGISTRY: dict[str, Type[BasePreprocessor]] = {}


def register(extensions: tuple[str, ...]):
    """装饰器：把 Preprocessor 类绑定到扩展名（小写、带点）。"""
    def deco(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
        for ext in extensions:
            _REGISTRY[ext.lower()] = cls
        return cls
    return deco


def get_preprocessor_for(file_path: str) -> BasePreprocessor:
    """根据扩展名取对应 Preprocessor 实例。"""
    ext = Path(file_path).suffix.lower()
    cls = _REGISTRY.get(ext)
    if cls is None:
        raise UnsupportedFileType(f"未注册的扩展名：{ext} ({file_path})")
    return cls()


def list_registered() -> dict[str, str]:
    return {ext: cls.__name__ for ext, cls in _REGISTRY.items()}
