"""文件扩展名 → Preprocessor 类 的注册表（查询层）。"""
from __future__ import annotations
from pathlib import Path

from app.core.exceptions import UnsupportedFileType
from app.core.decorator import get_registry
from app.preprocess.base import BasePreprocessor


def get_preprocessor_for(file_path: str) -> BasePreprocessor:
    """根据扩展名取对应 Preprocessor 实例。"""
    registry = get_registry()
    ext = Path(file_path).suffix.lower()
    cls = registry.get(ext)
    if cls is None:
        raise UnsupportedFileType(f"未注册的扩展名：{ext} ({file_path})")
    return cls()


def list_registered() -> dict[str, str]:
    registry = get_registry()
    return {ext: cls.__name__ for ext, cls in registry.items()}
