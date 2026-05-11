"""Preprocessor 抽象基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from app.core.types import Chunk


class BasePreprocessor(ABC):
    SUPPORTED_EXT: tuple[str, ...] = ()

    @abstractmethod
    def load(self, file_path: str) -> list[Chunk]:
        """单文件绝对路径 → Chunk 列表。不扫描目录。"""
