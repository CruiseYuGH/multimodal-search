"""Embedder 抽象基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from app.core.types import Chunk


class BaseEmbedder(ABC):
    space: str   # "bge-m3" | "fg-clip2"
    dim: int

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """批量 chunk → (N, dim) 已 L2 归一化。"""

    @abstractmethod
    def embed_query_text(self, text: str) -> np.ndarray:
        """单条查询文本 → (dim,) 已 L2 归一化。"""

    def embed_query_image(self, image_path: str) -> np.ndarray:
        """图像查询 → (dim,)。默认抛 NotImplementedError，由图像 embedder 覆盖。"""
        raise NotImplementedError
