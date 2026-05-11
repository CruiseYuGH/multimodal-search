"""BGE-M3 文本 embedder。模型内置 L2 归一化。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk
from app.embedders.base import BaseEmbedder
from config import settings


class BgeM3TextEmbedder(BaseEmbedder):
    space = "bge-m3"
    dim = settings.BGE_M3_DIM   # 1024

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 backend: str = "flagembedding"):
        """backend 仅支持 'flagembedding'（FlagEmbedding.BGEM3FlagModel）。"""
        self.model_path = model_path or str(settings.BGE_M3_PATH)
        self.device = device or settings.DEVICE
        self.backend = backend
        self._model = None  # 懒加载

    def _ensure_loaded(self) -> None:
        """加载 FlagEmbedding.BGEM3FlagModel(model_path, use_fp16=...)。"""
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """对 modality=TEXT 的 chunk 批量编码（model 已 normalize，不重复除）。"""
        raise NotImplementedError

    def embed_query_text(self, text: str) -> np.ndarray:
        raise NotImplementedError
