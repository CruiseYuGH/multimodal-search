"""BGE-M3 文本 embedder。模型内置 L2 归一化。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk, Modality
from app.embedders.base import BaseEmbedder
from app.core.exceptions import EmbeddingError
from config import settings


class BgeM3TextEmbedder(BaseEmbedder):
    space = "bge-m3"
    dim = settings.BGE_M3_DIM   # 1024

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 use_fp16: bool = True):
        self.model_path = model_path or str(settings.BGE_M3_PATH)
        self.device = device or settings.DEVICE
        self.use_fp16 = use_fp16
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise EmbeddingError("缺少 FlagEmbedding 依赖") from e
        try:
            self._model = BGEM3FlagModel(
                self.model_path,
                use_fp16=self.use_fp16,
                device=self.device,
            )
        except Exception as e:
            raise EmbeddingError(f"加载 BGE-M3 失败: {e}") from e

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """对 modality=TEXT 的 chunk 批量编码（模型已 normalize，不重复除）。"""
        self._ensure_loaded()
        texts = [c.text for c in chunks if c.modality == Modality.TEXT and c.text]
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        try:
            # encode 返回 dict，取 dense_vecs (N, 1024)
            out = self._model.encode(texts, batch_size=32, max_length=8192)
            vecs = out["dense_vecs"]  # 已 L2 归一化
            return vecs.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"BGE-M3 编码失败: {e}") from e

    def embed_query_text(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        try:
            out = self._model.encode([text], batch_size=1, max_length=8192)
            vec = out["dense_vecs"][0]
            return vec.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"BGE-M3 查询编码失败: {e}") from e
