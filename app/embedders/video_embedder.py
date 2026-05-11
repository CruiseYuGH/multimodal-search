"""视频 embedder 预留位。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk
from app.core.exceptions import VideoReserved
from app.embedders.base import BaseEmbedder
from config import settings


class VideoEmbedder(BaseEmbedder):
    space = "fg-clip2"
    dim = settings.FG_CLIP2_DIM

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        raise VideoReserved("video embedder reserved")

    def embed_query_text(self, text: str) -> np.ndarray:
        raise VideoReserved("video embedder reserved")
