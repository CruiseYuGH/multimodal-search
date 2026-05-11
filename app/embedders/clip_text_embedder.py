"""fg-clip2 文本塔。仅用于 query_t2i / hybrid 的 text→image 分支。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk
from app.embedders.base import BaseEmbedder
from config import settings


class FgClip2TextEmbedder(BaseEmbedder):
    space = "fg-clip2"
    dim = settings.FG_CLIP2_DIM   # 768

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.model_path = model_path or str(settings.FG_CLIP2_PATH)
        self.device = device or settings.DEVICE
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """AutoModelForCausalLM + AutoTokenizer，trust_remote_code=True。"""
        raise NotImplementedError

    def _pick_walk_type(self, text: str) -> tuple[str, int]:
        """长度阈值切换 short/long。"""
        if len(text) > settings.FGCLIP2_LONG_TEXT_THRESHOLD:
            return "long", settings.FGCLIP2_TEXT_LONG_MAX_LEN
        return "short", settings.FGCLIP2_TEXT_SHORT_MAX_LEN

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """索引侧不调用此函数；仅留作批量评估用。"""
        raise NotImplementedError

    def embed_query_text(self, text: str) -> np.ndarray:
        """get_text_features(**inp, walk_type=...) → L2 归一化。"""
        raise NotImplementedError
