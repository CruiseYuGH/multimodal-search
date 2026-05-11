"""fg-clip2 文本塔。仅用于 query_t2i / hybrid 的 text→image 分支。"""
from __future__ import annotations
import numpy as np
import torch
from app.core.types import Chunk
from app.embedders.base import BaseEmbedder
from app.core.exceptions import EmbeddingError
from config import settings


class FgClip2TextEmbedder(BaseEmbedder):
    space = "fg-clip2"
    dim = settings.FG_CLIP2_DIM   # 768

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 use_fp16: bool = True):
        self.model_path = model_path or str(settings.FG_CLIP2_PATH)
        self.device = device or settings.DEVICE
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise EmbeddingError("缺少 transformers 依赖") from e
        try:
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(self.device)
            self._model.eval()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            raise EmbeddingError(f"加载 fg-clip2 文本塔失败: {e}") from e

    def _pick_walk_type(self, text: str) -> tuple[str, int]:
        """长度阈值切换 short/long。"""
        if len(text) > settings.FGCLIP2_LONG_TEXT_THRESHOLD:
            return "long", settings.FGCLIP2_TEXT_LONG_MAX_LEN
        return "short", settings.FGCLIP2_TEXT_SHORT_MAX_LEN

    def _normalize(self, vecs: torch.Tensor) -> np.ndarray:
        vecs = vecs / vecs.norm(p=2, dim=-1, keepdim=True)
        return vecs.cpu().numpy().astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """索引侧不调用此函数；仅留作批量评估用。"""
        raise NotImplementedError("fg-clip2 text embedder 仅用于查询，不用于索引")

    def embed_query_text(self, text: str, walk_type: str | None = None) -> np.ndarray:
        """get_text_features(**inp, walk_type=...) → L2 归一化。"""
        self._ensure_loaded()
        if walk_type is None:
            walk_type, max_len = self._pick_walk_type(text)
        else:
            max_len = (settings.FGCLIP2_TEXT_LONG_MAX_LEN if walk_type == "long"
                       else settings.FGCLIP2_TEXT_SHORT_MAX_LEN)
        try:
            inp = self._tokenizer(
                [text.lower()],
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                feat = self._model.get_text_features(**inp, walk_type=walk_type)
            return self._normalize(feat)[0]
        except Exception as e:
            raise EmbeddingError(f"fg-clip2 文本编码失败: {e}") from e
