"""fg-clip2 图像塔。出口手动 L2 归一化。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk
from app.embedders.base import BaseEmbedder
from config import settings


def determine_max_value(width: int, height: int) -> int:
    """fg-clip2 官方动态 patches 选择：见 README。"""
    max_val = (width // 16) * (height // 16)
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128


class FgClip2ImageEmbedder(BaseEmbedder):
    space = "fg-clip2"
    dim = settings.FG_CLIP2_DIM   # 768

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.model_path = model_path or str(settings.FG_CLIP2_PATH)
        self.device = device or settings.DEVICE
        self._model = None
        self._image_processor = None

    def _ensure_loaded(self) -> None:
        """AutoModelForCausalLM + AutoImageProcessor，trust_remote_code=True。"""
        raise NotImplementedError

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """对 modality=IMAGE 的 chunk 批量编码（每张图按尺寸选 max_num_patches，出口 L2）。"""
        raise NotImplementedError

    def embed_query_text(self, text: str) -> np.ndarray:
        """图像 embedder 不实现 text 编码，请使用 FgClip2TextEmbedder。"""
        raise NotImplementedError

    def embed_query_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError
