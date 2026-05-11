"""fg-clip2 图像塔。出口手动 L2 归一化。"""
from __future__ import annotations
import numpy as np
import torch
from app.core.types import Chunk, Modality
from app.embedders.base import BaseEmbedder
from app.core.exceptions import EmbeddingError
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

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 use_fp16: bool = True):
        self.model_path = model_path or str(settings.FG_CLIP2_PATH)
        self.device = device or settings.DEVICE
        self.use_fp16 = use_fp16
        self._model = None
        self._image_processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoImageProcessor
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
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        except Exception as e:
            raise EmbeddingError(f"加载 fg-clip2 失败: {e}") from e

    def _normalize(self, vecs: torch.Tensor) -> np.ndarray:
        """L2 归一化 + 转 numpy float32。"""
        vecs = vecs / vecs.norm(p=2, dim=-1, keepdim=True)
        return vecs.cpu().numpy().astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """对 modality=IMAGE 的 chunk 批量编码（每张图按尺寸选 max_num_patches，出口 L2）。"""
        self._ensure_loaded()
        image_chunks = [c for c in chunks if c.modality == Modality.IMAGE and c.image is not None]
        if not image_chunks:
            return np.empty((0, self.dim), dtype=np.float32)

        try:
            from PIL import Image
            vecs_list = []
            for c in image_chunks:
                # c.image 是 np.ndarray (H,W,3) uint8
                pil = Image.fromarray(c.image, mode="RGB")
                w, h = pil.size
                max_patches = determine_max_value(w, h)
                inp = self._image_processor(
                    images=pil,
                    max_num_patches=max_patches,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    feat = self._model.get_image_features(**inp)  # (1, 768)
                vecs_list.append(feat)
            all_vecs = torch.cat(vecs_list, dim=0)
            return self._normalize(all_vecs)
        except Exception as e:
            raise EmbeddingError(f"fg-clip2 图像编码失败: {e}") from e

    def embed_query_text(self, text: str) -> np.ndarray:
        """图像 embedder 不实现 text 编码，请使用 FgClip2TextEmbedder。"""
        raise NotImplementedError("use FgClip2TextEmbedder for text→image")

    def embed_query_image(self, image_path: str) -> np.ndarray:
        self._ensure_loaded()
        try:
            from PIL import Image
            pil = Image.open(image_path).convert("RGB")
            w, h = pil.size
            max_patches = determine_max_value(w, h)
            inp = self._image_processor(
                images=pil,
                max_num_patches=max_patches,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                feat = self._model.get_image_features(**inp)
            return self._normalize(feat)[0]
        except Exception as e:
            raise EmbeddingError(f"fg-clip2 查询图像编码失败: {e}") from e
