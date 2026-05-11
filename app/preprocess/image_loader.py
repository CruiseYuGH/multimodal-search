"""image 文件 → 单个 IMAGE Chunk。"""
from __future__ import annotations
import numpy as np

from app.core.types import Chunk, Modality, SourceType
from app.core.decorator import register
from app.core.exceptions import PreprocessError
from app.preprocess.base import BasePreprocessor
from app.utils.hashing import sha256_file, make_chunk_id
from app.utils.io import ensure_absolute


@register((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"))
class ImageLoader(BasePreprocessor):
    SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    def load(self, file_path: str) -> list[Chunk]:
        path = ensure_absolute(file_path)
        try:
            from PIL import Image
        except ImportError as e:
            raise PreprocessError("缺少 Pillow 依赖") from e

        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.uint8)
                w, h = im.size
        except Exception as e:
            raise PreprocessError(f"解析图片失败: {path}: {e}") from e

        file_hash = sha256_file(path)
        return [Chunk(
            chunk_id=make_chunk_id(path, 0),
            source_path=path,
            source_type=SourceType.IMAGE,
            modality=Modality.IMAGE,
            file_hash=file_hash,
            image=arr,
            extra={"width": w, "height": h},
        )]
