"""image 查询 → fg-clip2 image → mm_image (+ mm_video 预留)。"""
from __future__ import annotations
from app.core.types import Hit
from config import settings


def query_image(image_path: str, topk: int = settings.DEFAULT_TOPK,
                include_video: bool = False) -> list[Hit]:
    raise NotImplementedError
