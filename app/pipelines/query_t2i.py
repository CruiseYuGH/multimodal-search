"""text-to-image：fg-clip2 text → mm_image (+ mm_video 预留)。"""
from __future__ import annotations
from app.core.types import Hit
from config import settings


def query_text_to_image(q: str, topk: int = settings.DEFAULT_TOPK,
                        include_video: bool = False,
                        walk_type: str | None = None) -> list[Hit]:
    """walk_type=None 时按 settings.FGCLIP2_LONG_TEXT_THRESHOLD 自动选 short/long。"""
    raise NotImplementedError
