"""hybrid：多路并行检索 + RRF/加权融合。"""
from __future__ import annotations
from app.core.types import Hit
from config import settings


def query_hybrid(
    q: str | None = None,
    image_path: str | None = None,
    topk: int = settings.DEFAULT_TOPK,
    weights: dict[str, float] | None = None,
    fusion: str = "rrf",
    include_video: bool = False,
) -> list[Hit]:
    """组合 query_text + query_t2i + query_image，按 fusion 融合。

    weights: {"doc": 1.0, "image": 1.0, "t2i": 1.0}，fusion="weighted" 时生效。
    """
    raise NotImplementedError
