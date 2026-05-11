"""hybrid：多路并行检索 + RRF/加权融合。"""
from __future__ import annotations
from app.core.types import Hit
from app.pipelines.query_text import query_text
from app.pipelines.query_image import query_image
from app.pipelines.query_t2i import query_text_to_image
from app.ranking.fusion import rrf, weighted_sum
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
    if not q and not image_path:
        raise ValueError("至少提供 q 或 image_path 之一")
    
    # 默认权重
    if weights is None:
        weights = {"doc": 1.0, "image": 1.0, "t2i": 1.0}
    
    result_lists: list[list[Hit]] = []
    weight_list: list[float] = []
    
    # 文本 → doc
    if q and weights.get("doc", 0) > 0:
        hits = query_text(q, topk=topk * 2)  # 多取一些，融合后再截断
        result_lists.append(hits)
        weight_list.append(weights["doc"])
    
    # 文本 → image (t2i)
    if q and weights.get("t2i", 0) > 0:
        hits = query_text_to_image(q, topk=topk * 2, include_video=include_video)
        result_lists.append(hits)
        weight_list.append(weights["t2i"])
    
    # 图像 → image
    if image_path and weights.get("image", 0) > 0:
        hits = query_image(image_path, topk=topk * 2, include_video=include_video)
        result_lists.append(hits)
        weight_list.append(weights["image"])
    
    if not result_lists:
        return []
    
    if len(result_lists) == 1:
        return result_lists[0][:topk]
    
    # 融合
    if fusion == "rrf":
        return rrf(result_lists, topk=topk)
    elif fusion == "weighted":
        return weighted_sum(result_lists, weight_list, topk=topk)
    else:
        raise ValueError(f"未知 fusion 方法: {fusion}")
