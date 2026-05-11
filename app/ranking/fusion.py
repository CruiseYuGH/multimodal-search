"""分数融合：RRF / 加权。"""
from __future__ import annotations
from app.core.types import Hit
from config import settings


def rrf(result_lists: list[list[Hit]], k: int = settings.RRF_K,
        topk: int = settings.DEFAULT_TOPK) -> list[Hit]:
    """Reciprocal Rank Fusion。"""
    raise NotImplementedError


def weighted_sum(result_lists: list[list[Hit]], weights: list[float],
                 topk: int = settings.DEFAULT_TOPK) -> list[Hit]:
    """每路分数 min-max 归一化后按权重相加。"""
    raise NotImplementedError
