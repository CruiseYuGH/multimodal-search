"""分数融合：RRF / 加权。"""
from __future__ import annotations
from collections import defaultdict
from app.core.types import Hit
from config import settings


def rrf(result_lists: list[list[Hit]], k: int = settings.RRF_K,
        topk: int = settings.DEFAULT_TOPK) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank_i))。"""
    scores: dict[str, float] = defaultdict(float)
    hit_map: dict[str, Hit] = {}
    
    for hits in result_lists:
        for rank, h in enumerate(hits, start=1):
            key = f"{h.source_path}|{h.chunk_id or ''}"
            scores[key] += 1.0 / (k + rank)
            if key not in hit_map:
                hit_map[key] = h
    
    # 按 RRF 分数排序
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[Hit] = []
    for key, score in ranked[:topk]:
        h = hit_map[key]
        # 用 RRF 分数覆盖原始分数
        out.append(Hit(
            source_path=h.source_path,
            modality=h.modality,
            score=score,
            page=h.page,
            slide_no=h.slide_no,
            sheet_name=h.sheet_name,
            frame_idx=h.frame_idx,
            preview=h.preview,
            chunk_id=h.chunk_id,
            extra=h.extra,
        ))
    return out


def weighted_sum(result_lists: list[list[Hit]], weights: list[float],
                 topk: int = settings.DEFAULT_TOPK) -> list[Hit]:
    """每路分数 min-max 归一化后按权重相加。"""
    if len(result_lists) != len(weights):
        raise ValueError(f"result_lists 与 weights 长度不匹配: {len(result_lists)} vs {len(weights)}")
    
    # 归一化每路分数到 [0,1]
    normalized: list[list[tuple[Hit, float]]] = []
    for hits in result_lists:
        if not hits:
            normalized.append([])
            continue
        scores = [h.score for h in hits]
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-9:
            norm = [(h, 1.0) for h in hits]
        else:
            norm = [(h, (h.score - min_s) / (max_s - min_s)) for h in hits]
        normalized.append(norm)
    
    # 加权合并
    combined: dict[str, tuple[Hit, float]] = {}
    for (hits_norm, w) in zip(normalized, weights):
        for h, ns in hits_norm:
            key = f"{h.source_path}|{h.chunk_id or ''}"
            if key in combined:
                combined[key] = (combined[key][0], combined[key][1] + w * ns)
            else:
                combined[key] = (h, w * ns)
    
    ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
    out: list[Hit] = []
    for h, score in ranked[:topk]:
        out.append(Hit(
            source_path=h.source_path,
            modality=h.modality,
            score=score,
            page=h.page,
            slide_no=h.slide_no,
            sheet_name=h.sheet_name,
            frame_idx=h.frame_idx,
            preview=h.preview,
            chunk_id=h.chunk_id,
            extra=h.extra,
        ))
    return out
