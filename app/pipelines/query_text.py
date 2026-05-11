"""text 查询 → BGE-M3 → mm_doc。"""
from __future__ import annotations
from app.core.types import Hit
from config import settings


def query_text(q: str, topk: int = settings.DEFAULT_TOPK) -> list[Hit]:
    raise NotImplementedError
