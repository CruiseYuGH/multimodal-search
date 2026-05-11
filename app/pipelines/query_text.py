"""text 查询 → BGE-M3 → mm_doc。"""
from __future__ import annotations
from app.core.types import Hit
from app.embedders.text_embedder import BgeM3TextEmbedder
from app.store.milvus_client import MilvusClient
from app.store.doc_store import DocStore
from config import settings


_client: MilvusClient | None = None
_bge: BgeM3TextEmbedder | None = None
_doc_store: DocStore | None = None


def query_text(q: str, topk: int = settings.DEFAULT_TOPK) -> list[Hit]:
    global _client, _bge, _doc_store
    if _client is None:
        _client = MilvusClient()
        _client.connect()
    if _bge is None:
        _bge = BgeM3TextEmbedder(use_fp16=True)
    if _doc_store is None:
        _doc_store = DocStore(_client)
    
    q_vec = _bge.embed_query_text(q)
    return _doc_store.search(q_vec, topk=topk)
