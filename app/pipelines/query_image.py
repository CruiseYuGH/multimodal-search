"""image 查询 → fg-clip2 image → mm_image (+ mm_video 预留)。"""
from __future__ import annotations
from app.core.types import Hit
from app.embedders.image_embedder import FgClip2ImageEmbedder
from app.store.milvus_client import MilvusClient
from app.store.image_store import ImageStore
from config import settings


_client: MilvusClient | None = None
_fg_img: FgClip2ImageEmbedder | None = None
_img_store: ImageStore | None = None


def query_image(image_path: str, topk: int = settings.DEFAULT_TOPK,
                include_video: bool = False) -> list[Hit]:
    global _client, _fg_img, _img_store
    if _client is None:
        _client = MilvusClient()
        _client.connect()
    if _fg_img is None:
        _fg_img = FgClip2ImageEmbedder(use_fp16=True)
    if _img_store is None:
        _img_store = ImageStore(_client)
    
    q_vec = _fg_img.embed_query_image(image_path)
    hits = _img_store.search(q_vec, topk=topk)
    
    # include_video 预留：暂时返回空 + warning
    if include_video:
        import warnings
        warnings.warn("video search reserved, not implemented")
    
    return hits
