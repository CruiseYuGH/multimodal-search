"""text-to-image：fg-clip2 text → mm_image (+ mm_video 预留)。"""
from __future__ import annotations
from app.core.types import Hit
from app.embedders.clip_text_embedder import FgClip2TextEmbedder
from app.store.milvus_client import MilvusClient
from app.store.image_store import ImageStore
from config import settings


_client: MilvusClient | None = None
_fg_txt: FgClip2TextEmbedder | None = None
_img_store: ImageStore | None = None


def query_text_to_image(q: str, topk: int = settings.DEFAULT_TOPK,
                        include_video: bool = False,
                        walk_type: str | None = None) -> list[Hit]:
    """walk_type=None 时按 settings.FGCLIP2_LONG_TEXT_THRESHOLD 自动选 short/long。"""
    global _client, _fg_txt, _img_store
    if _client is None:
        _client = MilvusClient()
        _client.connect()
    if _fg_txt is None:
        _fg_txt = FgClip2TextEmbedder(use_fp16=True)
    if _img_store is None:
        _img_store = ImageStore(_client)
    
    q_vec = _fg_txt.embed_query_text(q, walk_type=walk_type)
    hits = _img_store.search(q_vec, topk=topk)
    
    if include_video:
        import warnings
        warnings.warn("video search reserved, not implemented")
    
    return hits
