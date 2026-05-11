"""索引 Pipeline：file_path → loader → embedder → store。"""
from __future__ import annotations
from typing import Any
from pathlib import Path

from app.core.registry import get_preprocessor_for
from app.core.types import Modality
from app.core.exceptions import VideoReserved
from app.embedders.text_embedder import BgeM3TextEmbedder
from app.embedders.image_embedder import FgClip2ImageEmbedder
from app.store.milvus_client import MilvusClient
from app.store.doc_store import DocStore
from app.store.image_store import ImageStore
from app.utils.io import ensure_absolute
from app.utils.hashing import sha256_file


# 全局单例（懒加载）
_client: MilvusClient | None = None
_bge: BgeM3TextEmbedder | None = None
_fg_img: FgClip2ImageEmbedder | None = None
_doc_store: DocStore | None = None
_img_store: ImageStore | None = None


def _get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient()
        _client.connect()
    return _client


def _get_bge() -> BgeM3TextEmbedder:
    global _bge
    if _bge is None:
        _bge = BgeM3TextEmbedder(use_fp16=True)
    return _bge


def _get_fg_img() -> FgClip2ImageEmbedder:
    global _fg_img
    if _fg_img is None:
        _fg_img = FgClip2ImageEmbedder(use_fp16=True)
    return _fg_img


def _get_doc_store() -> DocStore:
    global _doc_store
    if _doc_store is None:
        _doc_store = DocStore(_get_client())
    return _doc_store


def _get_img_store() -> ImageStore:
    global _img_store
    if _img_store is None:
        _img_store = ImageStore(_get_client())
    return _img_store


def index_file(file_path: str, skip_if_exists: bool = True) -> dict[str, Any]:
    """构建并执行索引流程。

    Returns:
        {
          "file_path": str,
          "file_hash": str,
          "modality_counts": {"text": N, "image": M},
          "inserted": int,
          "skipped": bool,
          "error": Optional[str],
        }
    """
    path = ensure_absolute(file_path)
    file_hash = sha256_file(path)
    
    result: dict[str, Any] = {
        "file_path": path,
        "file_hash": file_hash,
        "modality_counts": {},
        "inserted": 0,
        "skipped": False,
        "error": None,
    }
    
    try:
        # 1. 预处理
        loader = get_preprocessor_for(path)
        chunks = loader.load(path)
        if not chunks:
            result["error"] = "no chunks extracted"
            return result
        
        # 按 modality 分组
        text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
        image_chunks = [c for c in chunks if c.modality == Modality.IMAGE]
        result["modality_counts"] = {"text": len(text_chunks), "image": len(image_chunks)}
        
        # 2. 去重检查（可选）
        doc_store = _get_doc_store()
        img_store = _get_img_store()
        if skip_if_exists:
            if text_chunks and doc_store.exists(file_hash):
                result["skipped"] = True
                return result
            if image_chunks and img_store.exists(file_hash):
                result["skipped"] = True
                return result
        
        # 3. 向量化 + 入库
        inserted = 0
        if text_chunks:
            bge = _get_bge()
            vecs = bge.embed_chunks(text_chunks)
            n = doc_store.insert(text_chunks, vecs)
            inserted += n
        
        if image_chunks:
            fg_img = _get_fg_img()
            vecs = fg_img.embed_chunks(image_chunks)
            n = img_store.insert(image_chunks, vecs)
            inserted += n
        
        result["inserted"] = inserted
        return result
    
    except VideoReserved as e:
        result["error"] = f"video reserved: {e}"
        result["skipped"] = True
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result
