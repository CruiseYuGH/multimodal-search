"""mm_doc collection 的 insert / search / delete。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk, Hit
from app.store.milvus_client import MilvusClient
from config import settings


class DocStore:
    collection_name = settings.COLLECTION_DOC

    def __init__(self, client: MilvusClient):
        self.client = client

    def insert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """vectors 已 L2 归一化；返回插入条数。"""
        raise NotImplementedError

    def delete_by_path(self, source_path: str) -> int:
        raise NotImplementedError

    def exists(self, file_hash: str) -> bool:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, topk: int = settings.DEFAULT_TOPK,
               expr: str | None = None) -> list[Hit]:
        raise NotImplementedError
