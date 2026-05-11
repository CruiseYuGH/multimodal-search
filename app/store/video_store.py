"""mm_video collection 占位，方法全 raise VideoReserved。"""
from __future__ import annotations
import numpy as np
from app.core.types import Chunk, Hit
from app.core.exceptions import VideoReserved
from app.store.milvus_client import MilvusClient
from config import settings


class VideoStore:
    collection_name = settings.COLLECTION_VIDEO

    def __init__(self, client: MilvusClient):
        self.client = client

    def insert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        raise VideoReserved("video store reserved")

    def delete_by_path(self, source_path: str) -> int:
        raise VideoReserved("video store reserved")

    def exists(self, file_hash: str) -> bool:
        raise VideoReserved("video store reserved")

    def search(self, query_vec: np.ndarray, topk: int = settings.DEFAULT_TOPK,
               expr: str | None = None) -> list[Hit]:
        raise VideoReserved("video store reserved")
