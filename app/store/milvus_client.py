"""Milvus 连接管理。"""
from __future__ import annotations
from pymilvus import connections, utility, Collection, CollectionSchema
from config import settings


class MilvusClient:
    def __init__(self, host: str | None = None, port: int | None = None, alias: str = "default"):
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.alias = alias

    def connect(self) -> None:
        raise NotImplementedError

    def ping(self) -> bool:
        raise NotImplementedError

    def ensure_collection(self, name: str, schema: CollectionSchema,
                          index_params: dict) -> Collection:
        """存在则返回；不存在则创建 + 建索引 + load。"""
        raise NotImplementedError

    def drop(self, name: str) -> None:
        raise NotImplementedError

    def get(self, name: str) -> Collection:
        raise NotImplementedError
