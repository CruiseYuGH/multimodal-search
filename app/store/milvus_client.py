"""Milvus 连接管理。"""
from __future__ import annotations
from pymilvus import connections, utility, Collection, CollectionSchema
from app.core.exceptions import StoreError
from config import settings


class MilvusClient:
    def __init__(self, host: str | None = None, port: int | None = None, alias: str = "default"):
        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.alias = alias
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=str(self.port),
            )
            self._connected = True
        except Exception as e:
            raise StoreError(f"Milvus 连接失败 {self.host}:{self.port}: {e}") from e

    def ping(self) -> bool:
        self.connect()
        try:
            return utility.get_server_version(using=self.alias) is not None
        except Exception:
            return False

    def ensure_collection(self, name: str, schema: CollectionSchema,
                          index_params: dict) -> Collection:
        """存在则返回；不存在则创建 + 建索引 + load。"""
        self.connect()
        try:
            if utility.has_collection(name, using=self.alias):
                coll = Collection(name, using=self.alias)
                if not utility.index_building_progress(name, using=self.alias):
                    # 索引未建或已删除，重建
                    coll.create_index("vector", index_params)
                coll.load()
                return coll
            # 创建
            coll = Collection(name, schema, using=self.alias)
            coll.create_index("vector", index_params)
            coll.load()
            return coll
        except Exception as e:
            raise StoreError(f"ensure_collection {name} 失败: {e}") from e

    def drop(self, name: str) -> None:
        self.connect()
        try:
            if utility.has_collection(name, using=self.alias):
                utility.drop_collection(name, using=self.alias)
        except Exception as e:
            raise StoreError(f"drop {name} 失败: {e}") from e

    def get(self, name: str) -> Collection:
        self.connect()
        if not utility.has_collection(name, using=self.alias):
            raise StoreError(f"collection {name} 不存在")
        return Collection(name, using=self.alias)
