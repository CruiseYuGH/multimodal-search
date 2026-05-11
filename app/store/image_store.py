"""mm_image collection。"""
from __future__ import annotations
import json
import numpy as np
from app.core.types import Chunk, Hit, Modality
from app.core.exceptions import StoreError
from app.store.milvus_client import MilvusClient
from app.store.schema import image_schema, HNSW_INDEX_PARAMS, HNSW_SEARCH_PARAMS
from config import settings


class ImageStore:
    collection_name = settings.COLLECTION_IMAGE

    def __init__(self, client: MilvusClient):
        self.client = client
        self._coll = None

    def _ensure_coll(self):
        if self._coll is None:
            self._coll = self.client.ensure_collection(
                self.collection_name, image_schema(), HNSW_INDEX_PARAMS
            )
        return self._coll

    def insert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks 与 vectors 长度不匹配: {len(chunks)} vs {len(vectors)}")
        coll = self._ensure_coll()
        entities = []
        for c, v in zip(chunks, vectors):
            entities.append({
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "file_hash": c.file_hash,
                "modality": c.modality.value,
                "source_type": c.source_type.value,
                "vector": v.tolist(),
                "slide_no": c.slide_no or -1,
                "extra_json": json.dumps(c.extra, ensure_ascii=False)[:1024],
            })
        try:
            coll.insert(entities)
            coll.flush()
            return len(entities)
        except Exception as e:
            raise StoreError(f"image_store insert 失败: {e}") from e

    def delete_by_path(self, source_path: str) -> int:
        coll = self._ensure_coll()
        try:
            expr = f'source_path == "{source_path}"'
            res = coll.delete(expr)
            coll.flush()
            return res.delete_count
        except Exception as e:
            raise StoreError(f"image_store delete_by_path 失败: {e}") from e

    def exists(self, file_hash: str) -> bool:
        coll = self._ensure_coll()
        try:
            expr = f'file_hash == "{file_hash}"'
            res = coll.query(expr, output_fields=["chunk_id"], limit=1)
            return len(res) > 0
        except Exception:
            return False

    def search(self, query_vec: np.ndarray, topk: int = settings.DEFAULT_TOPK,
               expr: str | None = None) -> list[Hit]:
        coll = self._ensure_coll()
        try:
            res = coll.search(
                data=[query_vec.tolist()],
                anns_field="vector",
                param=HNSW_SEARCH_PARAMS,
                limit=topk,
                expr=expr,
                output_fields=["source_path", "modality", "slide_no", "extra_json"],
            )
            hits: list[Hit] = []
            for r in res[0]:
                extra = {}
                try:
                    extra = json.loads(r.entity.get("extra_json") or "{}")
                except Exception:
                    pass
                hits.append(Hit(
                    source_path=r.entity.get("source_path"),
                    modality=Modality(r.entity.get("modality")),
                    score=float(r.distance),
                    slide_no=r.entity.get("slide_no") if r.entity.get("slide_no", -1) != -1 else None,
                    chunk_id=r.id,
                    extra=extra,
                ))
            return hits
        except Exception as e:
            raise StoreError(f"image_store search 失败: {e}") from e
