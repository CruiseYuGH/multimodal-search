"""D 阶段冒烟：Milvus 连接 + doc/image store insert/search/delete。

依赖：Milvus standalone 已启动（bash scripts/start_milvus.sh）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.store.milvus_client import MilvusClient
from app.store.doc_store import DocStore
from app.store.image_store import ImageStore
from app.core.types import Chunk, Modality, SourceType
from app.utils.hashing import make_chunk_id
from config import settings


def main() -> int:
    print(f"[milvus] {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    client = MilvusClient()
    try:
        client.connect()
        if not client.ping():
            print("[FAIL] Milvus 不可达，请先启动：bash scripts/start_milvus.sh")
            return 1
        print("[OK] Milvus 连接成功")
    except Exception as e:
        print(f"[FAIL] 连接失败: {e}")
        return 1

    rc = 0

    # ---- doc_store ----
    print("\n[1/2] DocStore")
    try:
        doc_store = DocStore(client)
        # 插入 3 个文本 chunk
        chunks = [
            Chunk(
                chunk_id=make_chunk_id("/tmp/test.txt", i),
                source_path="/tmp/test.txt",
                source_type=SourceType.TXT,
                modality=Modality.TEXT,
                file_hash="hash_doc",
                text=f"测试文档第 {i} 段",
                preview=f"测试文档第 {i} 段",
                page=i+1,
            )
            for i in range(3)
        ]
        vecs = np.random.randn(3, settings.BGE_M3_DIM).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        n = doc_store.insert(chunks, vecs)
        print(f"  insert: {n} chunks")
        # 查询
        q_vec = np.random.randn(settings.BGE_M3_DIM).astype(np.float32)
        q_vec = q_vec / np.linalg.norm(q_vec)
        hits = doc_store.search(q_vec, topk=2)
        print(f"  search: {len(hits)} hits")
        for h in hits:
            print(f"    - {h.source_path} page={h.page} score={h.score:.4f}")
        # exists
        exists = doc_store.exists("hash_doc")
        print(f"  exists(hash_doc): {exists}")
        # delete
        deleted = doc_store.delete_by_path("/tmp/test.txt")
        print(f"  delete: {deleted} chunks")
        print("  [OK] DocStore")
    except Exception as e:
        print(f"  [FAIL] DocStore: {type(e).__name__}: {e}")
        rc = 1

    # ---- image_store ----
    print("\n[2/2] ImageStore")
    try:
        img_store = ImageStore(client)
        chunks = [
            Chunk(
                chunk_id=make_chunk_id("/tmp/test.png", i),
                source_path="/tmp/test.png",
                source_type=SourceType.IMAGE,
                modality=Modality.IMAGE,
                file_hash="hash_img",
                image=np.zeros((240, 320, 3), dtype=np.uint8),
            )
            for i in range(2)
        ]
        vecs = np.random.randn(2, settings.FG_CLIP2_DIM).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        n = img_store.insert(chunks, vecs)
        print(f"  insert: {n} chunks")
        q_vec = np.random.randn(settings.FG_CLIP2_DIM).astype(np.float32)
        q_vec = q_vec / np.linalg.norm(q_vec)
        hits = img_store.search(q_vec, topk=2)
        print(f"  search: {len(hits)} hits")
        for h in hits:
            print(f"    - {h.source_path} score={h.score:.4f}")
        exists = img_store.exists("hash_img")
        print(f"  exists(hash_img): {exists}")
        deleted = img_store.delete_by_path("/tmp/test.png")
        print(f"  delete: {deleted} chunks")
        print("  [OK] ImageStore")
    except Exception as e:
        print(f"  [FAIL] ImageStore: {type(e).__name__}: {e}")
        rc = 1

    if rc == 0:
        print("\n[PASS] Milvus store 全部通过")
    else:
        print("\n[FAIL] 部分 store 失败")
    return rc


if __name__ == "__main__":
    sys.exit(main())
