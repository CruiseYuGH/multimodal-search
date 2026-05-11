"""一键创建 Milvus collections + 索引。

用法：
  python scripts/init_milvus.py              # 建 mm_doc + mm_image
  python scripts/init_milvus.py --with-video # 额外建 mm_video（占位）
  python scripts/init_milvus.py --drop-all   # 删除所有 collection
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.store.milvus_client import MilvusClient
from app.store.schema import doc_schema, image_schema, video_schema, HNSW_INDEX_PARAMS
from config import settings


def main(with_video: bool, drop_all: bool) -> int:
    client = MilvusClient()
    print(f"[milvus] 连接 {client.host}:{client.port}")
    client.connect()
    if not client.ping():
        print("[FAIL] Milvus 不可达，请先启动：bash scripts/start_milvus.sh")
        return 1
    print("[OK] Milvus 连接成功")

    if drop_all:
        for name in [settings.COLLECTION_DOC, settings.COLLECTION_IMAGE, settings.COLLECTION_VIDEO]:
            try:
                client.drop(name)
                print(f"[drop] {name}")
            except Exception as e:
                print(f"[skip] {name}: {e}")
        return 0

    # 建 doc + image
    for name, schema in [
        (settings.COLLECTION_DOC, doc_schema()),
        (settings.COLLECTION_IMAGE, image_schema()),
    ]:
        try:
            client.ensure_collection(name, schema, HNSW_INDEX_PARAMS)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            return 1

    if with_video:
        try:
            client.ensure_collection(settings.COLLECTION_VIDEO, video_schema(), HNSW_INDEX_PARAMS)
            print(f"[OK] {settings.COLLECTION_VIDEO} (占位)")
        except Exception as e:
            print(f"[FAIL] {settings.COLLECTION_VIDEO}: {e}")
            return 1

    print("[PASS] collections 初始化完成")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-video", action="store_true")
    p.add_argument("--drop-all", action="store_true", help="删除所有 collection")
    args = p.parse_args()
    sys.exit(main(args.with_video, args.drop_all))
