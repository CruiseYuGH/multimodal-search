"""一键创建 Milvus collections + 索引。

用法：
  python scripts/init_milvus.py              # 建 mm_doc + mm_image
  python scripts/init_milvus.py --with-video # 额外建 mm_video（占位）
"""
from __future__ import annotations
import argparse


def main(with_video: bool) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-video", action="store_true")
    args = p.parse_args()
    main(args.with_video)
