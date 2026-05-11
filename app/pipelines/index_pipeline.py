"""索引 Pipeline：file_path → loader → embedder → store。"""
from __future__ import annotations
from typing import Any


def index_file(file_path: str) -> dict[str, Any]:
    """构建并执行索引流程。

    Returns:
        {
          "file_path": str,
          "file_hash": str,
          "modality_counts": {"text": N, "image": M, "video": 0},
          "inserted": int,
          "skipped": bool,           # 已存在则跳过
          "error": Optional[str],
        }
    """
    raise NotImplementedError
