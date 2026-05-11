"""文件指纹 / chunk_id 生成。"""
from __future__ import annotations
import hashlib
from pathlib import Path


def sha256_file(file_path: str, buf_size: int = 1 << 20) -> str:
    """整文件 sha256。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def path_sha256_short(absolute_path: str, n: int = 16) -> str:
    """sha256(absolute_path) 截前 n 字符。"""
    return hashlib.sha256(absolute_path.encode("utf-8")).hexdigest()[:n]


def make_chunk_id(absolute_path: str, idx: int) -> str:
    """chunk_id 规则：sha256(path)[:16] + "_{idx:04d}"。"""
    return f"{path_sha256_short(absolute_path)}_{idx:04d}"
