"""IO 辅助。"""
from __future__ import annotations
from pathlib import Path


def ensure_absolute(file_path: str) -> str:
    p = Path(file_path)
    if not p.is_absolute():
        raise ValueError(f"必须传入绝对路径：{file_path}")
    if not p.exists():
        raise FileNotFoundError(file_path)
    if not p.is_file():
        raise ValueError(f"不是文件：{file_path}")
    return str(p)
