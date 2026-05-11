"""统一数据类型：Chunk / Hit / Modality / SourceType。"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class SourceType(str, Enum):
    TXT = "txt"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class Chunk:
    """预处理输出的统一中间结构。

    chunk_id 规则：sha256(absolute_path)[:16] + "_{idx:04d}"
    """
    chunk_id: str
    source_path: str
    source_type: SourceType
    modality: Modality
    file_hash: str

    # 定位字段
    page: Optional[int] = None
    sheet_name: Optional[str] = None
    slide_no: Optional[int] = None
    frame_idx: Optional[int] = None

    # 内容载荷（二选一）
    text: Optional[str] = None
    image: Optional[np.ndarray] = None  # RGB HWC uint8

    # 命中后展示
    preview: Optional[str] = None

    # 其它（char_start / char_end / row / image_idx_in_slide 等放这里）
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """查询结果统一结构。"""
    source_path: str
    modality: Modality
    score: float
    page: Optional[int] = None
    slide_no: Optional[int] = None
    sheet_name: Optional[str] = None
    frame_idx: Optional[int] = None
    preview: Optional[str] = None
    chunk_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.source_path,
            "modality": self.modality.value,
            "score": float(self.score),
            "page": self.page,
            "slide_no": self.slide_no,
            "sheet_name": self.sheet_name,
            "frame_idx": self.frame_idx,
            "preview": self.preview,
            "chunk_id": self.chunk_id,
            "extra": self.extra,
        }
