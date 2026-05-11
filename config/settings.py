"""全局配置。环境变量优先，未设置则使用默认值。"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------- 路径 ----------
MODELS_DIR: Path = Path(_env(
    "MODELS_DIR",
    "/home/admin/multimodel_search/multimodal-local-search_v2/models",
))
PROJECT_ROOT: Path = Path(_env(
    "PROJECT_ROOT",
    "/Users/zouzeyu/Documents/Code/0512",
))

BGE_M3_PATH: Path = MODELS_DIR / "bge-m3"
FG_CLIP2_PATH: Path = MODELS_DIR / "fg-clip2"

# ---------- Milvus ----------
MILVUS_HOST: str = _env("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT: int = int(_env("MILVUS_PORT", "19530"))

COLLECTION_DOC: str = "mm_doc"
COLLECTION_IMAGE: str = "mm_image"
COLLECTION_VIDEO: str = "mm_video"  # 占位

# ---------- 向量空间 ----------
BGE_M3_DIM: int = 1024
FG_CLIP2_DIM: int = 768

# ---------- HNSW ----------
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 200
HNSW_EF_SEARCH: int = 64
MILVUS_METRIC: str = "IP"

# ---------- 设备 ----------
DEVICE: str = _env("DEVICE", "cuda")

# ---------- 文档分块 (RecursiveDocumentSplitter) ----------
@dataclass
class SplitterConfig:
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", "；", " ", "",
    ])
    split_length: int = 800     # 字符
    split_overlap: int = 100
    split_unit: str = "char"


SPLITTER: SplitterConfig = SplitterConfig()

# ---------- fg-clip2 文本塔 ----------
FGCLIP2_TEXT_SHORT_MAX_LEN: int = 64
FGCLIP2_TEXT_LONG_MAX_LEN: int = 196
FGCLIP2_LONG_TEXT_THRESHOLD: int = 60  # 字符数超过则自动 walk_type=long

# ---------- PPTX 内嵌图过滤 ----------
PPTX_IMAGE_MIN_SIZE: int = 64  # 宽或高小于则丢弃

# ---------- 查询 ----------
DEFAULT_TOPK: int = 10
RRF_K: int = 60

# ---------- 日志 ----------
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
