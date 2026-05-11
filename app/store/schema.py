"""三个 collection 的 Milvus schema 定义。"""
from __future__ import annotations
from pymilvus import CollectionSchema, FieldSchema, DataType
from config import settings


def _common_fields(vector_dim: int, extra: list[FieldSchema]) -> list[FieldSchema]:
    base = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    ]
    return base + extra


def doc_schema() -> CollectionSchema:
    extra = [
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="sheet_name", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="slide_no", dtype=DataType.INT64),
        FieldSchema(name="preview", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="extra_json", dtype=DataType.VARCHAR, max_length=2048),
    ]
    return CollectionSchema(
        fields=_common_fields(settings.BGE_M3_DIM, extra),
        description="mm_doc: BGE-M3 文档向量",
    )


def image_schema() -> CollectionSchema:
    extra = [
        FieldSchema(name="slide_no", dtype=DataType.INT64),   # -1 表示独立图片
        FieldSchema(name="extra_json", dtype=DataType.VARCHAR, max_length=1024),
    ]
    return CollectionSchema(
        fields=_common_fields(settings.FG_CLIP2_DIM, extra),
        description="mm_image: fg-clip2 图像向量",
    )


def video_schema() -> CollectionSchema:
    extra = [
        FieldSchema(name="frame_idx", dtype=DataType.INT64),
        FieldSchema(name="segment_start", dtype=DataType.FLOAT),
        FieldSchema(name="segment_end", dtype=DataType.FLOAT),
        FieldSchema(name="extra_json", dtype=DataType.VARCHAR, max_length=1024),
    ]
    return CollectionSchema(
        fields=_common_fields(settings.FG_CLIP2_DIM, extra),
        description="mm_video: fg-clip2 视频向量（占位）",
    )


HNSW_INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": settings.MILVUS_METRIC,
    "params": {
        "M": settings.HNSW_M,
        "efConstruction": settings.HNSW_EF_CONSTRUCTION,
    },
}

HNSW_SEARCH_PARAMS = {
    "metric_type": settings.MILVUS_METRIC,
    "params": {"ef": settings.HNSW_EF_SEARCH},
}
