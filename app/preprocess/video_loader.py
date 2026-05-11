"""视频预留位，未实现。"""
from __future__ import annotations
from app.core.types import Chunk
from app.core.exceptions import VideoReserved
from app.core.decorator import register
from app.preprocess.base import BasePreprocessor


@register((".mp4", ".mov", ".mkv", ".avi", ".webm"))
class VideoLoader(BasePreprocessor):
    """TODO: 抽帧策略（均匀 vs 关键帧）、最大帧数、fps；接入 fg-clip2 视频方案。"""
    SUPPORTED_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm")

    def load(self, file_path: str) -> list[Chunk]:
        raise VideoReserved("video loader reserved, not implemented")
