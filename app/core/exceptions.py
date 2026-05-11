"""项目自定义异常。"""


class MMSearchError(Exception):
    """基类。"""


class UnsupportedFileType(MMSearchError):
    """文件扩展名未注册。"""


class PreprocessError(MMSearchError):
    """预处理（加载/分块）失败。"""


class EmbeddingError(MMSearchError):
    """向量化失败。"""


class StoreError(MMSearchError):
    """Milvus 操作失败。"""


class VideoReserved(MMSearchError):
    """视频功能保留位，尚未实现。"""
