"""触发各 loader 注册到 registry。"""
from app.preprocess import (
    text_loader,   # noqa: F401
    docx_loader,   # noqa: F401
    xlsx_loader,   # noqa: F401
    pptx_loader,   # noqa: F401
    image_loader,  # noqa: F401
    video_loader,  # noqa: F401
)
