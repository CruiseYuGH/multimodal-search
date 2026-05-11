"""loguru 简单封装。"""
from __future__ import annotations
import sys
from loguru import logger
from config import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level:<7}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}")
