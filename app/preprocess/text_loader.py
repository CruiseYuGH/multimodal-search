"""txt 文件 → 文本 Chunk 列表（走 RecursiveDocumentSplitter）。"""
from __future__ import annotations
from pathlib import Path

import chardet

from app.core.types import Chunk, Modality, SourceType
from app.core.decorator import register
from app.core.exceptions import PreprocessError
from app.preprocess.base import BasePreprocessor
from app.utils.hashing import sha256_file, make_chunk_id
from app.utils.io import ensure_absolute
from config import settings


def _detect_encoding(file_path: str) -> str:
    """读前 64KB 探测编码，失败回退 utf-8。"""
    with open(file_path, "rb") as f:
        raw = f.read(64 * 1024)
    if not raw:
        return "utf-8"
    guess = chardet.detect(raw)
    return guess.get("encoding") or "utf-8"


def _recursive_split(
    text: str,
    separators: list[str],
    split_length: int,
    split_overlap: int,
) -> list[tuple[int, int, str]]:
    """递归按分隔符切块，每块字符长度 <= split_length。

    Returns: [(char_start, char_end, chunk_text), ...] —— 字符偏移基于原始 text。
    """
    if len(text) <= split_length:
        return [(0, len(text), text)] if text.strip() else []

    sep = separators[0] if separators else ""
    rest = separators[1:] if len(separators) > 1 else [""]

    if sep == "":
        # 字符级强切（带 overlap）
        out: list[tuple[int, int, str]] = []
        i = 0
        n = len(text)
        step = max(1, split_length - split_overlap)
        while i < n:
            j = min(n, i + split_length)
            seg = text[i:j]
            if seg.strip():
                out.append((i, j, seg))
            if j == n:
                break
            i += step
        return out

    # 按 sep 切，保留 sep 在前段尾部
    pieces: list[tuple[int, int, str]] = []
    cursor = 0
    parts = text.split(sep)
    for idx, p in enumerate(parts):
        seg = p + (sep if idx < len(parts) - 1 else "")
        if not seg:
            continue
        pieces.append((cursor, cursor + len(seg), seg))
        cursor += len(seg)

    # 合并相邻小片到 <= split_length；过大的递归用下一层 sep 切
    merged: list[tuple[int, int, str]] = []
    buf_start: int | None = None
    buf_end = 0
    buf_text = ""
    for s, e, seg in pieces:
        if len(seg) > split_length:
            # flush 当前 buf
            if buf_text:
                merged.append((buf_start, buf_end, buf_text))
                buf_start, buf_end, buf_text = None, 0, ""
            # 递归
            sub = _recursive_split(seg, rest, split_length, split_overlap)
            for ss, ee, st in sub:
                merged.append((s + ss, s + ee, st))
            continue
        if buf_text and len(buf_text) + len(seg) > split_length:
            merged.append((buf_start, buf_end, buf_text))
            buf_start, buf_end, buf_text = s, e, seg
        else:
            if buf_start is None:
                buf_start = s
            buf_end = e
            buf_text += seg
    if buf_text:
        merged.append((buf_start, buf_end, buf_text))

    # 加 overlap：每块向前借 split_overlap 字符（不跨越文档起点）
    if split_overlap > 0 and len(merged) > 1:
        with_ov: list[tuple[int, int, str]] = []
        for i, (s, e, st) in enumerate(merged):
            if i == 0:
                with_ov.append((s, e, st))
            else:
                ns = max(0, s - split_overlap)
                with_ov.append((ns, e, text[ns:e]))
        merged = with_ov

    # 去空
    return [(s, e, st) for s, e, st in merged if st.strip()]


def split_text(text: str) -> list[tuple[int, int, str]]:
    """对外暴露：按 settings.SPLITTER 切。"""
    cfg = settings.SPLITTER
    return _recursive_split(text, cfg.separators, cfg.split_length, cfg.split_overlap)


@register((".txt",))
class TxtLoader(BasePreprocessor):
    SUPPORTED_EXT = (".txt",)

    def load(self, file_path: str) -> list[Chunk]:
        path = ensure_absolute(file_path)
        try:
            enc = _detect_encoding(path)
            with open(path, "r", encoding=enc, errors="replace") as f:
                text = f.read()
        except Exception as e:
            raise PreprocessError(f"读取 txt 失败: {path}: {e}") from e

        file_hash = sha256_file(path)
        pieces = split_text(text)
        chunks: list[Chunk] = []
        for idx, (cs, ce, seg) in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=make_chunk_id(path, idx),
                source_path=path,
                source_type=SourceType.TXT,
                modality=Modality.TEXT,
                file_hash=file_hash,
                text=seg,
                preview=seg[:200],
                extra={"char_start": cs, "char_end": ce, "encoding": enc},
            ))
        return chunks
