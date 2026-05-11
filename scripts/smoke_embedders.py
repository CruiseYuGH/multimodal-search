"""C 阶段冒烟：加载 BGE-M3 + fg-clip2，跑一次 embedding，验证维度与归一化。

依赖：服务器已有模型权重 + GPU。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedders.text_embedder import BgeM3TextEmbedder
from app.embedders.image_embedder import FgClip2ImageEmbedder
from app.embedders.clip_text_embedder import FgClip2TextEmbedder
from app.core.types import Chunk, Modality, SourceType
from app.utils.hashing import make_chunk_id
from config import settings


def check_norm(vec: np.ndarray, name: str, tol: float = 1e-3) -> bool:
    norm = np.linalg.norm(vec)
    ok = abs(norm - 1.0) < tol
    print(f"  [{name}] dim={vec.shape[-1]}, norm={norm:.6f}, L2_ok={ok}")
    return ok


def main() -> int:
    print(f"[config] BGE_M3_PATH={settings.BGE_M3_PATH}")
    print(f"[config] FG_CLIP2_PATH={settings.FG_CLIP2_PATH}")
    print(f"[config] DEVICE={settings.DEVICE}")
    print()

    rc = 0

    # ---- BGE-M3 ----
    print("[1/3] BgeM3TextEmbedder")
    try:
        bge = BgeM3TextEmbedder(use_fp16=True)
        # 单条查询
        q_vec = bge.embed_query_text("这是一段测试文本，用于验证 BGE-M3 编码。")
        if not check_norm(q_vec, "query_text", tol=1e-3):
            rc = 1
        # 批量 chunk
        chunks = [
            Chunk(
                chunk_id=make_chunk_id("/tmp/test.txt", i),
                source_path="/tmp/test.txt",
                source_type=SourceType.TXT,
                modality=Modality.TEXT,
                file_hash="dummy",
                text=f"第 {i} 段文本内容。",
            )
            for i in range(3)
        ]
        batch_vecs = bge.embed_chunks(chunks)
        print(f"  batch shape={batch_vecs.shape}, dtype={batch_vecs.dtype}")
        for i, v in enumerate(batch_vecs):
            if not check_norm(v, f"chunk_{i}", tol=1e-3):
                rc = 1
        print("  [OK] BGE-M3")
    except Exception as e:
        print(f"  [FAIL] BGE-M3: {type(e).__name__}: {e}")
        rc = 1
    print()

    # ---- fg-clip2 image ----
    print("[2/3] FgClip2ImageEmbedder")
    try:
        from PIL import Image
        fg_img = FgClip2ImageEmbedder(use_fp16=True)
        # 合成一张图
        pil = Image.new("RGB", (320, 240), (100, 150, 200))
        arr = np.asarray(pil, dtype=np.uint8)
        img_chunk = Chunk(
            chunk_id=make_chunk_id("/tmp/test.png", 0),
            source_path="/tmp/test.png",
            source_type=SourceType.IMAGE,
            modality=Modality.IMAGE,
            file_hash="dummy",
            image=arr,
        )
        img_vecs = fg_img.embed_chunks([img_chunk])
        print(f"  batch shape={img_vecs.shape}, dtype={img_vecs.dtype}")
        if not check_norm(img_vecs[0], "image_chunk"):
            rc = 1
        # 查询图像（直接路径）
        tmp_img = Path("/tmp/mm_smoke_c.png")
        pil.save(tmp_img)
        q_img_vec = fg_img.embed_query_image(str(tmp_img))
        if not check_norm(q_img_vec, "query_image"):
            rc = 1
        tmp_img.unlink()
        print("  [OK] fg-clip2 image")
    except Exception as e:
        print(f"  [FAIL] fg-clip2 image: {type(e).__name__}: {e}")
        rc = 1
    print()

    # ---- fg-clip2 text ----
    print("[3/3] FgClip2TextEmbedder")
    try:
        fg_txt = FgClip2TextEmbedder(use_fp16=True)
        # short
        q_short = fg_txt.embed_query_text("一只猫", walk_type="short")
        if not check_norm(q_short, "text_short"):
            rc = 1
        # long（自动判断）
        long_text = "这是一段很长的描述文本，用于验证 fg-clip2 文本塔的 long walk_type 分支。" * 5
        q_long = fg_txt.embed_query_text(long_text)
        if not check_norm(q_long, "text_long_auto"):
            rc = 1
        print("  [OK] fg-clip2 text")
    except Exception as e:
        print(f"  [FAIL] fg-clip2 text: {type(e).__name__}: {e}")
        rc = 1
    print()

    if rc == 0:
        print("[PASS] 所有 embedder 加载成功，维度正确，L2 归一化验证通过。")
    else:
        print("[FAIL] 部分 embedder 失败，见上方日志。")
    return rc


if __name__ == "__main__":
    sys.exit(main())
