"""E 阶段冒烟：端到端 index + query (text/image/t2i/hybrid)。

依赖：Milvus 已启动 + collections 已建 + /tmp/mm_smoke 样例已生成（B 阶段）。
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipelines.index_pipeline import index_file
from app.pipelines.query_text import query_text
from app.pipelines.query_image import query_image
from app.pipelines.query_t2i import query_text_to_image
from app.pipelines.query_hybrid import query_hybrid


SMOKE_DIR = Path("/tmp/mm_smoke")


def main() -> int:
    if not SMOKE_DIR.exists():
        print(f"[FAIL] {SMOKE_DIR} 不存在，请先运行 scripts/smoke_preprocess.py")
        return 1
    
    samples = {
        "txt": SMOKE_DIR / "sample.txt",
        "docx": SMOKE_DIR / "sample.docx",
        "xlsx": SMOKE_DIR / "sample.xlsx",
        "image": SMOKE_DIR / "sample.png",
        "pptx": SMOKE_DIR / "sample.pptx",
    }
    
    rc = 0
    
    # ---- 索引 ----
    print("[1/5] index_file")
    for kind, path in samples.items():
        try:
            res = index_file(str(path), skip_if_exists=False)
            if res.get("error"):
                print(f"  [FAIL] {kind}: {res['error']}")
                rc = 1
            else:
                print(f"  [OK] {kind}: inserted={res['inserted']}, modality={res['modality_counts']}")
        except Exception as e:
            print(f"  [FAIL] {kind}: {type(e).__name__}: {e}")
            rc = 1
    print()
    
    # ---- query_text ----
    print("[2/5] query_text")
    try:
        hits = query_text("测试文档", topk=3)
        print(f"  hits={len(hits)}")
        for h in hits[:2]:
            print(f"    - {Path(h.source_path).name} page={h.page} score={h.score:.4f}")
        if not hits:
            print("  [WARN] 无结果")
        else:
            print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        rc = 1
    print()
    
    # ---- query_image ----
    print("[3/5] query_image")
    try:
        hits = query_image(str(samples["image"]), topk=3)
        print(f"  hits={len(hits)}")
        for h in hits[:2]:
            print(f"    - {Path(h.source_path).name} score={h.score:.4f}")
        if not hits:
            print("  [WARN] 无结果")
        else:
            print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        rc = 1
    print()
    
    # ---- query_t2i ----
    print("[4/5] query_text_to_image")
    try:
        hits = query_text_to_image("一个简约的卧室", topk=3)
        print(f"  hits={len(hits)}")
        for h in hits[:2]:
            print(f"    - {Path(h.source_path).name} score={h.score:.4f}")
        if not hits:
            print("  [WARN] 无结果")
        else:
            print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        rc = 1
    print()
    
    # ---- query_hybrid ----
    print("[5/5] query_hybrid (rrf)")
    try:
        hits = query_hybrid(q="测试", image_path=str(samples["image"]), topk=5, fusion="rrf")
        print(f"  hits={len(hits)}")
        for h in hits[:3]:
            print(f"    - {Path(h.source_path).name} mod={h.modality.value} score={h.score:.4f}")
        if not hits:
            print("  [WARN] 无结果")
        else:
            print("  [OK]")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        rc = 1
    print()
    
    if rc == 0:
        print("[PASS] 端到端 pipelines 全部通过")
    else:
        print("[FAIL] 部分 pipeline 失败")
    return rc


if __name__ == "__main__":
    sys.exit(main())
