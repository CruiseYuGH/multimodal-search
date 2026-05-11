"""F 阶段最终冒烟：通过 CLI 完整流程测试。

依赖：Milvus 已启动 + collections 已建 + /tmp/mm_smoke 样例已生成。
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

SMOKE_DIR = Path("/tmp/mm_smoke")
CLI = ["python", "-m", "app.api.cli"]


def run_cli(args: list[str]) -> tuple[int, str]:
    """执行 CLI 命令，返回 (exit_code, output)。"""
    result = subprocess.run(
        CLI + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return result.returncode, result.stdout + result.stderr


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
    
    # ---- admin ping ----
    print("[1/8] admin ping")
    code, out = run_cli(["admin", "ping"])
    if code != 0:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        print("  [OK]")
    print()
    
    # ---- admin list-loaders ----
    print("[2/8] admin list-loaders")
    code, out = run_cli(["admin", "list-loaders"])
    if code != 0 or "LOADERS" not in out:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        print(f"  [OK] {out.count('→')} loaders")
    print()
    
    # ---- index ----
    print("[3/8] index (5 files)")
    for kind, path in samples.items():
        code, out = run_cli(["index", "--file", str(path), "--no-skip-exists"])
        if code != 0:
            print(f"  [FAIL] {kind}: exit={code}")
            print(out)
            rc = 1
        else:
            print(f"  [OK] {kind}")
    print()
    
    # ---- admin stats ----
    print("[4/8] admin stats")
    code, out = run_cli(["admin", "stats"])
    if code != 0 or "STATS" not in out:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        print("  [OK]")
        # 提取统计信息
        for line in out.split("\n"):
            if "mm_doc:" in line or "mm_image:" in line or "total:" in line:
                print(f"    {line.strip()}")
    print()
    
    # ---- search text ----
    print("[5/8] search text")
    code, out = run_cli(["search", "text", "--q", "测试文档", "--topk", "3"])
    if code != 0:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        hits = out.count("modality:")
        print(f"  [OK] {hits} hits")
    print()
    
    # ---- search image ----
    print("[6/8] search image")
    code, out = run_cli(["search", "image", "--image", str(samples["image"]), "--topk", "3"])
    if code != 0:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        hits = out.count("modality:")
        print(f"  [OK] {hits} hits")
    print()
    
    # ---- search t2i ----
    print("[7/8] search t2i")
    code, out = run_cli(["search", "t2i", "--q", "一个简约的卧室", "--topk", "3"])
    if code != 0:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        hits = out.count("modality:")
        print(f"  [OK] {hits} hits")
    print()
    
    # ---- search hybrid ----
    print("[8/8] search hybrid")
    code, out = run_cli([
        "search", "hybrid",
        "--q", "测试",
        "--image", str(samples["image"]),
        "--topk", "5",
        "--fusion", "rrf"
    ])
    if code != 0:
        print(f"  [FAIL] exit={code}")
        print(out)
        rc = 1
    else:
        hits = out.count("modality:")
        print(f"  [OK] {hits} hits")
    print()
    
    if rc == 0:
        print("[PASS] CLI 完整流程测试通过")
    else:
        print("[FAIL] 部分测试失败")
    return rc


if __name__ == "__main__":
    sys.exit(main())
