"""CLI 入口：index / search / admin 子命令。"""
from __future__ import annotations
import sys
import click
from pathlib import Path

from app.pipelines.index_pipeline import index_file
from app.pipelines.query_text import query_text
from app.pipelines.query_image import query_image
from app.pipelines.query_t2i import query_text_to_image
from app.pipelines.query_hybrid import query_hybrid
from app.store.milvus_client import MilvusClient
from app.core.registry import list_registered
from config import settings


@click.group()
def cli():
    """多模态搜索系统 CLI。"""
    pass


# ========== index 子命令 ==========
@cli.command()
@click.option("--file", "-f", required=True, help="文件绝对路径")
@click.option("--skip-exists/--no-skip-exists", default=True, help="跳过已索引文件（默认 True）")
def index(file: str, skip_exists: bool):
    """索引单个文件到向量库。"""
    try:
        res = index_file(file, skip_if_exists=skip_exists)
        if res.get("error"):
            click.echo(f"[ERROR] {res['error']}", err=True)
            sys.exit(1)
        if res.get("skipped"):
            click.echo(f"[SKIP] {Path(file).name} (已存在)")
        else:
            click.echo(f"[OK] {Path(file).name}")
            click.echo(f"  file_hash: {res['file_hash'][:16]}...")
            click.echo(f"  modality: {res['modality_counts']}")
            click.echo(f"  inserted: {res['inserted']}")
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


# ========== search 子命令组 ==========
@cli.group()
def search():
    """检索命令组。"""
    pass


@search.command(name="text")
@click.option("--q", "-q", required=True, help="查询文本")
@click.option("--topk", "-k", default=settings.DEFAULT_TOPK, help="返回 top-K 结果")
def search_text(q: str, topk: int):
    """文本检索（BGE-M3 → mm_doc）。"""
    try:
        hits = query_text(q, topk=topk)
        _print_hits(hits)
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


@search.command(name="image")
@click.option("--image", "-i", required=True, help="查询图片路径")
@click.option("--topk", "-k", default=settings.DEFAULT_TOPK, help="返回 top-K 结果")
@click.option("--include-video/--no-include-video", default=False, help="包含视频库（预留）")
def search_image(image: str, topk: int, include_video: bool):
    """图像检索（fg-clip2 image → mm_image）。"""
    try:
        hits = query_image(image, topk=topk, include_video=include_video)
        _print_hits(hits)
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


@search.command(name="t2i")
@click.option("--q", "-q", required=True, help="查询文本")
@click.option("--topk", "-k", default=settings.DEFAULT_TOPK, help="返回 top-K 结果")
@click.option("--include-video/--no-include-video", default=False, help="包含视频库（预留）")
@click.option("--walk-type", type=click.Choice(["short", "long"]), default=None, help="fg-clip2 文本塔模式")
def search_t2i(q: str, topk: int, include_video: bool, walk_type: str | None):
    """文本→图像检索（fg-clip2 text → mm_image）。"""
    try:
        hits = query_text_to_image(q, topk=topk, include_video=include_video, walk_type=walk_type)
        _print_hits(hits)
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


@search.command(name="hybrid")
@click.option("--q", "-q", default=None, help="查询文本")
@click.option("--image", "-i", default=None, help="查询图片路径")
@click.option("--topk", "-k", default=settings.DEFAULT_TOPK, help="返回 top-K 结果")
@click.option("--fusion", type=click.Choice(["rrf", "weighted"]), default="rrf", help="融合方法")
@click.option("--weight-doc", type=float, default=1.0, help="文档权重（weighted 模式）")
@click.option("--weight-image", type=float, default=1.0, help="图像权重（weighted 模式）")
@click.option("--weight-t2i", type=float, default=1.0, help="text-to-image 权重（weighted 模式）")
@click.option("--include-video/--no-include-video", default=False, help="包含视频库（预留）")
def search_hybrid(q: str | None, image: str | None, topk: int, fusion: str,
                  weight_doc: float, weight_image: float, weight_t2i: float,
                  include_video: bool):
    """混合检索（多路并行 + RRF/加权融合）。"""
    if not q and not image:
        click.echo("[ERROR] 至少提供 --q 或 --image 之一", err=True)
        sys.exit(1)
    try:
        weights = {"doc": weight_doc, "image": weight_image, "t2i": weight_t2i}
        hits = query_hybrid(q=q, image_path=image, topk=topk, weights=weights,
                           fusion=fusion, include_video=include_video)
        _print_hits(hits)
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


def _print_hits(hits: list):
    """统一打印检索结果。"""
    if not hits:
        click.echo("[EMPTY] 无结果")
        return
    click.echo(f"[HITS] {len(hits)} 条结果:\\n")
    for i, h in enumerate(hits, start=1):
        name = Path(h.source_path).name
        click.echo(f"{i}. {name}")
        click.echo(f"   modality: {h.modality.value}")
        click.echo(f"   score: {h.score:.4f}")
        if h.page is not None:
            click.echo(f"   page: {h.page}")
        if h.slide_no is not None:
            click.echo(f"   slide: {h.slide_no}")
        if h.sheet_name:
            click.echo(f"   sheet: {h.sheet_name}")
        if h.frame_idx is not None:
            click.echo(f"   frame: {h.frame_idx}")
        if h.preview:
            preview = h.preview[:80] + "..." if len(h.preview) > 80 else h.preview
            click.echo(f"   preview: {preview}")
        click.echo()


# ========== admin 子命令组 ==========
@cli.group()
def admin():
    """管理命令组。"""
    pass


@admin.command(name="ping")
def admin_ping():
    """检查 Milvus 连接。"""
    try:
        client = MilvusClient()
        client.connect()
        click.echo(f"[OK] Milvus 连接成功: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


@admin.command(name="list-loaders")
def admin_list_loaders():
    """列出已注册的文件类型。"""
    registry = list_registered()
    click.echo(f"[LOADERS] {len(registry)} 种扩展名已注册:\\n")
    for ext, cls_name in sorted(registry.items()):
        click.echo(f"  {ext:8s} → {cls_name}")


@admin.command(name="stats")
def admin_stats():
    """显示向量库统计信息。"""
    try:
        from app.store.doc_store import DocStore
        from app.store.image_store import ImageStore
        client = MilvusClient()
        client.connect()
        doc_store = DocStore(client)
        img_store = ImageStore(client)
        
        doc_count = doc_store.count()
        img_count = img_store.count()
        
        click.echo("[STATS]")
        click.echo(f"  mm_doc:   {doc_count:,} 条")
        click.echo(f"  mm_image: {img_count:,} 条")
        click.echo(f"  total:    {doc_count + img_count:,} 条")
    except Exception as e:
        click.echo(f"[FAIL] {type(e).__name__}: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
