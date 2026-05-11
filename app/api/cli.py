"""mmsearch CLI 入口（click）。

子命令：
  mmsearch index --file /abs/path
  mmsearch search text   --q "..." [--topk 10]
  mmsearch search image  --image /abs/path [--topk 10] [--include-video]
  mmsearch search t2i    --q "..." [--topk 10] [--include-video]
  mmsearch search hybrid [--q ...] [--image ...] [--topk 10] [--fusion rrf|weighted]
                         [--weights doc=1,image=1,t2i=1] [--include-video]
  mmsearch admin init-milvus [--with-video]
  mmsearch admin drop --collection mm_doc|mm_image|mm_video
"""
from __future__ import annotations
import json
import click


@click.group()
def cli() -> None:
    """multimodal-search CLI."""


@cli.command("index")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True, dir_okay=False))
def cmd_index(file_path: str) -> None:
    raise NotImplementedError


@cli.group("search")
def cmd_search() -> None:
    pass


@cmd_search.command("text")
@click.option("--q", required=True)
@click.option("--topk", default=10, type=int)
def cmd_search_text(q: str, topk: int) -> None:
    raise NotImplementedError


@cmd_search.command("image")
@click.option("--image", "image_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--topk", default=10, type=int)
@click.option("--include-video", is_flag=True, help="reserved, not implemented")
def cmd_search_image(image_path: str, topk: int, include_video: bool) -> None:
    raise NotImplementedError


@cmd_search.command("t2i")
@click.option("--q", required=True)
@click.option("--topk", default=10, type=int)
@click.option("--include-video", is_flag=True, help="reserved, not implemented")
@click.option("--walk-type", type=click.Choice(["short", "long"]), default=None)
def cmd_search_t2i(q: str, topk: int, include_video: bool, walk_type: str | None) -> None:
    raise NotImplementedError


@cmd_search.command("hybrid")
@click.option("--q", default=None)
@click.option("--image", "image_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--topk", default=10, type=int)
@click.option("--fusion", type=click.Choice(["rrf", "weighted"]), default="rrf")
@click.option("--weights", default=None, help='例如 "doc=1,image=1,t2i=1"')
@click.option("--include-video", is_flag=True, help="reserved, not implemented")
def cmd_search_hybrid(q: str | None, image_path: str | None, topk: int,
                      fusion: str, weights: str | None, include_video: bool) -> None:
    raise NotImplementedError


@cli.group("admin")
def cmd_admin() -> None:
    pass


@cmd_admin.command("init-milvus")
@click.option("--with-video", is_flag=True, help="同时建 mm_video collection")
def cmd_init_milvus(with_video: bool) -> None:
    raise NotImplementedError


@cmd_admin.command("drop")
@click.option("--collection", required=True,
              type=click.Choice(["mm_doc", "mm_image", "mm_video"]))
def cmd_drop(collection: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    cli()
