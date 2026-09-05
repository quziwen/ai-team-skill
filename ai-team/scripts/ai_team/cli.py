"""One-input command-line interface."""

from __future__ import annotations

import sys

from .presenter import render_markdown
from .workflow import run


def _configure_utf8_output() -> None:
    """Keep Chinese output stable when Windows pipes are decoded as UTF-8."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_output()
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        task = " ".join(args).strip()
    else:
        try:
            task = input("请输入任务：").strip()
        except EOFError:
            task = ""

    if not task:
        print("错误：任务不能为空。", file=sys.stderr)
        return 2

    result = run(task)
    print(render_markdown(result))
    return 0
