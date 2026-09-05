"""Run the bundled local Mock example, with an explicit status notice."""

from ai_team.cli import main


if __name__ == "__main__":
    import sys

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    print("这是本地 Mock 流程示例，输出不是实际任务成果。", file=sys.stderr)
    raise SystemExit(main())
