"""Offline integrity and behavior checks for the packaged AI Team Skill."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR", "LANG", "LC_ALL")
        if key in os.environ
    }
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
    return environment


def run_python(script: Path, *arguments: str, stdin: object | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(script), *arguments],
        input=None if stdin is None else json.dumps(stdin, ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        cwd=ROOT,
        env=clean_environment(),
        timeout=30,
        check=False,
    )


def check_manifest() -> str:
    manifest_path = ROOT / "package-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("name") != "ai-team" or not isinstance(manifest.get("version"), str):
        fail("package manifest name/version is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        fail("package manifest file map is missing")
    for relative, expected in files.items():
        target = (ROOT / relative).resolve()
        if not target.is_relative_to(ROOT) or target.is_symlink() or not target.is_file():
            fail(f"manifest target is missing or unsafe: {relative}")
        if sha256(target.read_bytes()).hexdigest() != expected:
            fail(f"manifest hash mismatch: {relative}")
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    match = re.search(
        r'^metadata:\s*\{[^}]*version:\s*["\']?([^,"\'}\s]+)',
        skill,
        flags=re.MULTILINE,
    )
    if not match or match.group(1) != manifest["version"]:
        fail("SKILL.md version does not match package manifest")
    return manifest["version"]


def check_python_sources() -> int:
    sources = sorted(SCRIPTS.rglob("*.py"))
    for source in sources:
        text = source.read_text(encoding="utf-8", errors="strict")
        compile(text, str(source), "exec")
    return len(sources)


def tool_request(capacity: int) -> dict[str, object]:
    return {
        "task": "脱敏月结流程练习",
        "tools": [{
            "id": "fixture-host",
            "models": [{
                "id": "fixture-model",
                "capabilities": ["general"],
                "evidence": "measured",
                "cost_rank": 0,
            }],
            "availability": "native",
            "independent_sessions": capacity > 1,
            "session_capacity": capacity,
            "max_parallel_agents": 1,
            "can_select_model": False,
            "default_model_id": "fixture-model",
            "current_model_ids": ["fixture-model"],
            "unavailable_model_ids": [],
        }],
    }


def check_planner_modes() -> dict[str, str]:
    expected = {3: "independent_agents", 2: "peer_review_only", 1: "self_check_only"}
    observed: dict[str, str] = {}
    for capacity, wanted in expected.items():
        result = run_python(SCRIPTS / "plan_team.py", stdin=tool_request(capacity))
        if result.returncode != 0:
            fail(f"plan_team failed for capacity {capacity}: {result.stderr.strip()}")
        payload = json.loads(result.stdout)
        if payload.get("mode") != wanted or payload.get("plan_only") is not True:
            fail(f"unexpected planner mode for capacity {capacity}: {payload.get('mode')}")
        observed[str(capacity)] = wanted
    return observed


def check_mock_output() -> int:
    result = run_python(SCRIPTS / "demo.py", "脱敏月结方案，检查成本与质量取舍")
    if result.returncode != 0:
        fail(f"Mock demo failed: {result.stderr.strip()}")
    if "Mock" not in result.stderr:
        fail("Mock demo did not disclose its Mock boundary")
    headings = (
        "## 分工", "## 结果", "## 审阅意见", "## 讨论结论", "## 未解决分歧", "## 最终成果",
    )
    if any(heading not in result.stdout for heading in headings):
        fail("Mock demo is missing a required output section")
    return len(headings)


def check_review_preparation() -> int:
    payload = {
        "id": "fixture-artifact", "assignment_id": "fixture-task", "author_id": "fixture-a",
        "version": 1, "content": "保留原文。\r\n第二行。  \n",
    }
    result = run_python(SCRIPTS / "prepare_review.py", stdin=payload)
    if result.returncode != 0:
        fail("host review preparation failed")
    prepared = json.loads(result.stdout)
    if prepared.get("ready_for_review") is not True or prepared.get("artifact") != payload:
        fail("host review preparation changed the source")
    if prepared.get("content_sha256") != sha256(payload["content"].encode("utf-8")).hexdigest():
        fail("host review preparation has an incorrect body hash")
    revised = run_python(SCRIPTS / "prepare_review.py", stdin={**payload, "version": 2})
    if revised.returncode != 0:
        fail("revised review preparation failed")
    second = json.loads(revised.stdout)
    if second.get("content_sha256") != prepared["content_sha256"] or second.get("artifact_sha256") == prepared["artifact_sha256"]:
        fail("review preparation confused the body hash and version hash")
    unavailable = run_python(SCRIPTS / "prepare_review.py", "--hash-unavailable")
    stopped = json.loads(unavailable.stdout)
    if unavailable.returncode != 2 or stopped.get("ready_for_review") is not False or "artifact_sha256" in stopped:
        fail("unavailable hashing did not stop review preparation")
    return 3


def main() -> int:
    try:
        version = check_manifest()
        source_count = check_python_sources()
        modes = check_planner_modes()
        section_count = check_mock_output()
        preparation_count = check_review_preparation()
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps({
        "status": "PASS",
        "version": version,
        "python_sources": source_count,
        "planner_modes": modes,
        "mock_sections": section_count,
        "review_preparation_checks": preparation_count,
        "real_agents_started": False,
        "acceptance_boundary": "local package check only",
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
