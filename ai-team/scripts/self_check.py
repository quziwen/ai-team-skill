"""Offline package consistency checks; not a signature or a trusted runtime."""

# sys is built-in; no package-resolved import may precede the path adjustment.
import sys

# Do not import an unlisted scripts/json.py (or another stdlib lookalike)
# before the package has been checked. Python 3.11+ omits this path with -P/-I.
if __name__ == "__main__" and not sys.flags.safe_path:
    del sys.path[0]

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess


# Keep lexical paths: resolve() would conceal a linked package or entry.
ROOT = Path(__file__).absolute().parents[1]
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


def _checked_stat(path: Path) -> os.stat_result:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode) or (
        getattr(status, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        fail(f"package contains a link or reparse point: {path}")
    return status


def _package_files() -> set[str]:
    for ancestor in (*reversed(ROOT.parents), ROOT):
        if not stat.S_ISDIR(_checked_stat(ancestor).st_mode):
            fail("package root or its parent is not a directory")
    found: set[str] = set()
    pending = [ROOT]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            status = _checked_stat(entry)
            if stat.S_ISDIR(status.st_mode):
                pending.append(entry)
            elif stat.S_ISREG(status.st_mode):
                found.add(entry.relative_to(ROOT).as_posix())
            else:
                fail(f"package contains a non-regular file: {entry}")
    return found


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            fail(f"package manifest has a duplicate JSON key: {key}")
        result[key] = value
    return result


def _check_manifest_path(relative: str) -> None:
    # Accept only unambiguous slash-separated relative names on every host.
    parts = relative.split("/")
    if (
        not relative
        or any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in parts)
        or re.search(r'[\\:<>"|?*\x00-\x1f]', relative)
    ):
        fail(f"package manifest has an unsafe path: {relative!r}")


def check_manifest() -> str:
    # Inspect all entries before opening the manifest, including links, hidden
    # files and __pycache__. No unlisted Python cache is exempt from this check.
    actual = _package_files()
    manifest_name = "package-manifest.json"
    if manifest_name not in actual:
        fail("package manifest is missing")
    manifest = json.loads(
        (ROOT / manifest_name).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("name") != "ai-team"
        or not isinstance(manifest.get("version"), str)
        or not manifest["version"]
    ):
        fail("package manifest name/version is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        fail("package manifest file map is missing")
    names: set[str] = {manifest_name.casefold()}
    for relative, expected in files.items():
        _check_manifest_path(relative)
        if relative.casefold() in names:
            fail(f"package manifest has a duplicate or aliased path: {relative}")
        names.add(relative.casefold())
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            fail(f"package manifest hash is invalid: {relative}")
    wanted = set(files) | {manifest_name}
    missing, extra = wanted - actual, actual - wanted
    if missing:
        fail(f"package files are missing: {', '.join(sorted(missing))}")
    if extra:
        fail(f"unexpected package files (Python caches are not exempt): {', '.join(sorted(extra))}")
    for relative, expected in files.items():
        target = ROOT / relative
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
    installation = (ROOT / "INSTALL.md").read_text(encoding="utf-8")
    heading = installation.splitlines()[0] if installation else ""
    match = re.fullmatch(r"# AI Team ([0-9]+\.[0-9]+\.[0-9]+)：安装和试用", heading)
    if not match or match.group(1) != manifest["version"]:
        fail("INSTALL.md version does not match package manifest")
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


def check_task_routing() -> int:
    request = tool_request(3)
    request["task_profile"] = {
        "basis": "Public deterministic fixture.", "clarity": "clear", "complexity": "routine",
        "consequence": "low", "verification": "objective", "data_scope": "public",
    }
    tool = request["tools"][0]
    tool["can_select_model"] = True
    tool["models"].append({
        "id": "fixture-strong", "capabilities": ["general", "high_impact_review"],
        "evidence": "measured", "cost_rank": 3,
    })
    tool["current_model_ids"].append("fixture-strong")
    for expected, changes in (
        ("fixture-model", {}),
        ("fixture-strong", {"consequence": "high"}),
        (None, {"data_scope": "business", "allowed_tool_ids": []}),
    ):
        request["task_profile"].update(changes)
        result = run_python(SCRIPTS / "plan_team.py", stdin=request)
        if result.returncode != 0:
            fail("task routing fixture failed to run")
        payload = json.loads(result.stdout)
        if payload.get("plan_only") is not True:
            fail("task routing did not remain plan-only")
        if expected is None:
            if payload.get("mode") != "unavailable" or payload.get("roles") != []:
                fail("task routing ignored the allowed-tool restriction")
        elif {role["model_id"] for role in payload.get("roles", [])} != {expected}:
            fail("task routing ignored cost or review requirements")
    return 3


def check_review_preparation() -> int:
    payload = {
        "id": "fixture-artifact", "assignment_id": "fixture-task", "author_id": "fixture-a",
        "version": 1, "content": "保留原文。\r\n第二行。  \n",
    }
    result = run_python(SCRIPTS / "prepare_review.py", stdin=payload)
    if result.returncode != 0:
        fail("host review preparation failed")
    prepared = json.loads(result.stdout)
    metadata = {key: value for key, value in payload.items() if key != "content"}
    if prepared.get("ready_for_review") is not True or prepared.get("artifact") != metadata:
        fail("host review preparation changed artifact metadata")
    if "content" in prepared or "保留原文" in json.dumps(prepared, ensure_ascii=False):
        fail("host review preparation echoed the source without permission")
    if prepared.get("content_sha256") != sha256(payload["content"].encode("utf-8")).hexdigest():
        fail("host review preparation has an incorrect body hash")
    expected_artifact = "\0".join(str(payload[key]) for key in (
        "id", "assignment_id", "author_id", "version", "content",
    ))
    if prepared.get("artifact_sha256") != sha256(expected_artifact.encode("utf-8")).hexdigest():
        fail("host review preparation has an incorrect artifact hash")
    echoed = run_python(SCRIPTS / "prepare_review.py", "--include-content", stdin=payload)
    included = json.loads(echoed.stdout)
    if (
        echoed.returncode != 0
        or included.get("artifact") != payload
        or included.get("content_sha256") != prepared["content_sha256"]
        or included.get("artifact_sha256") != prepared["artifact_sha256"]
    ):
        fail("explicit review source echo changed the source or hashes")
    revised = run_python(SCRIPTS / "prepare_review.py", stdin={**payload, "version": 2})
    if revised.returncode != 0:
        fail("revised review preparation failed")
    second = json.loads(revised.stdout)
    if second.get("artifact") != {**metadata, "version": 2} or "保留原文" in json.dumps(second, ensure_ascii=False):
        fail("revised review preparation changed metadata or echoed the source")
    if second.get("content_sha256") != prepared["content_sha256"] or second.get("artifact_sha256") == prepared["artifact_sha256"]:
        fail("review preparation confused the body hash and version hash")
    unavailable = run_python(SCRIPTS / "prepare_review.py", "--hash-unavailable")
    stopped = json.loads(unavailable.stdout)
    if unavailable.returncode != 2 or stopped.get("ready_for_review") is not False or "artifact_sha256" in stopped:
        fail("unavailable hashing did not stop review preparation")
    return 3


def check_delivery_verification() -> int:
    artifact = {
        "id": "fixture-result", "assignment_id": "fixture-task", "author_id": "fixture-a",
        "version": 1, "content": "1 + 2 = 3",
    }
    digest = sha256("\0".join(str(artifact[k]) for k in (
        "id", "assignment_id", "author_id", "version", "content"
    )).encode("utf-8")).hexdigest()
    package = {
        "id": "fixture-rp", "assignment_id": artifact["assignment_id"],
        "artifact_id": artifact["id"], "artifact_version": 1,
        "artifact_sha256": digest, "author_id": artifact["author_id"],
        "reviewer_id": "fixture-b", "scope": "local fixture",
        "acceptance_criteria": ["arithmetic"],
    }
    payload = {"artifact": artifact, "review_package": package,
               "checks": [{"criterion": "arithmetic", "method": "sum", "expression": "1 + 2 = 3"}]}
    checked = run_python(SCRIPTS / "check_delivery.py", stdin=payload)
    receipt = json.loads(checked.stdout)
    if checked.returncode != 0 or receipt.get("checks_passed") is not True or receipt.get("proves_task_completion") is not False:
        fail("delivery verification did not preserve its local-check boundary")
    package["acceptance_criteria"].append("missing requirement")
    stopped = run_python(SCRIPTS / "check_delivery.py", stdin=payload)
    receipt = json.loads(stopped.stdout)
    if stopped.returncode != 2 or receipt.get("blocking_criteria") != ["missing requirement"]:
        fail("delivery verification treated a missing requirement as passed")
    return 2


def check_delivery_workflow() -> int:
    # The package has already passed manifest and source checks at this point.
    sys.path.insert(0, str(SCRIPTS))
    from ai_team.agents import MockBackend
    from ai_team.contracts import ReviewFeedback
    from ai_team.workflow import AITeam

    class FixtureBackend(MockBackend):
        def __init__(self, body: str) -> None:
            super().__init__("normal")
            self.body = body

        def produce(self, agent, assignment, task):
            return self.body if assignment.id == "work-1" else "Local fixture."

        def review(self, reviewer, artifact, assignment, task):
            return ReviewFeedback("no_findings", "Local scripted agreement.")

    def checker(artifact, package):
        return [{"criterion": "arithmetic", "method": "sum", "expression": artifact.content}]

    for body, criteria, expected, merge_calls in (
        ("1 + 2 = 3", ("arithmetic",), "completed", 1),
        ("1 + 2 = 4", ("arithmetic",), "needs_verification", 0),
        ("1 + 2 = 3", ("arithmetic", "source verified"), "needs_verification", 0),
    ):
        backend = FixtureBackend(body)
        result = AITeam(backend).run(
            "Local arithmetic fixture only",
            delivery_requirements={"work-1": criteria, "work-2": ()},
            delivery_checker=checker,
        )
        if result.status != expected or backend.integrate_calls != merge_calls:
            fail("delivery workflow ignored necessary checks")
    return 3


def main() -> int:
    try:
        version = check_manifest()
        source_count = check_python_sources()
        modes = check_planner_modes()
        section_count = check_mock_output()
        preparation_count = check_review_preparation()
        delivery_count = check_delivery_verification()
        routing_count = check_task_routing()
        workflow_count = check_delivery_workflow()
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
        "delivery_verification_checks": delivery_count,
        "task_routing_checks": routing_count,
        "delivery_workflow_checks": workflow_count,
        "real_agents_started": False,
        "acceptance_boundary": "local package check only",
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
