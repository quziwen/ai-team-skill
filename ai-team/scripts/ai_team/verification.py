"""Small, offline checks attached to an existing, version-bound review package.

These checks establish only arithmetic or literal quotation facts. They do not
authenticate a supplied source, infer business truth, or complete a review.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import re

from .ccrl import MAX_REVIEW_INPUT_BYTES, artifact_sha256
from .contracts import Artifact, ReviewPackage
from .project import AcceptanceCheck


class VerificationInputError(ValueError):
    """Static diagnostics only: never include source text in an error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationInputError(message)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\0" not in value


def _keys(value: object, expected: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == expected


def _inputs(payload: object) -> tuple[Artifact, ReviewPackage, list[dict]]:
    _require(_keys(payload, {"artifact", "review_package", "checks"}), "invalid input fields")
    try:
        size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise VerificationInputError("invalid UTF-8 JSON input") from None
    _require(size <= MAX_REVIEW_INPUT_BYTES, "input exceeds 120 KiB")
    raw_artifact, raw_package, checks = (
        payload["artifact"], payload["review_package"], payload["checks"]
    )
    _require(_keys(raw_artifact, {f.name for f in fields(Artifact)}), "invalid artifact fields")
    _require(all(_text(raw_artifact[k]) for k in ("id", "assignment_id", "author_id", "content")),
             "invalid artifact text")
    _require(type(raw_artifact["version"]) is int and raw_artifact["version"] > 0,
             "invalid artifact version")
    _require(_keys(raw_package, {f.name for f in fields(ReviewPackage)}), "invalid review package fields")
    criteria = raw_package["acceptance_criteria"]
    _require(isinstance(criteria, (list, tuple)) and 0 < len(criteria) <= 64
             and all(_text(c) for c in criteria), "invalid acceptance criteria")
    _require(len(set(criteria)) == len(criteria), "duplicate acceptance criteria")
    _require(all(_text(raw_package[k]) for k in (
        "id", "assignment_id", "artifact_id", "artifact_sha256", "author_id", "reviewer_id", "scope"
    )), "invalid review package text")
    _require(type(raw_package["artifact_version"]) is int, "invalid review package version")
    artifact = Artifact(**raw_artifact)
    package = ReviewPackage(**{**raw_package, "acceptance_criteria": tuple(criteria)})
    _require((package.assignment_id, package.artifact_id, package.artifact_version, package.author_id)
             == (artifact.assignment_id, artifact.id, artifact.version, artifact.author_id),
             "review package targets a different artifact")
    _require(package.artifact_sha256 == artifact_sha256(artifact), "artifact hash mismatch")
    _require(package.reviewer_id != package.author_id, "author cannot be own reviewer")
    _require(isinstance(checks, list) and len(checks) <= 64, "invalid checks")
    seen: set[str] = set()
    for check in checks:
        _require(isinstance(check, dict) and _text(check.get("criterion")), "invalid check criterion")
        criterion = check["criterion"]
        _require(criterion in criteria, "check is outside acceptance criteria")
        _require(criterion not in seen, "duplicate check criterion")
        seen.add(criterion)
    return artifact, package, checks


def _result(criterion: str, status: str, evidence: str, method: str,
            source: str, source_sha256: str | None, impact: str) -> dict:
    # Use the same passed/pending/failed record as the existing project checks;
    # extra details describe this check, not a second review or approval record.
    return {
        **asdict(AcceptanceCheck(criterion, status, evidence)),
        "method": method, "source": source, "source_sha256": source_sha256,
        "impact": impact,
    }


_NUMBER = r"-?[0-9]{1,32}(?:\.[0-9]{1,32})?"
_SUM = re.compile(rf"({_NUMBER}(?:\s*\+\s*{_NUMBER}){{1,63}})\s*=\s*({_NUMBER})")


def _check(artifact: Artifact, item: dict) -> dict:
    criterion, method = item["criterion"], item.get("method")
    if method == "unverified":
        _require(_keys(item, {"criterion", "method", "reason", "impact"})
                 and _text(item["reason"]) and _text(item["impact"]), "invalid unverified check")
        return _result(criterion, "pending", item["reason"], method, "未取得检查依据", None, item["impact"])
    if method == "sum":
        _require(_keys(item, {"criterion", "method", "expression"}) and _text(item["expression"]),
                 "invalid sum check")
        expression = item["expression"]
        # An entire line, not a cherry-picked numeric substring of another sum.
        _require(expression in (line.strip() for line in artifact.content.splitlines()),
                 "sum expression is not a complete artifact line")
        matched = _SUM.fullmatch(expression)
        _require(matched is not None, "unsupported sum expression")
        # Precision covers every permitted term, with no binary float rounding.
        with localcontext() as context:
            context.prec = 100
            computed = sum((Decimal(term.strip()) for term in matched[1].split("+")), Decimal(0))
            stated = Decimal(matched[2])
        passed = computed == stated
        return _result(
            criterion, "passed" if passed else "failed",
            f"本地重算为 {computed}，正文等号右侧为 {stated}。只核对加法，不证明输入数字、单位或业务口径正确。",
            method, f"artifact:{artifact.id}:v{artifact.version}", artifact_sha256(artifact),
            "" if passed else "正文计算结果不符；更正后需按新版本重新检查。",
        )
    if method == "source_quote":
        _require(_keys(item, {"criterion", "method", "quote", "source_ref", "source_text"})
                 and _text(item["quote"]), "invalid source quote check")
        _require(item["quote"] in artifact.content, "quote is absent from artifact")
        _require(isinstance(item["source_ref"], str) and isinstance(item["source_text"], str),
                 "invalid source text")
        if not _text(item["source_ref"]) or not _text(item["source_text"]):
            return _result(criterion, "pending", "缺少来源位置或原文，尚未核对。", method,
                           "未取得完整来源", None, "补齐来源前不能把这项引用标为已核对。")
        passed = item["quote"] in item["source_text"]
        return _result(
            criterion, "passed" if passed else "failed",
            "引用与提供的来源原文逐字一致；未验证来源真伪、完整性或它是否支持整段结论。" if passed
            else "在提供的来源原文中未找到相同引用。",
            method, item["source_ref"], sha256(item["source_text"].encode("utf-8")).hexdigest(),
            "" if passed else "引用尚无对应原句；更正引用或补充正确来源后再检查。",
        )
    raise VerificationInputError("unsupported check method")


def verify_delivery(payload: object) -> dict:
    """Return check facts, never a standalone approval of the final deliverable.

    The host supplies the already fixed ReviewPackage, its exact artifact and
    permitted source copies. No file, URL, command or model is opened here.
    Missing criteria are retained as pending instead of silently discarded.
    """
    artifact, package, checks = _inputs(payload)
    supplied = {item["criterion"]: item for item in checks}
    results = []
    for criterion in package.acceptance_criteria:
        if criterion in supplied:
            results.append(_check(artifact, supplied[criterion]))
        else:
            results.append(_result(criterion, "pending", "尚未提供这项要求的检查结果。",
                                   "unverified", "未取得检查依据", None, "该必要要求尚未确认，不能宣布全部完成。"))
    blocking = [item["criterion"] for item in results if item["status"] != "passed"]
    return {
        "status": "checks_passed" if not blocking else "needs_verification",
        "checks_passed": not blocking,
        "review_package_id": package.id,
        "artifact_id": artifact.id, "artifact_version": artifact.version,
        "artifact_sha256": package.artifact_sha256,
        "acceptance_status": results, "blocking_criteria": blocking,
        "proves_task_completion": False, "real_agents_started": False,
        "limitation": "只核对本次提供的版本、加法或原句；不能代替独立审阅、未决事项处理或业务验收。",
    }
