"""Prepare exact review text locally; never start an Agent or write a file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import sys

from ai_team.ccrl import MAX_REVIEW_INPUT_BYTES, artifact_sha256
from ai_team.contracts import Artifact


def prepare_review_input(payload: object, *, hashing_available: bool = True) -> dict:
    """The host supplies one exact artifact, not a hash invented by a reviewer.

    hashing_available describes this task's permitted local capability. This
    helper does not discover tools, grant permission, or validate Agent work.
    """
    if type(hashing_available) is not bool:
        raise ValueError("hashing_available must be a boolean")
    if not hashing_available:
        return {
            "status": "needs_version_check",
            "ready_for_review": False,
            "reason": "本次没有获准使用的摘要计算能力，版本核对未完成；不能沿用旧摘要或声称审阅完成。",
            "real_agents_started": False,
        }
    required = {"id", "assignment_id", "author_id", "version", "content"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("provide exactly id, assignment_id, author_id, version and content")
    for field in ("id", "assignment_id", "author_id", "content"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip() or "\0" in value:
            raise ValueError(f"{field} must be nonempty text without NUL")
    if type(payload["version"]) is not int or payload["version"] < 1:
        raise ValueError("version must be a positive integer")
    artifact = Artifact(**payload)
    body = artifact.content.encode("utf-8", errors="strict")
    if len(body) > MAX_REVIEW_INPUT_BYTES:
        raise ValueError("review content exceeds 120 KiB")
    return {
        "status": "prepared",
        "ready_for_review": True,
        "artifact": asdict(artifact),
        "content_sha256": sha256(body).hexdigest(),
        "artifact_sha256": artifact_sha256(artifact),
        "prepared_by": "host-local-tool",
        "real_agents_started": False,
    }


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="主助手准备固定版本的审阅正文；不写文件、不调用模型")
    parser.add_argument("--hash-unavailable", action="store_true", help="明确记录本次无法计算摘要")
    args = parser.parse_args()
    try:
        if args.hash_unavailable:
            result = prepare_review_input(None, hashing_available=False)
        else:
            raw = sys.stdin.buffer.read(MAX_REVIEW_INPUT_BYTES + 1)
            if len(raw) > MAX_REVIEW_INPUT_BYTES:
                raise ValueError("review JSON exceeds 120 KiB")
            payload = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=unique_object)
            result = prepare_review_input(payload)
    except (ValueError, TypeError, UnicodeError) as exc:
        print(json.dumps({"status": "invalid_input", "ready_for_review": False, "error": str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["ready_for_review"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
