"""Prepare review hashes locally, with exact text output only on explicit request."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import sys

from ai_team.ccrl import MAX_REVIEW_INPUT_BYTES, artifact_sha256
from ai_team.contracts import Artifact


_ERROR_MESSAGES = {
    "invalid_arguments": "unsupported command-line arguments",
    "invalid_options": "hashing_available and include_content must be booleans",
    "invalid_artifact": "provide exactly id, assignment_id, author_id, version and content",
    "invalid_text": "artifact text fields must be nonempty text without NUL",
    "invalid_version": "version must be a positive integer",
    "invalid_utf8": "review input must contain valid UTF-8 text",
    "input_too_large": "review input exceeds 120 KiB",
    "duplicate_json_key": "duplicate JSON key",
    "invalid_json": "review input must be one valid JSON object",
    "invalid_input": "review input could not be prepared",
    "internal_error": "review preparation failed",
}


class ReviewInputError(ValueError):
    """Only static, known diagnostics may cross the command's output boundary."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _ERROR_MESSAGES else "invalid_input"
        super().__init__(_ERROR_MESSAGES[self.code])


class ReviewArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default message includes unknown arguments supplied by users.
        raise ReviewInputError("invalid_arguments")


def prepare_review_input(
    payload: object, *, hashing_available: bool = True, include_content: bool = False
) -> dict:
    """The host supplies one exact artifact, not a hash invented by a reviewer.

    hashing_available describes this task's permitted local capability. This
    helper does not discover tools, grant permission, or validate Agent work.
    The default receipt omits content. The host must still give the reviewer the
    same exact in-memory text; this receipt alone is not review material.
    """
    if type(hashing_available) is not bool or type(include_content) is not bool:
        raise ReviewInputError("invalid_options")
    if not hashing_available:
        return {
            "status": "needs_version_check",
            "ready_for_review": False,
            "reason": "本次没有获准使用的摘要计算能力，版本核对未完成；不能沿用旧摘要或声称审阅完成。",
            "real_agents_started": False,
        }
    required = {"id", "assignment_id", "author_id", "version", "content"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ReviewInputError("invalid_artifact")
    for field in ("id", "assignment_id", "author_id", "content"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip() or "\0" in value:
            raise ReviewInputError("invalid_text")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError:
            raise ReviewInputError("invalid_utf8") from None
    if type(payload["version"]) is not int or payload["version"] < 1:
        raise ReviewInputError("invalid_version")
    artifact = Artifact(**payload)
    body = artifact.content.encode("utf-8", errors="strict")
    if len(body) > MAX_REVIEW_INPUT_BYTES:
        raise ReviewInputError("input_too_large")
    artifact_output = asdict(artifact) if include_content else {
        "id": artifact.id,
        "assignment_id": artifact.assignment_id,
        "author_id": artifact.author_id,
        "version": artifact.version,
    }
    return {
        "status": "prepared",
        "ready_for_review": True,
        "artifact": artifact_output,
        "content_sha256": sha256(body).hexdigest(),
        "artifact_sha256": artifact_sha256(artifact),
        "prepared_by": "host-local-tool",
        "real_agents_started": False,
    }


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReviewInputError("duplicate_json_key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ReviewInputError("invalid_json")


def main() -> int:
    try:
        parser = ReviewArgumentParser(
            prog="prepare_review.py", allow_abbrev=False,
            description="主助手准备固定版本的审阅摘要；不写文件、不调用模型",
        )
        parser.add_argument("--hash-unavailable", action="store_true", help="明确记录本次无法计算摘要")
        parser.add_argument("--include-content", action="store_true", help="成功时在输出中包含完整原正文")
        args = parser.parse_args()
        if args.hash_unavailable:
            result = prepare_review_input(None, hashing_available=False, include_content=args.include_content)
        else:
            raw = sys.stdin.buffer.read(MAX_REVIEW_INPUT_BYTES + 1)
            if len(raw) > MAX_REVIEW_INPUT_BYTES:
                raise ReviewInputError("input_too_large")
            try:
                decoded = raw.decode("utf-8", errors="strict")
            except UnicodeError:
                raise ReviewInputError("invalid_utf8") from None
            try:
                payload = json.loads(decoded, object_pairs_hook=unique_object, parse_constant=reject_constant)
            except ReviewInputError:
                raise
            except (ValueError, RecursionError):
                raise ReviewInputError("invalid_json") from None
            result = prepare_review_input(payload, include_content=args.include_content)
        serialized = json.dumps(result, ensure_ascii=True, sort_keys=True)
    except ReviewInputError as exc:
        error_code = exc.code
    except (ValueError, TypeError, UnicodeError, RecursionError):
        error_code = "invalid_input"
    except Exception:
        # Unexpected failures must not expose a traceback or an exception's input.
        error_code = "internal_error"
    else:
        print(serialized)
        return 0 if result["ready_for_review"] else 2
    print(json.dumps({
        "status": "invalid_input", "ready_for_review": False,
        "error_code": error_code, "error": _ERROR_MESSAGES[error_code],
    }, ensure_ascii=True, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
