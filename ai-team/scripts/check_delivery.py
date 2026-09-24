"""Read-only delivery checks. Input through stdin; no sources echoed on failure."""

import json
import sys

from ai_team.ccrl import MAX_REVIEW_INPUT_BYTES
from ai_team.verification import VerificationInputError, verify_delivery


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationInputError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise VerificationInputError("invalid JSON constant")


def main() -> int:
    try:
        if len(sys.argv) != 1:
            raise VerificationInputError("unsupported arguments; provide JSON on stdin")
        raw = sys.stdin.buffer.read(MAX_REVIEW_INPUT_BYTES + 1)
        if len(raw) > MAX_REVIEW_INPUT_BYTES:
            raise VerificationInputError("input exceeds 120 KiB")
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        result = verify_delivery(payload)
    except VerificationInputError as exc:
        error = str(exc)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        error = "invalid UTF-8 JSON input"
    except Exception:
        error = "delivery check failed"
    else:
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result["checks_passed"] else 2
    print(json.dumps({"status": "invalid_input", "checks_passed": False,
                      "proves_task_completion": False, "error": error}, ensure_ascii=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
