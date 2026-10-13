"""Optional, in-memory call accounting. This is not user authorization or a sandbox.

Only calls that share this object are checked. No discovery, persistence, retry,
provider invocation, or session creation happens here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import re
from subprocess import TimeoutExpired
from threading import Lock
from time import monotonic
from typing import Iterator


OPERATIONS = frozenset({"produce", "review", "respond", "confirm", "arbitrate", "integrate"})


class CallBudgetError(RuntimeError):
    """An attempt is outside the locally recorded task or cannot safely continue."""


def _identifier(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_./:-]{1,180}", value) is not None


@dataclass(frozen=True)
class CallActor:
    actor_id: str
    tool_id: str
    operations: tuple[str, ...]
    model_id: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not _identifier(self.actor_id) or not _identifier(self.tool_id):
            raise ValueError("调用身份和工具必须使用不含正文的短编号")
        if (type(self.operations) is not tuple or not self.operations
                or any(type(op) is not str or op not in OPERATIONS for op in self.operations)
                or len(set(self.operations)) != len(self.operations)):
            raise ValueError("必须明确本身份允许的动作")
        for value in (self.model_id, self.session_id):
            if value is not None and (type(value) is not str or not value.strip()
                                      or len(value) > 180 or any(ord(c) < 32 for c in value)):
                raise ValueError("模型或会话编号不合格；未知应为空")


@dataclass(frozen=True)
class CallRecord:
    call_id: str
    request_sha256: str
    actor_id: str
    operation: str
    tool_id: str
    model_id: str | None
    session_id: str | None
    started_at: str
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    status: str = "running"
    error_type: str | None = None
    token_count: int | None = None
    cost: float | None = None
    verification_status: str = "not_assessed"


class TaskCallBudget:
    """Single-task, sequential accounting; construction never authorizes a call.

    A return only means the wrapped invocation returned; review/business checks
    stay in the existing workflow. Missing usage stays unknown, never zero.
    """

    def __init__(self, task_id: str, task: str, max_calls: int,
                 actors: tuple[CallActor, ...]) -> None:
        if not _identifier(task_id) or type(task) is not str or not task.strip():
            raise ValueError("必须固定任务编号和原文")
        try:
            task.encode("utf-8", errors="strict")
        except UnicodeError:
            raise ValueError("任务原文必须是有效 UTF-8 文本") from None
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("调用上限必须是正整数")
        if (type(actors) is not tuple or not actors
                or any(type(a) is not CallActor for a in actors)
                or len({a.actor_id for a in actors}) != len(actors)):
            raise ValueError("调用身份必须唯一")
        sessions = [(a.tool_id, a.session_id) for a in actors if a.session_id is not None]
        if len(sessions) != len(set(sessions)):
            raise ValueError("不同身份不能复用同一工具内的同一会话")
        self._task_id, self._task, self._max_calls = task_id, task, max_calls
        self._actors = {a.actor_id: a for a in actors}
        self._records: list[CallRecord] = []
        self._call_ids: set[str] = set()
        self._attempt_keys: set[tuple[str, str, str]] = set()
        self._active = False
        self._uncertain = False
        self._lock = Lock()

    def actor(self, actor_id: str) -> CallActor:
        if type(actor_id) is not str or actor_id not in self._actors:
            raise CallBudgetError("身份不在本次任务范围内")
        return self._actors[actor_id]

    def check_reported_session(self, actor_id: str, session_id: str | None) -> None:
        expected = self.actor(actor_id).session_id
        if expected is not None and (type(session_id) is not str or session_id != expected):
            raise CallBudgetError("返回会话与本次固定身份不符或未提供")

    @property
    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def report(self) -> dict:
        with self._lock:
            return {
                "task_id": self._task_id,
                "task_sha256": sha256(self._task.encode("utf-8")).hexdigest(),
                "max_calls": self._max_calls, "used_calls": len(self._records),
                "remaining_calls": self._max_calls - len(self._records),
                "outcome_unknown": self._uncertain,
                "records": [asdict(r) for r in self._records],
                "proves_task_completion": False,
            }

    @contextmanager
    def attempt(self, *, task: str, actor_id: str, operation: str,
                call_id: str, request_sha256: str, content_sha256: str) -> Iterator[None]:
        actor = self.actor(actor_id)
        if type(task) is not str or task != self._task:
            raise CallBudgetError("任务原文与本次记录不一致")
        if type(operation) is not str or operation not in actor.operations:
            raise CallBudgetError("动作不在本次身份允许范围内")
        if not _identifier(call_id) or any(
            type(h) is not str or re.fullmatch(r"[0-9a-f]{64}", h) is None
            for h in (request_sha256, content_sha256)
        ):
            raise CallBudgetError("调用编号或内容摘要不合格")
        key = (actor_id, operation, content_sha256)
        with self._lock:
            if self._uncertain:
                raise CallBudgetError("前次执行结果不明确，停止后续调用，不自动重试")
            if self._active:
                raise CallBudgetError("本地记录不支持递归委派或同时调用")
            if call_id in self._call_ids or key in self._attempt_keys:
                raise CallBudgetError("该调用或相同内容已经尝试，不重复执行")
            if len(self._records) >= self._max_calls:
                raise CallBudgetError("本次任务调用次数已用完")
            index = len(self._records)
            self._records.append(CallRecord(
                call_id, request_sha256, actor_id, operation, actor.tool_id,
                actor.model_id, actor.session_id, datetime.now(timezone.utc).isoformat(),
            ))
            self._call_ids.add(call_id)
            self._attempt_keys.add(key)
            self._active = True
        start = monotonic()
        status, error_type = "returned", None
        try:
            yield
        except BaseException as exc:
            status = ("outcome_unknown" if isinstance(exc, (TimeoutError, TimeoutExpired))
                      or not isinstance(exc, Exception) else "failed")
            error_type = type(exc).__name__
            raise
        finally:
            with self._lock:
                self._records[index] = replace(
                    self._records[index], status=status, error_type=error_type,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=monotonic() - start,
                )
                self._uncertain = status == "outcome_unknown"
                self._active = False
