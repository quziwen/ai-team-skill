"""One synchronous, in-memory feedback continuation; no model or I/O driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from hashlib import sha256
import json
import re
from typing import Callable, Literal, TYPE_CHECKING

from .ccrl import artifact_sha256
from .contracts import AgentProfile, Artifact, ReviewPackage, RunResult

if TYPE_CHECKING:
    from .workflow import AITeam


@dataclass(frozen=True)
class FeedbackTarget:
    assignment_id: str
    artifact_id: str
    artifact_version: int
    artifact_sha256: str
    editable_fragments: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackRequest:
    id: str
    original_task: str
    text: str
    targets: tuple[FeedbackTarget, ...]


@dataclass(frozen=True)
class FeedbackRevision:
    author_id: str
    source_artifact_id: str
    source_version: int
    source_sha256: str
    content: str
    summary: str


@dataclass(frozen=True)
class FeedbackChange:
    assignment_id: str
    author_id: str
    reviewer_id: str
    before_artifact_id: str
    before_version: int
    after_artifact_id: str
    after_version: int
    summary: str


@dataclass(frozen=True)
class FeedbackOutcome:
    feedback: FeedbackRequest
    previous: RunResult
    result: RunResult | None
    status: Literal["completed", "needs_user_decision", "needs_verification", "failed"]
    changes: tuple[FeedbackChange, ...]
    candidate_artifacts: tuple[Artifact, ...]
    reused_assignment_ids: tuple[str, ...]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


RevisionProvider = Callable[
    [AgentProfile, Artifact, FeedbackRequest, FeedbackTarget], FeedbackRevision
]


def _text(value, label):
    if type(value) is not str or not value.strip() or "\0" in value:
        raise ValueError(f"{label}必须是非空文字")


def _fragment_spans(content: str, fragments: tuple[str, ...]):
    if type(fragments) is not tuple or not 1 <= len(fragments) <= 32:
        raise ValueError("必须明确本次可以修改的原文片段")
    spans = []
    for fragment in fragments:
        _text(fragment, "可修改片段")
        if content.count(fragment) != 1:
            raise ValueError("可修改片段必须在当前原文中唯一出现")
        begin = content.index(fragment)
        spans.append((begin, begin + len(fragment)))
    spans.sort()
    if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
        raise ValueError("可修改片段不能重叠")
    return tuple(spans)


def _check_scope(before: str, after: str, spans):
    _text(after, "修改后的正文")
    # Character preservation only. Review and current-version evidence still
    # decide whether an edit actually satisfies the user's request.
    for kind, begin, end, _, _ in SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if kind != "equal" and not any(
            allowed_begin <= begin <= end <= allowed_end
            for allowed_begin, allowed_end in spans
        ):
            raise ValueError("修改超出了本次明确的原文范围")


def _validate_request(team, previous, feedback, provider, checker):
    if team._feedback_busy:
        raise ValueError("修改接续正在处理，不能递归或同时开始另一条修改")
    if type(feedback) is not FeedbackRequest:
        raise ValueError("修改意见必须由主助手准备为 FeedbackRequest")
    if type(feedback.id) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", feedback.id):
        raise ValueError("修改意见编号不合要求")
    _text(feedback.text, "修改意见")
    if len(feedback.text.encode("utf-8")) > 8192:
        raise ValueError("本小步只接受一条短修改意见")
    if feedback.original_task != previous.task:
        raise ValueError("原任务必须逐字保留，补充意见不能替换原任务")
    if type(feedback.targets) is not tuple or not feedback.targets:
        raise ValueError("必须先明确受影响的成果，不能凭含糊意见开始修改")
    if not callable(provider) or (checker is not None and not callable(checker)):
        raise ValueError("修改和检查方法必须可调用")


def continue_feedback(
    team: AITeam,
    previous: RunResult,
    feedback: FeedbackRequest,
    *,
    revision_provider: RevisionProvider,
    delivery_checker: Callable[[Artifact, ReviewPackage], list[dict]] | None = None,
) -> FeedbackOutcome:
    """The host supplies scope; identical delivery IDs reuse their old receipt.

    This API does not classify arbitrary language, grant permissions, restore
    sessions after restart, or authenticate a provider's claimed identity.
    """
    from .workflow import _freeze_delivery_requirements

    if type(previous) is not RunResult:
        raise ValueError("必须使用本次运行的成果记录")
    _validate_request(team, previous, feedback, revision_provider, delivery_checker)
    fingerprint = sha256((str(id(previous)) + "\0" + json.dumps(
        asdict(feedback), ensure_ascii=False, sort_keys=True
    )).encode("utf-8")).hexdigest()
    saved = team._feedback_cache.get(feedback.id)
    if saved is not None:
        if saved[0] != fingerprint:
            raise ValueError("同一修改编号不能换正文、对象、版本或范围")
        if saved[1] is None:
            raise ValueError("这条修改已开始处理但没有完整回执，不能自动重试")
        return saved[1]
    if previous is not team._current_result:
        raise ValueError("只能继续同一 AITeam 实例中的当前运行，不能修改旧记录")
    if previous.status not in {"completed", "needs_verification"} or previous.unresolved_disagreements:
        raise ValueError("已有未解决分歧，先明确取舍，不能用新修改绕过")

    assignments = {item.id: item for item in previous.assignments}
    latest = {}
    for artifact in previous.artifacts:
        if artifact.assignment_id not in latest or artifact.version > latest[artifact.assignment_id].version:
            latest[artifact.assignment_id] = artifact
    cycles = {item.assignment_id: item for item in previous.review_cycles}
    targets = {}
    scope_spans = {}
    requirements = dict(team._current_requirements or {key: () for key in assignments})
    for target in feedback.targets:
        if type(target) is not FeedbackTarget or target.assignment_id not in assignments:
            raise ValueError("修改意见引用了未知成果")
        if target.assignment_id in targets:
            raise ValueError("同一修改不能重复列出同一成果")
        original = latest[target.assignment_id]
        cycle = cycles[target.assignment_id]
        if (
            target.artifact_id != original.id
            or type(target.artifact_version) is not int
            or target.artifact_version != original.version
            or target.artifact_sha256 != artifact_sha256(original)
        ):
            raise ValueError("修改意见引用的不是当前成果版本或摘要")
        if (
            assignments[target.assignment_id].agent_id != original.author_id
            or cycle.author_id != original.author_id
            or cycle.result_artifact_id != original.id
            or cycle.result_version != original.version
            or cycle.reviewer_id == original.author_id
        ):
            raise ValueError("原作者、原审阅者或成果记录不一致")
        if type(target.acceptance_criteria) is not tuple or not target.acceptance_criteria:
            raise ValueError("必须说明怎样检查这条修改意见")
        # Validate before deduplication, so malformed/duplicate new criteria are
        # not silently repaired. Existing requirements may be added to, not lost.
        _freeze_delivery_requirements(
            {key: target.acceptance_criteria if key == target.assignment_id else ()
             for key in assignments}, previous.assignments, delivery_checker,
        )
        requirements[target.assignment_id] = tuple(dict.fromkeys(
            (*requirements[target.assignment_id], *target.acceptance_criteria)
        ))
        scope_spans[target.assignment_id] = _fragment_spans(original.content, target.editable_fragments)
        team.router.get(original.author_id)
        team.router.get(cycle.reviewer_id)
        targets[target.assignment_id] = target
    requirements = _freeze_delivery_requirements(requirements, previous.assignments, delivery_checker)

    team._feedback_busy = True
    team._feedback_cache[feedback.id] = (fingerprint, None)
    staged = []
    changes = []
    artifacts = list(previous.artifacts)
    events = ["feedback_received", "feedback_scope_fixed"]
    phase = "作者修改"
    try:
        for assignment in previous.assignments:
            if assignment.id not in targets:
                continue
            target = targets[assignment.id]
            original = latest[assignment.id]
            reviewer_id = cycles[assignment.id].reviewer_id
            author = team.router.get(original.author_id)
            phase = "作者修改"
            reply = revision_provider(author, original, feedback, target)
            if type(reply) is not FeedbackRevision or (
                reply.author_id != original.author_id
                or reply.source_artifact_id != original.id
                or type(reply.source_version) is not int
                or reply.source_version != original.version
                or reply.source_sha256 != artifact_sha256(original)
            ):
                raise ValueError("作者修改必须对应原作者和当前成果")
            _text(reply.summary, "修改说明")
            _check_scope(original.content, reply.content, scope_spans[assignment.id])
            if reply.content == original.content:
                raise ValueError("修改意见尚未落实，不能生成无变化的新版本")
            revised = Artifact(
                f"{assignment.id}-v{original.version + 1}", assignment.id,
                original.author_id, original.version + 1, reply.content,
            )
            artifacts.append(revised)
            staged.append(revised)
            events.append(f"feedback_revised:{assignment.id}:v{revised.version}")

            def guard(candidate, before=original, spans=scope_spans[assignment.id]):
                _check_scope(before.content, candidate.content, spans)

            phase = "原审阅者复核"
            notes = feedback.text + "\n本次检查：" + "；".join(target.acceptance_criteria)
            cycle, final, updates = team._review_artifact(
                previous.task, assignment, revised, reviewer_id, events,
                cycle_id=f"review-{assignment.id}-feedback-{feedback.id}",
                review_notes=notes, revision_guard=guard,
            )
            artifacts.extend(updates)
            staged.extend(updates)
            latest[assignment.id] = final
            cycles[assignment.id] = cycle
            changes.append(FeedbackChange(
                assignment.id, original.author_id, reviewer_id,
                original.id, original.version, final.id, final.version, reply.summary,
            ))
        active_cycle_ids = {cycle.id for cycle in cycles.values()}
        active_arbitrations = tuple(
            item for item in previous.arbitrations if item.review_cycle_id in active_cycle_ids
        )
        phase = "必要检查与合并"
        result = team._finish(
            previous.task, previous.assignments, artifacts, latest,
            [cycles[item.id] for item in previous.assignments], requirements,
            delivery_checker, events, active_arbitrations,
        )
        outcome = FeedbackOutcome(
            feedback, previous, result, result.status, tuple(changes), tuple(staged),
            tuple(key for key in assignments if key not in targets),
            "修改已处理；新版是否可交付以当前复核和必要检查结果为准。",
        )
    except Exception:
        # A failed request remains consumed. Keep valid local drafts and the old
        # run; never echo provider exceptions or silently retry side effects.
        outcome = FeedbackOutcome(
            feedback, previous, None, "failed", tuple(changes), tuple(staged),
            tuple(key for key in assignments if key not in targets),
            f"{phase}未完成。旧版和已核对范围内的草稿保留；本条意见不自动重试，也不交付新版。",
        )
    finally:
        team._feedback_busy = False
    team._feedback_cache[feedback.id] = (fingerprint, outcome)
    return outcome
