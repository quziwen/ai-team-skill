"""Provider-neutral Agent runner adapter for the existing AI Team workflow.

This module only defines an in-process contract.  It deliberately contains no
SDK, subprocess, credential, network, or external-write implementation.  A
caller may supply a runner, while :class:`ai_team.workflow.AITeam` continues to
use the local ``MockBackend`` by default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping, Protocol, cast
import unicodedata
from uuid import uuid4

from .agents import AgentBackend
from .contracts import (
    AgentProfile,
    ArbitrationDecision,
    Artifact,
    Assignment,
    AuthorResponse,
    Confirmation,
    FindingArbitrationDecision,
    IntegrationResult,
    ReviewFeedback,
    ReviewCycle,
)


MAX_AGENT_INPUT_BYTES = 120 * 1024
MAX_AGENT_OUTPUT_BYTES = 120 * 1024


# A compact project adaptation of the qu-ai-wei skill.  It is attached to
# every request so authors, reviewers and the final integrator see the same
# requirements.  The full skill remains the source for dedicated rewriting;
# this tuple keeps the normal AI Team request small and avoids an extra call.
PLAIN_CHINESE_REQUIREMENTS: tuple[str, ...] = (
    "默认使用简体中文；代码、命令、路径、API 和必须保留的原文术语除外。",
    "保留输入中的事实、数字、名称、结论、责任和限制；不得为增加人味编造细节。",
    "先判断材料的使用场景，保留必要的财务、技术和正式用语，不把所有内容强行改成口语。",
    "按出现密度判断表达问题；删掉套路开场、空洞总结、客服式客套、机械排比、堆叠的四字词和没有实际信息的抽象动词。",
    "优先写清谁做什么、需要什么、怎样检查、失败后怎么办；能用具体动作时少用泛化表达。",
    "不要把门禁、门、链、层、线等词当作抽象比喻；直接说明实际对象和动作。",
    "只返回任务要求的内容，不向用户展示内部诊断、改写草稿、自审过程或打磨报告。",
)


class AgentOperation(str, Enum):
    """The six operations already understood by ``AgentBackend``."""

    PRODUCE = "produce"
    REVIEW = "review"
    RESPOND = "respond"
    CONFIRM = "confirm"
    ARBITRATE = "arbitrate"
    INTEGRATE = "integrate"


@dataclass(frozen=True)
class AgentCallRequest:
    """Frozen, self-describing input sent to one provider-neutral runner."""

    call_id: str
    operation: AgentOperation
    actor_id: str
    actor_name: str
    actor_capabilities: tuple[str, ...]
    subject_ids: tuple[str, ...]
    instructions: str
    writing_requirements: tuple[str, ...]
    payload_json: str
    payload_sha256: str
    response_schema: str
    request_sha256: str


@dataclass(frozen=True)
class AgentCallResult:
    """Raw output carrying the request identifiers it claims to answer."""

    call_id: str
    request_sha256: str
    output_text: str


@dataclass(frozen=True)
class _ReviewBinding:
    """Local facts that a later confirmation must match exactly."""

    artifact: Artifact
    reviewer_id: str
    feedback: ReviewFeedback


class AgentRunner(Protocol):
    """Small interface that future local or real runners must implement."""

    def invoke(self, request: AgentCallRequest) -> AgentCallResult: ...


class RunnerContractError(RuntimeError):
    """Raised when a runner request or response fails local validation."""


_COMMON_OBJECT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
}


def _object_schema(
    properties: Mapping[str, object], required: tuple[str, ...]
) -> dict[str, object]:
    return {
        **_COMMON_OBJECT_SCHEMA,
        "properties": dict(properties),
        "required": list(required),
    }


def _conditional_schema(
    schema: dict[str, object], *conditions: Mapping[str, object]
) -> dict[str, object]:
    """Add JSON Schema conditions without weakening the common object rules."""

    return {**schema, "allOf": [dict(item) for item in conditions]}


_RESPONSE_SCHEMAS: dict[AgentOperation, dict[str, object]] = {
    AgentOperation.PRODUCE: _object_schema(
        {"content": {"type": "string", "minLength": 1}},
        ("content",),
    ),
    AgentOperation.REVIEW: _conditional_schema(
        _object_schema(
            {
                "verdict": {
                    "type": "string",
                    "enum": ["no_findings", "changes_requested", "disputed"],
                },
                "comment": {"type": "string", "minLength": 1},
                "disagreement_topic": {"type": ["string", "null"]},
                "author_position": {"type": ["string", "null"]},
                "reviewer_position": {"type": ["string", "null"]},
            },
            (
                "verdict",
                "comment",
                "disagreement_topic",
                "author_position",
                "reviewer_position",
            ),
        ),
        {
            "if": {"properties": {"verdict": {"const": "disputed"}}},
            "then": {
                "properties": {
                    "disagreement_topic": {"type": "string", "minLength": 1},
                    "author_position": {"type": "string", "minLength": 1},
                    "reviewer_position": {"type": "string", "minLength": 1},
                }
            },
            "else": {
                "properties": {
                    "disagreement_topic": {"type": "null"},
                    "author_position": {"type": "null"},
                    "reviewer_position": {"type": "null"},
                }
            },
        },
    ),
    AgentOperation.RESPOND: _conditional_schema(
        _object_schema(
            {
                "decision": {
                    "type": "string",
                    "enum": ["accept", "reject", "clarify"],
                },
                "message": {"type": "string", "minLength": 1},
                "revised_content": {"type": ["string", "null"]},
            },
            ("decision", "message", "revised_content"),
        ),
        {
            "if": {"properties": {"decision": {"const": "accept"}}},
            "then": {
                "properties": {
                    "revised_content": {"type": "string", "minLength": 1}
                }
            },
            "else": {"properties": {"revised_content": {"type": "null"}}},
        },
    ),
    AgentOperation.CONFIRM: _object_schema(
        {
            "resolved": {"type": "boolean"},
            "comment": {"type": "string", "minLength": 1},
        },
        ("resolved", "comment"),
    ),
    AgentOperation.ARBITRATE: _object_schema(
        {
            "finding_decisions": {
                "type": "array",
                "items": _conditional_schema(
                    _object_schema(
                        {
                            "finding_id": {"type": "string", "minLength": 1},
                            "resolved": {"type": "boolean"},
                            "decision": {"type": ["string", "null"]},
                            "reason": {"type": "string", "minLength": 1},
                            "required_user_input": {
                                "type": ["string", "null"]
                            },
                        },
                        (
                            "finding_id",
                            "resolved",
                            "decision",
                            "reason",
                            "required_user_input",
                        ),
                    ),
                    {
                        "if": {"properties": {"resolved": {"const": True}}},
                        "then": {
                            "properties": {
                                "decision": {"type": "string", "minLength": 1},
                                "required_user_input": {"type": "null"},
                            }
                        },
                        "else": {
                            "properties": {
                                "decision": {"type": "null"},
                                "required_user_input": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            }
                        },
                    },
                ),
            }
        },
        ("finding_decisions",),
    ),
    AgentOperation.INTEGRATE: _object_schema(
        {"content": {"type": "string", "minLength": 1}},
        ("content",),
    ),
}


_INSTRUCTIONS: dict[AgentOperation, str] = {
    AgentOperation.PRODUCE: (
        "Complete the assigned work. Return exactly one JSON object matching "
        "response_schema, with no markdown fence or surrounding text. Follow "
        "every item in writing_requirements."
    ),
    AgentOperation.REVIEW: (
        "Review the supplied artifact against its assignment. Return exactly "
        "one JSON object matching response_schema. Apply writing_requirements "
        "when wording is repetitive or obscures the actual action; do not flag "
        "one ordinary professional term by itself."
    ),
    AgentOperation.RESPOND: (
        "Respond to this one review finding. Accept and revise, reject with "
        "reasons, or request clarification. Return exactly one JSON object. "
        "Any revised content must follow writing_requirements."
    ),
    AgentOperation.CONFIRM: (
        "Recheck this one author response and current artifact. Return exactly "
        "one JSON object matching response_schema. Write the explanation using "
        "writing_requirements."
    ),
    AgentOperation.ARBITRATE: (
        "Decide every listed unresolved finding independently and explain each "
        "reason. Return exactly one JSON object matching response_schema. "
        "Write each decision using writing_requirements."
    ),
    AgentOperation.INTEGRATE: (
        "Combine every supplied confirmed artifact and resolved arbitration "
        "decision. Keep all approved content, remove repetitive packaging, and "
        "follow writing_requirements. Return exactly one JSON object matching "
        "response_schema."
    ),
}


def _utf8_bytes(value: str, label: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RunnerContractError(f"{label}必须是有效 UTF-8 文本") from exc


_DEFAULT_IGNORABLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)


def _has_visible_text(value: str) -> bool:
    """Return whether text contains something a person can actually see."""

    _utf8_bytes(value, "文本")
    return any(
        not character.isspace()
        and not _is_default_ignorable(character)
        and unicodedata.category(character)[0] in {"L", "N", "P", "S"}
        for character in value
    )


def _is_default_ignorable(character: str) -> bool:
    """Identify format marks that do not constitute a human-visible edit."""

    codepoint = ord(character)
    if any(start <= codepoint <= end for start, end in _DEFAULT_IGNORABLE_RANGES):
        return True
    category = unicodedata.category(character)
    if category in {"Cf", "Cs", "Co", "Cn"}:
        return True
    if category == "Cc" and not character.isspace():
        return True
    name = unicodedata.name(character, "")
    return (
        "VARIATION SELECTOR" in name
        or name == "COMBINING GRAPHEME JOINER"
        or name.endswith(" FILLER")
        or name
        in {
            "BRAILLE PATTERN BLANK",
            "EGYPTIAN HIEROGLYPH FULL BLANK",
            "EGYPTIAN HIEROGLYPH HALF BLANK",
        }
    )


def _visible_signature(value: str) -> str:
    """Normalize content while keeping meaningful whitespace and indentation."""

    _utf8_bytes(value, "文本")
    normalized = unicodedata.normalize("NFC", value)
    return "".join(
        character
        for character in normalized
        if not _is_default_ignorable(character)
    )


def _validate_json_text(value: object, label: str) -> None:
    """Reject invalid Unicode anywhere in parsed or outgoing JSON data."""

    if type(value) is str:
        _utf8_bytes(cast(str, value), label)
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _validate_json_text(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise RunnerContractError(f"{label}的 JSON 字段名必须是字符串")
            _utf8_bytes(cast(str, key), f"{label}字段名")
            _validate_json_text(item, f"{label}.{key}")


def _jsonable(value: object) -> object:
    """Convert only the small, explicit contract value set to JSON data."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if type(value) is str:
        _utf8_bytes(cast(str, value), "Agent 请求文本")
        return value
    if value is None or type(value) in {int, float, bool}:
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RunnerContractError("Agent 请求的 JSON 字段名必须是字符串")
            _utf8_bytes(key, "Agent 请求的 JSON 字段名")
            result[key] = _jsonable(item)
        return result
    raise RunnerContractError(
        f"Agent 请求包含无法转成 JSON 的类型: {type(value).__name__}"
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RunnerContractError("Agent 请求无法形成规范 JSON") from exc


def _request_material(request: AgentCallRequest) -> dict[str, object]:
    return {
        "call_id": request.call_id,
        "operation": request.operation.value,
        "actor_id": request.actor_id,
        "actor_name": request.actor_name,
        "actor_capabilities": list(request.actor_capabilities),
        "subject_ids": list(request.subject_ids),
        "instructions": request.instructions,
        "writing_requirements": list(request.writing_requirements),
        "payload_json": request.payload_json,
        "payload_sha256": request.payload_sha256,
        "response_schema": request.response_schema,
    }


def calculate_request_sha256(request: AgentCallRequest) -> str:
    """Recalculate the request digest from every non-digest request field."""

    material = _canonical_json(_request_material(request))
    return sha256(_utf8_bytes(material, "Agent 请求")).hexdigest()


class _DuplicateJSONField(ValueError):
    pass


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONField(key)
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(value)


def _parse_json_object(output_text: object) -> dict[str, Any]:
    if type(output_text) is not str:
        raise RunnerContractError("Agent Runner 输出必须是 UTF-8 文本")
    encoded = _utf8_bytes(cast(str, output_text), "Agent Runner 输出")
    if len(encoded) > MAX_AGENT_OUTPUT_BYTES:
        raise RunnerContractError("Agent Runner 输出超过 120 KiB")
    stripped = output_text.strip()
    if not stripped:
        raise RunnerContractError("Agent Runner 输出为空")
    if stripped.startswith("```") or stripped.endswith("```"):
        raise RunnerContractError("Agent Runner 只能输出纯 JSON object")
    try:
        value = json.loads(
            stripped,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonstandard_number,
        )
    except (_DuplicateJSONField, json.JSONDecodeError, ValueError) as exc:
        raise RunnerContractError(
            "Agent Runner 只能输出一个纯 JSON object，不能有前后文字"
        ) from exc
    if type(value) is not dict:
        raise RunnerContractError("Agent Runner 输出的顶层必须是 JSON object")
    _validate_json_text(value, "Agent Runner 输出")
    return cast(dict[str, Any], value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: tuple[str, ...], label: str
) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        details: list[str] = []
        if missing:
            details.append(f"缺少 {', '.join(missing)}")
        if extra:
            details.append(f"多出 {', '.join(extra)}")
        raise RunnerContractError(f"{label}字段错误：{'；'.join(details)}")


def _required_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise RunnerContractError(f"{label}必须是非空字符串")
    text = cast(str, value)
    if not _has_visible_text(text):
        raise RunnerContractError(f"{label}必须包含可见文字")
    return text


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise RunnerContractError(f"{label}必须是字符串或 null")
    text = cast(str, value)
    if not _has_visible_text(text):
        raise RunnerContractError(f"{label}必须包含可见文字或使用 null")
    return text


def _required_identifier(value: object, label: str) -> str:
    text = _required_text(value, label)
    if unicodedata.normalize("NFC", text) != text:
        raise RunnerContractError(f"{label}必须使用标准 Unicode 形式")
    if any(
        character.isspace() or _is_default_ignorable(character)
        for character in text
    ):
        raise RunnerContractError(f"{label}不能包含空白或不可见字符")
    return text


def _validate_capability_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise RunnerContractError(f"{label}必须是非空 tuple")
    capabilities = cast(tuple[object, ...], value)
    checked = tuple(
        _required_identifier(item, f"{label}[{index}]")
        for index, item in enumerate(capabilities)
    )
    if len(checked) != len(set(checked)):
        raise RunnerContractError(f"{label}不能包含重复值")
    return checked


def _validate_agent(agent: AgentProfile) -> None:
    if not isinstance(agent, AgentProfile):
        raise RunnerContractError("Agent 资料格式错误")
    _required_identifier(agent.id, "Agent.id")
    _required_text(agent.name, "Agent.name")
    _validate_capability_tuple(agent.capabilities, "Agent.capabilities")


def _validate_assignment(assignment: Assignment) -> None:
    if not isinstance(assignment, Assignment):
        raise RunnerContractError("子任务资料格式错误")
    _required_identifier(assignment.id, "assignment.id")
    _required_text(assignment.title, "assignment.title")
    _required_text(assignment.goal, "assignment.goal")
    _required_identifier(assignment.agent_id, "assignment.agent_id")
    required = _validate_capability_tuple(
        assignment.required_capabilities, "assignment.required_capabilities"
    )
    matched = _validate_capability_tuple(
        assignment.matched_capabilities, "assignment.matched_capabilities"
    )
    if not set(matched).issubset(required):
        raise RunnerContractError(
            "assignment.matched_capabilities 必须来自 required_capabilities"
        )


def _validate_artifact(artifact: Artifact) -> None:
    if not isinstance(artifact, Artifact):
        raise RunnerContractError("成果资料格式错误")
    _required_identifier(artifact.id, "artifact.id")
    _required_identifier(artifact.assignment_id, "artifact.assignment_id")
    _required_identifier(artifact.author_id, "artifact.author_id")
    if type(artifact.version) is not int or artifact.version < 1:
        raise RunnerContractError("artifact.version 必须是正整数")
    _required_text(artifact.content, "artifact.content")


def _request_state(request: AgentCallRequest) -> tuple[object, ...]:
    """Capture a type-safe immutable snapshot before an in-process call."""

    if type(request.call_id) is not str:
        raise RunnerContractError("Agent 请求 call_id 必须是字符串")
    if type(request.operation) is not AgentOperation:
        raise RunnerContractError("Agent 请求 operation 格式错误")
    scalar_fields = (
        (request.actor_id, "actor_id"),
        (request.actor_name, "actor_name"),
        (request.instructions, "instructions"),
        (request.payload_json, "payload_json"),
        (request.payload_sha256, "payload_sha256"),
        (request.response_schema, "response_schema"),
        (request.request_sha256, "request_sha256"),
    )
    for value, label in scalar_fields:
        if type(value) is not str:
            raise RunnerContractError(f"Agent 请求 {label} 必须是字符串")
        _utf8_bytes(cast(str, value), f"Agent 请求 {label}")
    if type(request.actor_capabilities) is not tuple or any(
        type(item) is not str for item in request.actor_capabilities
    ):
        raise RunnerContractError("Agent 请求的能力标签格式错误")
    if type(request.subject_ids) is not tuple or any(
        type(item) is not str for item in request.subject_ids
    ):
        raise RunnerContractError("Agent 请求的处理对象格式错误")
    if type(request.writing_requirements) is not tuple or any(
        type(item) is not str or not _has_visible_text(item)
        for item in request.writing_requirements
    ):
        raise RunnerContractError("Agent 请求的中文表达要求格式错误")
    if request.writing_requirements != PLAIN_CHINESE_REQUIREMENTS:
        raise RunnerContractError("Agent 请求的中文表达要求与项目规则不一致")
    return (
        request.call_id,
        request.operation,
        request.actor_id,
        request.actor_name,
        request.actor_capabilities,
        request.subject_ids,
        request.instructions,
        request.writing_requirements,
        request.payload_json,
        request.payload_sha256,
        request.response_schema,
        request.request_sha256,
    )


class RunnerBackend(AgentBackend):
    """Adapt one strict ``AgentRunner`` to the existing six backend actions."""

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner
        self._issued_call_ids: set[str] = set()
        self._review_binding_by_assignment_id: dict[str, _ReviewBinding] = {}

    def _new_call_id(self) -> str:
        while True:
            call_id = f"call-{uuid4().hex}"
            if call_id not in self._issued_call_ids:
                self._issued_call_ids.add(call_id)
                return call_id

    def _request(
        self,
        operation: AgentOperation,
        actor: AgentProfile,
        subject_ids: tuple[str, ...],
        payload: Mapping[str, object],
    ) -> AgentCallRequest:
        _validate_agent(actor)
        if type(subject_ids) is not tuple or not subject_ids:
            raise RunnerContractError("Agent 调用必须明确标明处理对象")
        for index, subject_id in enumerate(subject_ids):
            _required_identifier(subject_id, f"subject_ids[{index}]")
        if len(subject_ids) != len(set(subject_ids)):
            raise RunnerContractError("Agent 调用的处理对象不能重复")

        payload_json = _canonical_json(payload)
        payload_bytes = _utf8_bytes(payload_json, "Agent 请求正文")
        if len(payload_bytes) > MAX_AGENT_INPUT_BYTES:
            raise RunnerContractError("Agent 请求正文超过 120 KiB")
        payload_digest = sha256(payload_bytes).hexdigest()
        schema_json = _canonical_json(_RESPONSE_SCHEMAS[operation])
        draft = AgentCallRequest(
            call_id=self._new_call_id(),
            operation=operation,
            actor_id=actor.id,
            actor_name=actor.name,
            actor_capabilities=actor.capabilities,
            subject_ids=subject_ids,
            instructions=_INSTRUCTIONS[operation],
            writing_requirements=PLAIN_CHINESE_REQUIREMENTS,
            payload_json=payload_json,
            payload_sha256=payload_digest,
            response_schema=schema_json,
            request_sha256="",
        )
        request_bytes = _utf8_bytes(
            _canonical_json(_request_material(draft)), "Agent 调用请求"
        )
        if len(request_bytes) > MAX_AGENT_INPUT_BYTES:
            raise RunnerContractError("Agent 调用请求超过 120 KiB")
        return AgentCallRequest(
            **{
                **draft.__dict__,
                "request_sha256": calculate_request_sha256(draft),
            }
        )

    def _invoke(
        self,
        operation: AgentOperation,
        actor: AgentProfile,
        subject_ids: tuple[str, ...],
        payload: Mapping[str, object],
    ) -> dict[str, Any]:
        request = self._request(operation, actor, subject_ids, payload)
        if sha256(_utf8_bytes(request.payload_json, "Agent 请求正文")).hexdigest() != (
            request.payload_sha256
        ):
            raise RunnerContractError("Agent 请求正文的摘要不一致")
        if calculate_request_sha256(request) != request.request_sha256:
            raise RunnerContractError("Agent 请求摘要不一致")

        expected_state = _request_state(request)
        expected_call_id = request.call_id
        expected_request_sha256 = request.request_sha256
        result = self._runner.invoke(request)
        try:
            current_state = _request_state(request)
            current_digest = calculate_request_sha256(request)
        except RunnerContractError as exc:
            raise RunnerContractError("Agent Runner 修改了调用请求") from exc
        if (
            current_state != expected_state
            or current_digest != expected_request_sha256
        ):
            raise RunnerContractError("Agent Runner 修改了调用请求")
        if not isinstance(result, AgentCallResult):
            raise RunnerContractError("Agent Runner 必须返回 AgentCallResult")
        if type(result.call_id) is not str or result.call_id != expected_call_id:
            raise RunnerContractError("Agent Runner 返回了错误的 call_id")
        if (
            type(result.request_sha256) is not str
            or result.request_sha256 != expected_request_sha256
        ):
            raise RunnerContractError("Agent Runner 返回了错误的 request_sha256")
        return _parse_json_object(result.output_text)

    def produce(
        self, agent: AgentProfile, assignment: Assignment, task: str
    ) -> str:
        _validate_agent(agent)
        _validate_assignment(assignment)
        _required_text(task, "task")
        if agent.id != assignment.agent_id:
            raise RunnerContractError("执行者不是该子任务指定的 Agent")
        expected_matches = tuple(
            capability
            for capability in assignment.required_capabilities
            if capability in agent.capabilities
        )
        if assignment.matched_capabilities != expected_matches:
            raise RunnerContractError(
                "子任务记录的能力匹配与实际执行 Agent 不一致"
            )
        raw = self._invoke(
            AgentOperation.PRODUCE,
            agent,
            (assignment.id,),
            {"assignment": assignment, "task": task},
        )
        _require_exact_fields(raw, ("content",), "produce ")
        content = _required_text(raw["content"], "produce.content")
        # A successful new production starts a fresh review exchange. This lets
        # one backend instance handle later tasks that reuse work-1/work-2.
        self._review_binding_by_assignment_id.pop(assignment.id, None)
        return content

    def review(
        self,
        reviewer: AgentProfile,
        artifact: Artifact,
        assignment: Assignment,
        task: str,
    ) -> ReviewFeedback:
        _validate_agent(reviewer)
        _validate_assignment(assignment)
        _validate_artifact(artifact)
        _required_text(task, "task")
        if artifact.assignment_id != assignment.id:
            raise RunnerContractError("待审成果不属于本次子任务")
        if artifact.author_id != assignment.agent_id:
            raise RunnerContractError("待审成果的作者与子任务记录不一致")
        if reviewer.id == artifact.author_id:
            raise RunnerContractError("作者不能审阅自己的成果")
        recorded_binding = self._review_binding_by_assignment_id.get(
            assignment.id
        )
        if (
            recorded_binding is not None
            and recorded_binding.reviewer_id != reviewer.id
        ):
            raise RunnerContractError("同一子任务不能中途更换审阅者")
        raw = self._invoke(
            AgentOperation.REVIEW,
            reviewer,
            (assignment.id, artifact.id),
            {"artifact": artifact, "assignment": assignment, "task": task},
        )
        expected = (
            "verdict",
            "comment",
            "disagreement_topic",
            "author_position",
            "reviewer_position",
        )
        _require_exact_fields(raw, expected, "review ")
        verdict = raw["verdict"]
        if type(verdict) is not str or verdict not in {
            "no_findings",
            "changes_requested",
            "disputed",
        }:
            raise RunnerContractError("review.verdict 的值不受支持")
        comment = _required_text(raw["comment"], "review.comment")
        topic = _optional_text(raw["disagreement_topic"], "disagreement_topic")
        author_position = _optional_text(raw["author_position"], "author_position")
        reviewer_position = _optional_text(
            raw["reviewer_position"], "reviewer_position"
        )
        positions = (topic, author_position, reviewer_position)
        if verdict == "disputed" and any(item is None for item in positions):
            raise RunnerContractError(
                "disputed 必须同时说明争议事项和双方意见"
            )
        if verdict != "disputed" and any(item is not None for item in positions):
            raise RunnerContractError(
                "非 disputed 审阅不能声称存在双方分歧"
            )
        feedback = ReviewFeedback(
            verdict=cast(Any, verdict),
            comment=comment,
            disagreement_topic=topic,
            author_position=author_position,
            reviewer_position=reviewer_position,
        )
        self._review_binding_by_assignment_id[assignment.id] = _ReviewBinding(
            artifact=artifact,
            reviewer_id=reviewer.id,
            feedback=feedback,
        )
        return feedback

    def respond(
        self,
        author: AgentProfile,
        artifact: Artifact,
        feedback: ReviewFeedback,
        task: str,
    ) -> AuthorResponse:
        _validate_agent(author)
        _validate_artifact(artifact)
        _required_text(task, "task")
        if author.id != artifact.author_id:
            raise RunnerContractError("只有成果作者本人可以回应审阅意见")
        raw = self._invoke(
            AgentOperation.RESPOND,
            author,
            (artifact.id,),
            {"artifact": artifact, "feedback": feedback, "task": task},
        )
        _require_exact_fields(
            raw, ("decision", "message", "revised_content"), "respond "
        )
        decision = raw["decision"]
        if type(decision) is not str or decision not in {
            "accept",
            "reject",
            "clarify",
        }:
            raise RunnerContractError("respond.decision 的值不受支持")
        message = _required_text(raw["message"], "respond.message")
        revised = _optional_text(raw["revised_content"], "revised_content")
        if decision == "accept":
            if revised is None:
                raise RunnerContractError("accept 后必须返回修改后的完整内容")
            if _visible_signature(revised) == _visible_signature(artifact.content):
                raise RunnerContractError("accept 后必须对成果做实质修改")
        elif revised is not None:
            raise RunnerContractError("reject 或 clarify 时不能同时提交修改内容")
        return AuthorResponse(
            message=message,
            revised_content=revised,
            decision=cast(Any, decision),
        )

    def confirm(
        self,
        reviewer: AgentProfile,
        feedback: ReviewFeedback,
        response: AuthorResponse,
        artifact: Artifact,
    ) -> Confirmation:
        _validate_agent(reviewer)
        _validate_artifact(artifact)
        if reviewer.id == artifact.author_id:
            raise RunnerContractError("作者不能复核自己的回应")
        binding = self._review_binding_by_assignment_id.get(
            artifact.assignment_id
        )
        if binding is None:
            raise RunnerContractError("复核前必须先完成对应子任务的初审")
        if reviewer.id != binding.reviewer_id:
            raise RunnerContractError("复核者必须是本次初审的原审阅者")
        if artifact.author_id != binding.artifact.author_id:
            raise RunnerContractError("复核成果的作者与初审成果不一致")
        if not isinstance(feedback, ReviewFeedback) or feedback != binding.feedback:
            raise RunnerContractError("复核意见与本次初审记录不一致")
        if not isinstance(response, AuthorResponse):
            raise RunnerContractError("作者回应格式错误")
        if type(response.decision) is not str or response.decision not in {
            "accept",
            "reject",
            "clarify",
        }:
            raise RunnerContractError("作者回应缺少明确决定")
        _required_text(response.message, "response.message")
        if response.decision == "accept":
            revised_content = _optional_text(
                response.revised_content, "response.revised_content"
            )
            if revised_content is None:
                raise RunnerContractError("接受意见后必须提供修改后的完整内容")
            if artifact.version != binding.artifact.version + 1:
                raise RunnerContractError("修改后成果的版本号必须递增一次")
            if artifact.id == binding.artifact.id:
                raise RunnerContractError("修改后成果必须使用新编号")
            if artifact.content != revised_content:
                raise RunnerContractError("复核成果与作者提交的修改内容不一致")
            if _visible_signature(artifact.content) == _visible_signature(
                binding.artifact.content
            ):
                raise RunnerContractError("复核成果没有实质修改")
        else:
            if response.revised_content is not None:
                raise RunnerContractError("未接受意见时不能同时提交修改内容")
            if artifact != binding.artifact:
                raise RunnerContractError("未接受意见时必须复核原成果")
        raw = self._invoke(
            AgentOperation.CONFIRM,
            reviewer,
            (artifact.id,),
            {"artifact": artifact, "feedback": feedback, "response": response},
        )
        _require_exact_fields(raw, ("resolved", "comment"), "confirm ")
        resolved = raw["resolved"]
        if type(resolved) is not bool:
            raise RunnerContractError("confirm.resolved 必须是 boolean")
        return Confirmation(
            resolved=resolved,
            comment=_required_text(raw["comment"], "confirm.comment"),
        )

    def arbitrate(
        self, arbiter: AgentProfile, cycle: ReviewCycle, task: str
    ) -> ArbitrationDecision:
        _validate_agent(arbiter)
        _required_text(task, "task")
        if not isinstance(cycle, ReviewCycle):
            raise RunnerContractError("审阅记录格式错误")
        _required_identifier(cycle.id, "cycle.id")
        _required_identifier(cycle.author_id, "cycle.author_id")
        _required_identifier(cycle.reviewer_id, "cycle.reviewer_id")
        if cycle.author_id == cycle.reviewer_id:
            raise RunnerContractError("审阅记录中的作者和审阅者不能是同一人")
        if arbiter.id in {cycle.author_id, cycle.reviewer_id}:
            raise RunnerContractError("第三方判断者不能是争议双方")
        if cycle.status != "disputed":
            raise RunnerContractError("只有确认存在分歧的审阅记录才能交给第三方")
        finding_ids = tuple(
            item.finding_id
            for item in cycle.reviewer_reassessment_record.finding_reassessments
            if item.decision in {"contested", "owner_required"}
        )
        if not finding_ids:
            raise RunnerContractError("第三方判断前没有可处理的未解决意见")
        raw = self._invoke(
            AgentOperation.ARBITRATE,
            arbiter,
            (cycle.id, *finding_ids),
            {"cycle": cycle, "task": task, "unresolved_finding_ids": finding_ids},
        )
        _require_exact_fields(raw, ("finding_decisions",), "arbitrate ")
        items = raw["finding_decisions"]
        if type(items) is not list:
            raise RunnerContractError("finding_decisions 必须是 array")

        decisions: list[FindingArbitrationDecision] = []
        returned_ids: list[str] = []
        item_fields = (
            "finding_id",
            "resolved",
            "decision",
            "reason",
            "required_user_input",
        )
        for index, value in enumerate(items):
            if type(value) is not dict:
                raise RunnerContractError(
                    f"finding_decisions[{index}] 必须是 JSON object"
                )
            item = cast(dict[str, Any], value)
            _require_exact_fields(item, item_fields, f"finding_decisions[{index}] ")
            finding_id = _required_identifier(
                item["finding_id"], f"finding_decisions[{index}].finding_id"
            )
            resolved = item["resolved"]
            if type(resolved) is not bool:
                raise RunnerContractError(
                    f"finding_decisions[{index}].resolved 必须是 boolean"
                )
            decision = _optional_text(
                item["decision"], f"finding_decisions[{index}].decision"
            )
            reason = _required_text(
                item["reason"], f"finding_decisions[{index}].reason"
            )
            user_input = _optional_text(
                item["required_user_input"],
                f"finding_decisions[{index}].required_user_input",
            )
            if resolved and decision is None:
                raise RunnerContractError("已解决的意见必须给出明确结论")
            if resolved and user_input is not None:
                raise RunnerContractError("已解决的意见不能再要求用户判断")
            if not resolved and decision is not None:
                raise RunnerContractError("未解决的意见不能声称已有结论")
            if not resolved and user_input is None:
                raise RunnerContractError("未解决的意见必须说明需要用户提供什么")
            returned_ids.append(finding_id)
            decisions.append(
                FindingArbitrationDecision(
                    finding_id=finding_id,
                    resolved=resolved,
                    decision=decision,
                    reason=reason,
                    required_user_input=user_input,
                )
            )
        if len(returned_ids) != len(set(returned_ids)):
            raise RunnerContractError("第三方判断包含重复的意见编号")
        if set(returned_ids) != set(finding_ids):
            raise RunnerContractError("第三方判断没有逐条覆盖全部未解决意见")
        return ArbitrationDecision(
            review_cycle_id=cycle.id,
            arbiter_id=arbiter.id,
            finding_decisions=tuple(decisions),
        )

    def integrate(
        self,
        integrator: AgentProfile,
        task: str,
        assignments: tuple[Assignment, ...],
        artifacts: tuple[Artifact, ...],
        arbitrations: tuple[ArbitrationDecision, ...],
    ) -> IntegrationResult:
        _validate_agent(integrator)
        _required_text(task, "task")
        if type(assignments) is not tuple or type(artifacts) is not tuple:
            raise RunnerContractError("合并输入的子任务和成果必须是 tuple")
        if type(arbitrations) is not tuple:
            raise RunnerContractError("合并输入的第三方判断必须是 tuple")
        if not assignments or not artifacts:
            raise RunnerContractError("合并前必须提供子任务和已确认成果")
        assignment_by_id: dict[str, Assignment] = {}
        for assignment in assignments:
            _validate_assignment(assignment)
            assignment_by_id[assignment.id] = assignment
        if len(assignment_by_id) != len(assignments):
            raise RunnerContractError("合并输入包含重复的子任务编号")
        artifact_by_assignment_id: dict[str, Artifact] = {}
        artifact_ids: set[str] = set()
        for artifact in artifacts:
            _validate_artifact(artifact)
            if artifact.id in artifact_ids:
                raise RunnerContractError("合并输入包含重复的成果编号")
            artifact_ids.add(artifact.id)
            if artifact.assignment_id in artifact_by_assignment_id:
                raise RunnerContractError("每个子任务只能提交一份最终成果")
            assignment = assignment_by_id.get(artifact.assignment_id)
            if assignment is None:
                raise RunnerContractError("合并输入包含不属于本次任务的成果")
            if artifact.author_id != assignment.agent_id:
                raise RunnerContractError("成果作者与子任务指定 Agent 不一致")
            artifact_by_assignment_id[artifact.assignment_id] = artifact
        if set(artifact_by_assignment_id) != set(assignment_by_id):
            raise RunnerContractError("合并输入没有覆盖全部子任务")
        arbitration_cycle_ids_list: list[str] = []
        for index, item in enumerate(arbitrations):
            if not isinstance(item, ArbitrationDecision):
                raise RunnerContractError(
                    f"arbitrations[{index}] 的第三方判断格式错误"
                )
            arbitration_cycle_ids_list.append(
                _required_identifier(
                    item.review_cycle_id,
                    f"arbitrations[{index}].review_cycle_id",
                )
            )
            _required_identifier(
                item.arbiter_id, f"arbitrations[{index}].arbiter_id"
            )
        arbitration_cycle_ids = tuple(arbitration_cycle_ids_list)
        if len(arbitration_cycle_ids) != len(set(arbitration_cycle_ids)):
            raise RunnerContractError("合并输入包含重复的第三方判断记录")
        if any(not item.resolved for item in arbitrations):
            raise RunnerContractError("还有未解决的事项时不能生成统一结果")
        subjects = tuple(item.id for item in artifacts) + tuple(
            item.review_cycle_id for item in arbitrations
        )
        raw = self._invoke(
            AgentOperation.INTEGRATE,
            integrator,
            subjects,
            {
                "task": task,
                "assignments": assignments,
                "artifacts": artifacts,
                "arbitrations": arbitrations,
            },
        )
        _require_exact_fields(raw, ("content",), "integrate ")
        return IntegrationResult(
            content=_required_text(raw["content"], "integrate.content"),
            artifact_ids=tuple(item.id for item in artifacts),
            arbitration_review_cycle_ids=tuple(
                item.review_cycle_id for item in arbitrations if item.resolved
            ),
        )


__all__ = [
    "AgentCallRequest",
    "AgentCallResult",
    "AgentOperation",
    "AgentRunner",
    "MAX_AGENT_INPUT_BYTES",
    "MAX_AGENT_OUTPUT_BYTES",
    "PLAIN_CHINESE_REQUIREMENTS",
    "RunnerBackend",
    "RunnerContractError",
    "calculate_request_sha256",
]
