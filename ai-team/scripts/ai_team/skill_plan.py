"""Offline role suggestions from capabilities reported by the host.

This does not discover accounts, call models, or grant permission to run them.
The host assistant supplies only the options it can actually select.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from itertools import product
import json
import sys
from typing import Literal

from .agents import CapabilityRouter


@dataclass(frozen=True)
class ModelOption:
    id: str
    capabilities: tuple[str, ...]
    evidence: Literal["measured", "declared", "unknown"] = "unknown"
    cost_rank: int | None = None

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("模型编号不能为空")
        if type(self.capabilities) is not tuple or any(
            type(item) is not str or not item.strip() for item in self.capabilities
        ):
            raise ValueError("能力标签必须是非空文字")
        if self.evidence not in ("measured", "declared", "unknown"):
            raise ValueError("能力依据必须为 measured、declared 或 unknown")
        if self.cost_rank is not None and (
            type(self.cost_rank) is not int or self.cost_rank < 0
        ):
            raise ValueError("成本顺序必须为空或非负整数")


@dataclass(frozen=True)
class AgentToolOption:
    """One tool and the models reachable through that tool, as reported."""

    id: str
    models: tuple[ModelOption, ...]
    availability: Literal["native", "connected", "installed_only", "unavailable"] = "installed_only"
    independent_sessions: bool = False
    session_capacity: int = 0
    max_parallel_agents: int = 1
    can_select_model: bool = False
    default_model_id: str | None = None
    current_model_ids: tuple[str, ...] | None = None
    unavailable_model_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise ValueError("工具编号不能为空")
        if type(self.models) is not tuple or any(type(m) is not ModelOption for m in self.models):
            raise ValueError("工具的模型列表格式错误")
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("同一工具中的模型编号重复")
        if self.availability not in ("native", "connected", "installed_only", "unavailable"):
            raise ValueError("工具的可调用状态不明确")
        if type(self.independent_sessions) is not bool or type(self.can_select_model) is not bool:
            raise ValueError("工具能力必须明确为 true 或 false")
        if type(self.session_capacity) is not int or self.session_capacity < 0:
            raise ValueError("可保留会话数必须是非负整数，未知填 0")
        if type(self.max_parallel_agents) is not int or self.max_parallel_agents < 1:
            raise ValueError("同时运行数量必须是正整数")
        if self.default_model_id is not None and self.default_model_id not in {m.id for m in self.models}:
            raise ValueError("默认模型不在本工具已知模型中")
        for field, ids in (
            ("当前模型清单", self.current_model_ids),
            ("不可用模型清单", self.unavailable_model_ids),
        ):
            if field == "当前模型清单" and ids is None:
                continue
            if type(ids) is not tuple or any(type(item) is not str or not item.strip() for item in ids):
                raise ValueError(f"{field}必须由非空模型编号组成")
            if len(set(ids)) != len(ids):
                raise ValueError(f"{field}中的模型编号重复")


@dataclass(frozen=True)
class RoleSuggestion:
    agent_id: str
    job: str
    model_id: str
    required_capabilities: tuple[str, ...]
    reviews_agent_id: str | None
    activation: str
    reason: str
    tool_id: str = "current-tool"


@dataclass(frozen=True)
class TeamSuggestion:
    mode: str
    roles: tuple[RoleSuggestion, ...]
    max_parallel_agents: int
    merger: str
    limitations: tuple[str, ...]
    plan_only: bool = True
    full_workflow_supported: bool = False


def _rank(
    option: ModelOption,
    required: tuple[str, ...],
    prefer_other_than: str | None = None,
) -> tuple[object, ...]:
    return (
        not set(required).issubset(option.capabilities),
        {"measured": 0, "declared": 1, "unknown": 2}[option.evidence],
        option.cost_rank is None,
        option.cost_rank if option.cost_rank is not None else 0,
        option.id == prefer_other_than,
        option.id,
    )


def suggest_team(
    task: str,
    options: tuple[ModelOption, ...],
    *,
    independent_sessions: bool,
    max_parallel_agents: int = 1,
) -> TeamSuggestion:
    """Compatibility API: a single host reporting full independent sessions."""
    tool = AgentToolOption(
        "current-tool", options, "native", independent_sessions,
        3 if independent_sessions else 0, max_parallel_agents, True,
    )
    return suggest_team_for_tools(task, (tool,), max_parallel_agents=max_parallel_agents)


def suggest_team_for_tools(
    task: str,
    tools: tuple[AgentToolOption, ...],
    *,
    max_parallel_agents: int = 1,
) -> TeamSuggestion:
    """Suggest compatible tool/model pairs without invoking any of them."""
    if type(task) is not str or not task.strip():
        raise ValueError("任务不能为空")
    if type(max_parallel_agents) is not int or max_parallel_agents < 1:
        raise ValueError("同时运行数量必须是正整数")
    if type(tools) is not tuple or any(type(t) is not AgentToolOption for t in tools):
        raise ValueError("工具列表格式错误")
    if len({tool.id for tool in tools}) != len(tools):
        raise ValueError("工具编号重复")

    limitations: list[str] = []
    candidates: list[tuple[AgentToolOption, ModelOption]] = []
    has_callable_tool = False
    has_session_tool = False
    for tool in tools:
        if tool.availability not in ("native", "connected"):
            limitations.append(f"{tool.id} 仅为安装候选或当前不可用，不参与分工。")
            continue
        has_callable_tool = True
        if not tool.independent_sessions or tool.session_capacity < 1:
            limitations.append(f"{tool.id} 未确认可创建并继续独立子会话，不参与互审。")
            continue
        has_session_tool = True
        models = tool.models
        if not tool.can_select_model:
            if tool.default_model_id is not None:
                models = tuple(m for m in models if m.id == tool.default_model_id)
            elif len(models) != 1:
                limitations.append(f"{tool.id} 不能选择模型且默认模型未知，不猜测使用哪个模型。")
                models = ()
        if tool.current_model_ids is None:
            limitations.append(f"{tool.id} 未提供单独的当前模型清单；仅按当次报告的候选安排，未验证实时可用性。")
        else:
            current_ids = set(tool.current_model_ids)
            absent = [model.id for model in models if model.id not in current_ids]
            if absent:
                limitations.append(f"{tool.id} 的候选不在当前模型清单中，已排除：{', '.join(absent)}。")
            models = tuple(model for model in models if model.id in current_ids)
        unavailable_ids = set(tool.unavailable_model_ids)
        rejected = [model.id for model in models if model.id in unavailable_ids]
        if rejected:
            limitations.append(f"{tool.id} 本次已确认不可用的模型不再参与分工：{', '.join(rejected)}。")
        models = tuple(model for model in models if model.id not in unavailable_ids)
        candidates.extend((tool, model) for model in models)

    if not candidates:
        mode = "self_check_only" if has_callable_tool and not has_session_tool else "unavailable"
        limitations.append("没有可用于独立互审的工具和模型组合；只能保留当前助手的草稿及自查。")
        return TeamSuggestion(
            mode, (), 1, "current_assistant", tuple(limitations),
        )

    # Reuse the project's existing task split rather than maintain another one.
    assignments = CapabilityRouter().plan(task)
    requirements = [
        tuple(dict.fromkeys((*assignment.required_capabilities, "review")))
        for assignment in assignments
    ] + [("arbitration", "review")]
    eligible = [
        [(t, m) for t, m in candidates if set(required).issubset(m.capabilities) or "general" in m.capabilities]
        for required in requirements
    ]

    def fits_capacity(picks: tuple[tuple[AgentToolOption, ModelOption], ...]) -> bool:
        counts = Counter(tool.id for tool, _ in picks)
        return all(counts[tool.id] <= tool.session_capacity for tool, _ in picks)

    best = None
    best_score = None
    for author_a, author_b in product(eligible[0], eligible[1]):
        if not fits_capacity((author_a, author_b)):
            continue
        arbiters = [c for c in eligible[2] if fits_capacity((author_a, author_b, c))]
        arbiters.sort(key=lambda c: (_rank(c[1], requirements[2], author_a[1].id), c[0].id))
        arbiter = arbiters[0] if arbiters else None
        score = (
            arbiter is None,
            _rank(author_a[1], requirements[0]),
            _rank(author_b[1], requirements[1], author_a[1].id),
            _rank(arbiter[1], requirements[2], author_a[1].id) if arbiter else (),
            author_a[0].id, author_b[0].id,
        )
        if best_score is None or score < best_score:
            best, best_score = (author_a, author_b, arbiter), score

    if best is None:
        limitations.append("无法为两份互补工作安排两个可保留的独立会话，并同时满足工作和审阅能力。")
        available_capacity = sum({tool.id: tool.session_capacity for tool, _ in candidates}.values())
        mode = "self_check_only" if available_capacity < 2 else "unavailable"
        return TeamSuggestion(mode, (), 1, "current_assistant", tuple(limitations))

    roles: list[RoleSuggestion] = []
    for index, (tool, option) in enumerate(best[:2]):
        required = requirements[index]
        exact = set(required).issubset(option.capabilities)
        reason = "已报告的能力覆盖本次工作和审阅要求。" if exact else "按通用能力暂定，尚未验证这项任务的专项能力。"
        if option.evidence != "measured":
            reason += " 依据来自能力说明，未经过本地专项实测。"
        if option.cost_rank is None:
            reason += " 成本未知。"
        roles.append(RoleSuggestion(
            f"author-{index + 1}", assignments[index].title, option.id, required,
            f"author-{2 - index}", "start", reason, tool.id,
        ))

    arbiter = best[2]
    if arbiter is None:
        limitations.append("只够双人互审，缺少第三个可保留会话或合适的判断能力；有分歧时交给用户，不能称为完整功能。")
    else:
        roles.append(RoleSuggestion(
            "arbiter", "逐条判断剩余分歧并说明理由", arbiter[1].id,
            ("arbitration", "review"), None, "only_if_disputed",
            "使用争议双方以外的新会话；模型可以相同，不能沿用争议双方的身份。",
            arbiter[0].id,
        ))
    if len({role.model_id for role in roles}) == 1:
        if len({role.tool_id for role in roles}) == 1:
            limitations.append("各角色使用同一个模型的独立会话；不代表拥有不同模型的判断方式。")
        else:
            limitations.append("不同工具报告了相同模型标识，是否实际为同一个模型需核对调用记录。")
    limitations.append("能力信息由当前工具或用户提供，分工建议本身不验证账号、效果或费用。")
    author_tools = {tool.id: tool for tool, _ in best[:2]}
    author_counts = Counter(tool.id for tool, _ in best[:2])
    parallel = sum(min(tool.max_parallel_agents, author_counts[tool.id]) for tool in author_tools.values())
    return TeamSuggestion(
        "independent_agents" if arbiter else "peer_review_only", tuple(roles), min(max_parallel_agents, parallel),
        "current_assistant", tuple(limitations),
        full_workflow_supported=arbiter is not None,
    )


def main() -> int:
    """Read an internal capability description from stdin; print a suggestion."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        data = json.load(sys.stdin)
        tools = []
        for item in data["tools"]:
            current_model_ids = item.get("current_model_ids")
            unavailable_model_ids = item.get("unavailable_model_ids", [])
            if (
                current_model_ids is not None and type(current_model_ids) is not list
            ) or type(unavailable_model_ids) is not list:
                raise ValueError("模型清单必须使用 JSON 数组；当前清单未知时填 null")
            models = []
            for model in item["models"]:
                if type(model["capabilities"]) is not list:
                    raise ValueError("能力标签必须使用 JSON 数组")
                models.append(ModelOption(
                    model["id"], tuple(model["capabilities"]),
                    model.get("evidence", "unknown"), model.get("cost_rank"),
                ))
            tools.append(AgentToolOption(
                id=item["id"], models=tuple(models),
                availability=item.get("availability", "installed_only"),
                independent_sessions=item.get("independent_sessions", False),
                session_capacity=item.get("session_capacity", 0),
                max_parallel_agents=item.get("max_parallel_agents", 1),
                can_select_model=item.get("can_select_model", False),
                default_model_id=item.get("default_model_id"),
                current_model_ids=None if current_model_ids is None else tuple(current_model_ids),
                unavailable_model_ids=tuple(unavailable_model_ids),
            ))
        suggestion = suggest_team_for_tools(
            data["task"], tuple(tools),
            max_parallel_agents=data.get("max_parallel_agents", 1),
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"无法生成分工建议：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(suggestion), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
