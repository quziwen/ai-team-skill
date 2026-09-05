"""Capability registry and deterministic local Mock Agent behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

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


Scenario = Literal["auto", "normal", "arbitrated", "unresolved"]


class AgentBackend(Protocol):
    """Vendor-neutral actions; normal MVP runs still use the local Mock."""

    def produce(
        self, agent: AgentProfile, assignment: Assignment, task: str
    ) -> str: ...

    def review(
        self,
        reviewer: AgentProfile,
        artifact: Artifact,
        assignment: Assignment,
        task: str,
    ) -> ReviewFeedback: ...

    def respond(
        self,
        author: AgentProfile,
        artifact: Artifact,
        feedback: ReviewFeedback,
        task: str,
    ) -> AuthorResponse: ...

    def confirm(
        self,
        reviewer: AgentProfile,
        feedback: ReviewFeedback,
        response: AuthorResponse,
        artifact: Artifact,
    ) -> Confirmation: ...

    def arbitrate(
        self,
        arbiter: AgentProfile,
        cycle: ReviewCycle,
        task: str,
    ) -> ArbitrationDecision: ...

    def integrate(
        self,
        integrator: AgentProfile,
        task: str,
        assignments: tuple[Assignment, ...],
        artifacts: tuple[Artifact, ...],
        arbitrations: tuple[ArbitrationDecision, ...],
    ) -> IntegrationResult: ...


DEFAULT_AGENTS: tuple[AgentProfile, ...] = (
    AgentProfile("insight", "分析 Agent", ("analysis", "research", "risk")),
    AgentProfile(
        "delivery", "交付 Agent", ("implementation", "delivery", "testing")
    ),
    AgentProfile("content", "内容 Agent", ("writing", "analysis", "structure")),
    AgentProfile("quality", "质量 Agent", ("quality", "review", "testing")),
    AgentProfile("arbiter", "判断 Agent", ("arbitration", "decision", "risk")),
    AgentProfile("synthesis", "合并 Agent", ("synthesis", "writing", "delivery")),
)


@dataclass
class CapabilityRouter:
    agents: tuple[AgentProfile, ...] = DEFAULT_AGENTS

    def plan(self, task: str) -> tuple[Assignment, ...]:
        task_lower = task.lower()
        if any(
            word in task_lower
            for word in ("代码", "系统", "应用", "api", "程序", "开发", "修复")
        ):
            specs = (
                (
                    "需求与风险分析",
                    "明确目标、约束、风险和完成标准",
                    ("analysis", "risk"),
                ),
                (
                    "最小实现与验证",
                    "给出可执行方案、验证方法和失败兜底",
                    ("implementation", "testing"),
                ),
            )
        elif any(
            word in task_lower
            for word in ("报告", "文章", "文案", "汇报", "ppt", "总结", "纪要")
        ):
            specs = (
                (
                    "内容结构与核心观点",
                    "组织结构并形成有依据的核心内容",
                    ("writing", "analysis"),
                ),
                (
                    "可执行性与质量校验",
                    "检查遗漏、可执行性和验收标准",
                    ("quality", "testing"),
                ),
            )
        else:
            specs = (
                (
                    "目标与约束分析",
                    "明确目标、关键假设、约束和主要风险",
                    ("analysis", "risk"),
                ),
                (
                    "方案与验收设计",
                    "形成最小可执行方案、步骤和验收方法",
                    ("delivery", "testing"),
                ),
            )

        used: set[str] = set()
        assignments: list[Assignment] = []
        for index, (title, goal, required) in enumerate(specs, start=1):
            agent = self.select(required, exclude=used)
            used.add(agent.id)
            matches = tuple(cap for cap in required if cap in agent.capabilities)
            assignments.append(
                Assignment(
                    id=f"work-{index}",
                    title=title,
                    goal=goal,
                    required_capabilities=required,
                    agent_id=agent.id,
                    matched_capabilities=matches,
                )
            )

        if len({item.agent_id for item in assignments}) < 2:
            raise RuntimeError("能力路由必须选择至少两个不同 Agent")
        return tuple(assignments)

    def select(
        self, required: tuple[str, ...], exclude: set[str] | None = None
    ) -> AgentProfile:
        excluded = exclude or set()
        candidates = [agent for agent in self.agents if agent.id not in excluded]
        if not candidates:
            raise RuntimeError("没有可用 Agent")
        return max(
            candidates,
            key=lambda agent: (
                len(set(required).intersection(agent.capabilities)),
                -self.agents.index(agent),
            ),
        )

    def get(self, agent_id: str) -> AgentProfile:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"未知 Agent: {agent_id}")


class MockBackend:
    """Deterministic behavior for local workflow verification only."""

    _UNRESOLVED_PHRASES = (
        "政策口径未明确",
        "预算上限待定",
        "需负责人决定",
        "待我决定",
        "不可自动裁决",
    )
    _ARBITRATED_PHRASES = (
        "存在取舍",
        "优先级冲突",
        "成本与质量冲突",
        "成本优先还是质量优先",
    )

    def __init__(self, scenario: Scenario = "auto") -> None:
        self.scenario = scenario
        self.integrate_calls = 0

    def _scenario_for(self, task: str) -> Scenario:
        if self.scenario != "auto":
            return self.scenario
        if any(phrase in task for phrase in self._UNRESOLVED_PHRASES):
            return "unresolved"
        if any(phrase in task for phrase in self._ARBITRATED_PHRASES):
            return "arbitrated"
        return "normal"

    def produce(
        self, agent: AgentProfile, assignment: Assignment, task: str
    ) -> str:
        if "分析" in assignment.title or "观点" in assignment.title:
            return (
                f"针对“{task}”，先找出最影响结果的原因，记录现状证据和"
                "仍需确认的信息。按影响大小和出错后的补救难度排序；没有证据的"
                "判断明确标成待验证。"
            )
        return (
            f"把“{task}”整理成一份能马上执行的清单。每项写清输入、动作、"
            "交付物和检查方法；先用最小样本试跑，结果符合标准后再扩大范围。"
        )

    def review(
        self,
        reviewer: AgentProfile,
        artifact: Artifact,
        assignment: Assignment,
        task: str,
    ) -> ReviewFeedback:
        scenario = self._scenario_for(task)
        if scenario in ("arbitrated", "unresolved") and assignment.id == "work-1":
            return ReviewFeedback(
                verdict="disputed",
                comment="当前执行顺序存在实质取舍，双方依据不足以直接互相说服。",
                disagreement_topic="方案优先顺序",
                author_position="先保证质量和必要的控制要求，再优化成本和速度。",
                reviewer_position="先控制成本和投入，再逐步提高质量标准。",
            )
        if assignment.id == "work-1":
            comment = (
                "这版还没说明现状依据由谁确认、何时确认，也没有写证据不足时"
                "怎么处理。请补齐。"
            )
        else:
            comment = (
                "这版还没指定每项工作的负责人和截止时间，也没有写试跑失败后"
                "怎么处理。请补齐。"
            )
        return ReviewFeedback(verdict="changes_requested", comment=comment)

    def respond(
        self,
        author: AgentProfile,
        artifact: Artifact,
        feedback: ReviewFeedback,
        task: str,
    ) -> AuthorResponse:
        if feedback.verdict == "changes_requested":
            if artifact.assignment_id == "work-1":
                revised = (
                    f"{artifact.content}\n"
                    "确认安排：每项现状依据标明确认人和确认日期，并保留原始数据"
                    "或访谈记录；证据不足时暂不下结论。"
                )
                message = "接受，已补上确认人、时间和证据不足时的处理。"
            else:
                revised = (
                    f"{artifact.content}\n"
                    "执行安排：每项工作标明负责人、截止时间和完成凭据；试跑不通过"
                    "时停止扩大范围，修正后从最小样本重试。"
                )
                message = "接受，已补上负责人、截止时间和试跑失败后的处理。"
            return AuthorResponse(
                message,
                revised,
                decision="accept",
            )
        if self._scenario_for(task) == "unresolved":
            return AuthorResponse(
                "现有信息不足，请先明确预算、政策或风险偏好。",
                None,
                decision="clarify",
            )
        return AuthorResponse(
            "已说明依据，但该取舍依赖独立判断，暂不修改原方案。",
            None,
            decision="reject",
        )

    def confirm(
        self,
        reviewer: AgentProfile,
        feedback: ReviewFeedback,
        response: AuthorResponse,
        artifact: Artifact,
    ) -> Confirmation:
        if feedback.verdict == "changes_requested" and response.revised_content:
            return Confirmation(True, f"已核对 v{artifact.version} 修订，意见关闭。")
        return Confirmation(False, "已复核作者回应，双方仍未达成一致。")

    def arbitrate(
        self,
        arbiter: AgentProfile,
        cycle: ReviewCycle,
        task: str,
    ) -> ArbitrationDecision:
        scenario = self._scenario_for(task)
        finding_ids = tuple(
            item.finding_id
            for item in cycle.reviewer_reassessment_record.finding_reassessments
            if item.decision in {"contested", "owner_required"}
        )
        if scenario == "unresolved":
            finding_decisions = tuple(
                FindingArbitrationDecision(
                    finding_id=finding_id,
                    resolved=False,
                    decision=None,
                    reason=(
                        "该取舍依赖尚未提供的预算上限、政策口径或风险偏好，"
                        "现有证据不足。"
                    ),
                    required_user_input=(
                        "请明确预算/政策硬约束，以及最低质量要求与成本目标谁优先。"
                    ),
                )
                for finding_id in finding_ids
            )
            return ArbitrationDecision(
                review_cycle_id=cycle.id,
                arbiter_id=arbiter.id,
                finding_decisions=finding_decisions,
            )
        return ArbitrationDecision(
            review_cycle_id=cycle.id,
            arbiter_id=arbiter.id,
            finding_decisions=tuple(
                FindingArbitrationDecision(
                    finding_id=finding_id,
                    resolved=True,
                    decision=(
                        "先满足最低质量和必要的控制要求，再优化成本和速度。"
                    ),
                    reason=(
                        "这些要求一旦未满足，后续补救代价更高；"
                        "成本和节奏则可以在后续小步调整；"
                        "该选择风险更低且可验证。"
                    ),
                )
                for finding_id in finding_ids
            ),
        )

    def integrate(
        self,
        integrator: AgentProfile,
        task: str,
        assignments: tuple[Assignment, ...],
        artifacts: tuple[Artifact, ...],
        arbitrations: tuple[ArbitrationDecision, ...],
    ) -> IntegrationResult:
        self.integrate_calls += 1
        artifact_by_assignment = {item.assignment_id: item for item in artifacts}
        lines = [f"任务：{task}", "", "合并结果："]
        for index, assignment in enumerate(assignments, start=1):
            artifact = artifact_by_assignment[assignment.id]
            compact_content = artifact.content.replace("\n", " ")
            lines.append(f"{index}. {assignment.title}：{compact_content}")
        for arbitration in arbitrations:
            for decision in arbitration.finding_decisions:
                if decision.resolved and decision.decision:
                    lines.append(
                        f"{len(lines) - 2}. 已生效的第三方判断"
                        f"（{decision.finding_id}）：{decision.decision}"
                    )
        return IntegrationResult(
            content="\n".join(lines),
            artifact_ids=tuple(item.id for item in artifacts),
            arbitration_review_cycle_ids=tuple(
                item.review_cycle_id for item in arbitrations if item.resolved
            ),
        )
