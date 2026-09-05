"""Phase 2 first step: local, deterministic, single-process project stages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol

from .agents import MockBackend
from .ccrl import artifact_sha256
from .contracts import RunResult
from .workflow import AITeam


LOCAL_EXECUTION_MODE = "local_deterministic_mock"

ProjectDecisionKind = Literal[
    "NEXT_STAGE", "PROJECT_COMPLETED", "NEEDS_USER_DECISION", "BLOCKED"
]
ProjectStatus = Literal["project_completed", "needs_user_decision", "blocked"]
StageRecordStatus = Literal["stage_completed", "needs_user_decision", "blocked"]
AcceptanceState = Literal["passed", "pending", "failed"]


class ProjectContractError(ValueError):
    """A project decision or handoff contradicts checked local facts."""


class ProjectBlockedError(RuntimeError):
    """A known local problem that must stop without an automatic retry."""

    def __init__(self, reason: str, recovery_condition: str) -> None:
        self.reason = reason.strip()
        self.recovery_condition = recovery_condition.strip()
        if not self.reason or not self.recovery_condition:
            raise ValueError("停止原因和继续条件都不能为空")
        super().__init__(self.reason)


@dataclass(frozen=True)
class StageTask:
    stage_id: str
    objective: str
    task: str
    acceptance_criteria: tuple[str, ...]


@dataclass(frozen=True)
class AcceptanceCheck:
    criterion: str
    status: AcceptanceState
    evidence: str


@dataclass(frozen=True)
class ActiveArtifactRef:
    id: str
    stage_id: str
    artifact_id: str
    version: int
    sha256: str


@dataclass(frozen=True)
class HandoffIssue:
    item: str
    reason: str
    owner: str
    required_to_continue: str


@dataclass(frozen=True)
class ProjectPlanningDecision:
    kind: ProjectDecisionKind
    reason: str
    acceptance_status: tuple[AcceptanceCheck, ...]
    next_stage: StageTask | None = None
    unresolved: tuple[HandoffIssue, ...] = tuple()


@dataclass(frozen=True)
class StageHandoff:
    project_goal: str
    stage_id: str
    stage_objective: str
    completed_work: str
    evidence: tuple[str, ...]
    active_artifacts: tuple[ActiveArtifactRef, ...]
    unresolved: tuple[HandoffIssue, ...]
    next_decision: ProjectDecisionKind
    acceptance_status: tuple[AcceptanceCheck, ...]

    @property
    def status(self) -> str:
        return {
            "NEXT_STAGE": "next_stage_required",
            "PROJECT_COMPLETED": "project_completed",
            "NEEDS_USER_DECISION": "needs_user_decision",
            "BLOCKED": "blocked",
        }[self.next_decision]


@dataclass(frozen=True)
class StageRecord:
    status: StageRecordStatus
    stage: StageTask
    run_result: RunResult | None
    decision: ProjectPlanningDecision
    handoff: StageHandoff


@dataclass(frozen=True)
class ProjectRunResult:
    project_goal: str
    status: ProjectStatus
    execution_mode: str
    acceptance_criteria: tuple[str, ...]
    stages: tuple[StageRecord, ...]
    unresolved: tuple[HandoffIssue, ...]
    final_result: str | None
    events: tuple[str, ...]


@dataclass(frozen=True)
class _PlannerInput:
    """Checked facts given to the local planner; full RunResult stays outside."""

    project_goal: str
    stage: StageTask
    completed_work: str
    evidence: tuple[str, ...]
    active_artifacts: tuple[ActiveArtifactRef, ...]
    acceptance_status: tuple[AcceptanceCheck, ...]
    next_stage_number: int


class _StageRunner(Protocol):
    execution_mode: str

    def run(self, task: str) -> RunResult: ...


class _ProjectPlanner(Protocol):
    execution_mode: str

    def decide(self, facts: _PlannerInput) -> ProjectPlanningDecision: ...


class LocalStageRunner:
    """Use Phase 1 with its deterministic Mock backend."""

    execution_mode = LOCAL_EXECUTION_MODE

    def __init__(self) -> None:
        self._team = AITeam(backend=MockBackend())

    def run(self, task: str) -> RunResult:
        return self._team.run(task)


class LocalProjectPlanner:
    """Fixed local rules; this is not a real planning Agent."""

    execution_mode = LOCAL_EXECUTION_MODE

    def decide(self, facts: _PlannerInput) -> ProjectPlanningDecision:
        checks = _checks_after_success(facts)
        pending = tuple(item for item in checks if item.status == "pending")
        if not pending:
            return ProjectPlanningDecision(
                kind="PROJECT_COMPLETED",
                reason="全部本地流程样例已经按预设规则执行。",
                acceptance_status=checks,
            )

        criterion = pending[0].criterion
        return ProjectPlanningDecision(
            kind="NEXT_STAGE",
            reason="本地流程样例还有未执行的验收项。",
            acceptance_status=checks,
            next_stage=StageTask(
                stage_id=f"stage-{facts.next_stage_number}",
                objective=f"完成验收项：{criterion}",
                task=f"围绕“{facts.project_goal}”，完成以下验收项：{criterion}",
                acceptance_criteria=(criterion,),
            ),
        )


class ProjectWorkflow:
    """Internal/testable runner. The public entry is safe-default run_project()."""

    execution_mode = LOCAL_EXECUTION_MODE

    def __init__(
        self,
        stage_runner: _StageRunner | None = None,
        planner: _ProjectPlanner | None = None,
    ) -> None:
        self.stage_runner = stage_runner or LocalStageRunner()
        self.planner = planner or LocalProjectPlanner()
        if getattr(self.stage_runner, "execution_mode", None) != LOCAL_EXECUTION_MODE:
            raise ValueError("Phase 2 第一小步只允许本地确定性 Mock 阶段执行器")
        if getattr(self.planner, "execution_mode", None) != LOCAL_EXECUTION_MODE:
            raise ValueError("Phase 2 第一小步只允许本地确定性 Mock 项目判断器")

    def run(
        self,
        project_goal: str,
        acceptance_criteria: tuple[str, ...],
        initial_stage_task: str,
    ) -> ProjectRunResult:
        goal, criteria, task = _clean_inputs(
            project_goal, acceptance_criteria, initial_stage_task
        )
        checks = tuple(
            AcceptanceCheck(
                criterion=item,
                status="pending",
                evidence="本地流程样例：对应阶段尚未执行。",
            )
            for item in criteria
        )
        stage = StageTask(
            stage_id="stage-1",
            objective=f"完成验收项：{criteria[0]}",
            task=task,
            acceptance_criteria=(criteria[0],),
        )
        records: list[StageRecord] = []
        events: list[str] = ["project_received"]
        used_stage_ids: set[str] = set()
        used_task_hashes: set[str] = set()

        # Each accepted next stage must consume a still-pending criterion.
        # The fixed range gives a hard end and there are no automatic retries.
        for stage_number in range(1, len(criteria) + 1):
            problem = _stage_problem(
                stage, checks, used_stage_ids, used_task_hashes
            )
            if problem:
                return _stopped_result(
                    goal,
                    criteria,
                    stage,
                    checks,
                    "BLOCKED",
                    problem,
                    "修正下一阶段任务后重新开始本地运行。",
                    None,
                    records,
                    events,
                )

            used_stage_ids.add(_normalize(stage.stage_id))
            used_task_hashes.add(_task_hash(stage.task))
            events.append(f"stage_started:{stage.stage_id}")
            try:
                phase_one = self.stage_runner.run(stage.task)
            except ProjectBlockedError as exc:
                return _stopped_result(
                    goal,
                    criteria,
                    stage,
                    checks,
                    "BLOCKED",
                    exc.reason,
                    exc.recovery_condition,
                    None,
                    records,
                    events,
                )

            if not isinstance(phase_one, RunResult):
                raise TypeError("阶段执行器必须返回 RunResult")
            if phase_one.task != stage.task:
                return _stopped_result(
                    goal,
                    criteria,
                    stage,
                    checks,
                    "BLOCKED",
                    "Phase 1 返回的是另一项任务的结果。",
                    "修正阶段执行器，确保返回结果与本次任务逐字一致。",
                    phase_one,
                    records,
                    events,
                )
            if phase_one.status == "needs_user_decision":
                issues = tuple(
                    HandoffIssue(
                        item=item.topic,
                        reason=item.arbitration_reason,
                        owner="用户",
                        required_to_continue=item.required_user_input,
                    )
                    for item in phase_one.unresolved_disagreements
                )
                if not issues:
                    raise ProjectContractError(
                        "Phase 1 要求用户决定时必须列明具体事项"
                    )
                return _stopped_result(
                    goal,
                    criteria,
                    stage,
                    checks,
                    "NEEDS_USER_DECISION",
                    "Phase 1 仍有必须由用户决定的事项。",
                    issues[0].required_to_continue,
                    phase_one,
                    records,
                    events,
                    issues,
                )
            if phase_one.status != "completed":
                raise RuntimeError(f"未知的 Phase 1 状态：{phase_one.status}")

            try:
                facts = _planner_input(
                    goal,
                    stage,
                    phase_one,
                    checks,
                    next_stage_number=stage_number + 1,
                )
            except ProjectContractError as exc:
                return _stopped_result(
                    goal,
                    criteria,
                    stage,
                    checks,
                    "BLOCKED",
                    f"Phase 1 结果未通过项目交接检查：{exc}",
                    "修正阶段结果后重新开始本地运行。",
                    phase_one,
                    records,
                    events,
                )

            proposed = self.planner.decide(facts)
            try:
                _check_decision(
                    proposed, facts, used_stage_ids, used_task_hashes
                )
                decision = proposed
            except ProjectContractError as exc:
                issue = HandoffIssue(
                    item="项目判断不合格",
                    reason=str(exc),
                    owner="AI Team",
                    required_to_continue="修正项目判断规则并重新运行本地测试。",
                )
                decision = ProjectPlanningDecision(
                    kind="BLOCKED",
                    reason=f"项目判断未通过检查：{exc}",
                    acceptance_status=_checks_after_success(facts),
                    unresolved=(issue,),
                )

            handoff = _completed_handoff(facts, decision)
            validate_stage_handoff(handoff, goal, criteria)
            records.append(
                StageRecord(
                    status="stage_completed",
                    stage=stage,
                    run_result=phase_one,
                    decision=decision,
                    handoff=handoff,
                )
            )
            checks = decision.acceptance_status
            events.extend((f"stage_completed:{stage.stage_id}", handoff.status))

            if decision.kind == "NEXT_STAGE":
                if decision.next_stage is None:  # Checked above.
                    raise RuntimeError("NEXT_STAGE 缺少下一阶段任务")
                stage = decision.next_stage
                continue
            if decision.kind == "PROJECT_COMPLETED":
                return ProjectRunResult(
                    project_goal=goal,
                    status="project_completed",
                    execution_mode=LOCAL_EXECUTION_MODE,
                    acceptance_criteria=criteria,
                    stages=tuple(records),
                    unresolved=tuple(),
                    final_result=_project_summary(goal, records),
                    events=tuple(events),
                )
            return ProjectRunResult(
                project_goal=goal,
                status=(
                    "needs_user_decision"
                    if decision.kind == "NEEDS_USER_DECISION"
                    else "blocked"
                ),
                execution_mode=LOCAL_EXECUTION_MODE,
                acceptance_criteria=criteria,
                stages=tuple(records),
                unresolved=decision.unresolved,
                final_result=None,
                events=tuple(events),
            )

        raise RuntimeError("项目阶段数量超过项目验收项数量")


def run_project(
    project_goal: str,
    acceptance_criteria: tuple[str, ...],
    initial_stage_task: str,
) -> ProjectRunResult:
    """Run the local sample; it does not perform real business acceptance."""

    return ProjectWorkflow().run(
        project_goal, acceptance_criteria, initial_stage_task
    )


def _clean_inputs(
    project_goal: str,
    acceptance_criteria: tuple[str, ...],
    initial_stage_task: str,
) -> tuple[str, tuple[str, ...], str]:
    goal = project_goal.strip()
    task = initial_stage_task.strip()
    if not goal:
        raise ValueError("项目目标不能为空")
    if not task:
        raise ValueError("当前阶段任务不能为空")
    if isinstance(acceptance_criteria, str):
        raise ValueError("项目验收项必须是字符串元组")
    criteria = tuple(item.strip() for item in acceptance_criteria)
    if not criteria or any(not item for item in criteria):
        raise ValueError("项目验收项不能为空")
    normalized = tuple(_normalize(item) for item in criteria)
    if len(normalized) != len(set(normalized)):
        raise ValueError("项目验收项不能重复")
    return goal, criteria, task


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _task_hash(task: str) -> str:
    return sha256(_normalize(task).encode("utf-8")).hexdigest()


def _stage_problem(
    stage: StageTask,
    checks: tuple[AcceptanceCheck, ...],
    used_stage_ids: set[str],
    used_task_hashes: set[str],
) -> str | None:
    values = (stage.stage_id, stage.objective, stage.task)
    if any(not value.strip() for value in values):
        return "阶段编号、阶段目标和阶段任务都不能为空"
    if any(value != value.strip() for value in values):
        return "阶段编号、阶段目标和阶段任务不能带首尾空白"
    if not stage.acceptance_criteria or any(
        not item.strip() or item != item.strip()
        for item in stage.acceptance_criteria
    ):
        return "阶段必须负责至少一个有效的项目验收项"
    normalized = tuple(_normalize(item) for item in stage.acceptance_criteria)
    if len(normalized) != len(set(normalized)):
        return "同一阶段不能重复登记项目验收项"
    pending = {item.criterion for item in checks if item.status == "pending"}
    if not set(stage.acceptance_criteria).issubset(pending):
        return "下一阶段必须处理尚未完成的项目验收项"
    if _normalize(stage.stage_id) in used_stage_ids:
        return f"阶段编号重复：{stage.stage_id}"
    if _task_hash(stage.task) in used_task_hashes:
        return "下一阶段任务与已经执行的任务重复"
    return None


def _planner_input(
    goal: str,
    stage: StageTask,
    result: RunResult,
    checks: tuple[AcceptanceCheck, ...],
    next_stage_number: int,
) -> _PlannerInput:
    if not result.final_result:
        raise ProjectContractError("已完成阶段必须有 Phase 1 最终结果")
    if len(result.final_artifact_ids) != len(set(result.final_artifact_ids)):
        raise ProjectContractError("Phase 1 最终成果编号重复")
    artifacts = {item.id: item for item in result.artifacts}
    active: list[ActiveArtifactRef] = []
    for artifact_id in result.final_artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise ProjectContractError("Phase 1 最终成果编号不存在")
        active.append(
            ActiveArtifactRef(
                id=f"{stage.stage_id}:{artifact.id}",
                stage_id=stage.stage_id,
                artifact_id=artifact.id,
                version=artifact.version,
                sha256=artifact_sha256(artifact),
            )
        )
    if not active:
        raise ProjectContractError("已完成阶段缺少可继续使用的成果")
    evidence = (
        f"local_deterministic_mock:{stage.stage_id}:run_status=completed",
        *(
            f"artifact:{item.id}@v{item.version}:sha256={item.sha256}"
            for item in active
        ),
    )
    return _PlannerInput(
        project_goal=goal,
        stage=stage,
        completed_work=result.final_result,
        evidence=tuple(evidence),
        active_artifacts=tuple(active),
        acceptance_status=checks,
        next_stage_number=next_stage_number,
    )


def _checks_after_success(facts: _PlannerInput) -> tuple[AcceptanceCheck, ...]:
    stage_criteria = set(facts.stage.acceptance_criteria)
    artifacts = ", ".join(
        f"{item.id}@v{item.version}:sha256={item.sha256}"
        for item in facts.active_artifacts
    )
    evidence = (
        "本地流程样例按预设规则标记为 passed；只验证阶段推进、交接和"
        f"停止判断，未核验真实业务内容。阶段={facts.stage.stage_id}；"
        f"成果={artifacts}。"
    )
    return tuple(
        AcceptanceCheck(item.criterion, "passed", evidence)
        if item.criterion in stage_criteria
        else item
        for item in facts.acceptance_status
    )


def _validate_checks(
    checks: tuple[AcceptanceCheck, ...], criteria: tuple[str, ...]
) -> None:
    if tuple(item.criterion for item in checks) != criteria:
        raise ProjectContractError("项目验收结果必须逐项对应原始验收项")
    if any(item.status not in {"passed", "pending", "failed"} for item in checks):
        raise ProjectContractError("项目验收结果包含未知状态")
    if any(not item.evidence.strip() for item in checks):
        raise ProjectContractError("每个项目验收结果都必须附本地证据或说明")


def _check_decision(
    decision: ProjectPlanningDecision,
    facts: _PlannerInput,
    used_stage_ids: set[str],
    used_task_hashes: set[str],
) -> None:
    if not isinstance(decision, ProjectPlanningDecision):
        raise ProjectContractError("项目判断器必须返回 ProjectPlanningDecision")
    if not decision.reason.strip():
        raise ProjectContractError("项目判断必须说明理由")
    criteria = tuple(item.criterion for item in facts.acceptance_status)
    _validate_checks(decision.acceptance_status, criteria)
    if decision.acceptance_status != _checks_after_success(facts):
        raise ProjectContractError("项目判断修改了未经本次结果支持的验收状态")

    pending = tuple(
        item for item in decision.acceptance_status if item.status == "pending"
    )
    failed = tuple(
        item for item in decision.acceptance_status if item.status == "failed"
    )
    if decision.kind == "PROJECT_COMPLETED":
        if pending or failed:
            raise ProjectContractError("项目仍有未通过或待完成事项，不能声称完成")
        if decision.next_stage is not None or decision.unresolved:
            raise ProjectContractError("项目完成时不能再生成下一任务或未决事项")
        return
    if decision.kind == "NEXT_STAGE":
        if failed or not pending or decision.next_stage is None:
            raise ProjectContractError("NEXT_STAGE 与当前项目验收结果不一致")
        if decision.unresolved:
            raise ProjectContractError("生成下一任务时不能同时留下未决事项")
        problem = _stage_problem(
            decision.next_stage,
            decision.acceptance_status,
            used_stage_ids,
            used_task_hashes,
        )
        if problem:
            raise ProjectContractError(problem)
        return
    if decision.kind not in {"NEEDS_USER_DECISION", "BLOCKED"}:
        raise ProjectContractError("项目判断包含未知决定")
    if decision.next_stage is not None or not decision.unresolved:
        raise ProjectContractError("停止处理时必须取消下一任务并写清未解决事项")


def _completed_handoff(
    facts: _PlannerInput, decision: ProjectPlanningDecision
) -> StageHandoff:
    return StageHandoff(
        project_goal=facts.project_goal,
        stage_id=facts.stage.stage_id,
        stage_objective=facts.stage.objective,
        completed_work=facts.completed_work,
        evidence=facts.evidence,
        active_artifacts=facts.active_artifacts,
        unresolved=decision.unresolved,
        next_decision=decision.kind,
        acceptance_status=decision.acceptance_status,
    )


def validate_stage_handoff(
    handoff: StageHandoff,
    project_goal: str,
    acceptance_criteria: tuple[str, ...],
) -> None:
    if handoff.project_goal != project_goal:
        raise ProjectContractError("阶段交接引用了错误的项目目标")
    if handoff.stage_id != handoff.stage_id.strip():
        raise ProjectContractError("阶段编号不能带首尾空白")
    if not handoff.stage_id or not handoff.stage_objective.strip():
        raise ProjectContractError("阶段交接缺少阶段编号或阶段目标")
    if not handoff.completed_work.strip():
        raise ProjectContractError("阶段交接缺少实际完成内容")
    if not handoff.evidence or any(not item.strip() for item in handoff.evidence):
        raise ProjectContractError("阶段交接缺少本地证据")
    artifact_ids = tuple(item.id for item in handoff.active_artifacts)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ProjectContractError("阶段交接包含重复成果")
    for item in handoff.active_artifacts:
        if item.stage_id != handoff.stage_id or not item.id.startswith(
            f"{handoff.stage_id}:"
        ):
            raise ProjectContractError("阶段成果引用了错误的阶段编号")
        if item.version < 1 or len(item.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in item.sha256
        ):
            raise ProjectContractError("阶段成果版本或 SHA-256 不合格")
    _validate_checks(handoff.acceptance_status, acceptance_criteria)
    for item in handoff.unresolved:
        if not all(
            value.strip()
            for value in (
                item.item,
                item.reason,
                item.owner,
                item.required_to_continue,
            )
        ):
            raise ProjectContractError("未解决事项必须写清内容、原因、负责人和继续条件")
    incomplete = tuple(
        item for item in handoff.acceptance_status if item.status != "passed"
    )
    if handoff.next_decision == "PROJECT_COMPLETED":
        if incomplete or handoff.unresolved or not handoff.active_artifacts:
            raise ProjectContractError("项目完成结论与验收结果或成果不一致")
    elif handoff.next_decision == "NEXT_STAGE":
        if not incomplete or handoff.unresolved or not handoff.active_artifacts:
            raise ProjectContractError("下一阶段结论与验收结果或成果不一致")
    elif handoff.next_decision in {"NEEDS_USER_DECISION", "BLOCKED"}:
        if not handoff.unresolved:
            raise ProjectContractError("停止项目时必须列明未解决事项")
    else:
        raise ProjectContractError("阶段交接包含未知项目决定")


def _stopped_result(
    goal: str,
    criteria: tuple[str, ...],
    stage: StageTask,
    checks: tuple[AcceptanceCheck, ...],
    kind: Literal["NEEDS_USER_DECISION", "BLOCKED"],
    reason: str,
    recovery: str,
    run_result: RunResult | None,
    records: list[StageRecord],
    events: list[str],
    issues: tuple[HandoffIssue, ...] | None = None,
) -> ProjectRunResult:
    unresolved = issues or (
        HandoffIssue(
            item=stage.objective,
            reason=reason,
            owner="AI Team",
            required_to_continue=recovery,
        ),
    )
    decision = ProjectPlanningDecision(
        kind=kind,
        reason=reason,
        acceptance_status=checks,
        unresolved=unresolved,
    )
    status: ProjectStatus = (
        "needs_user_decision" if kind == "NEEDS_USER_DECISION" else "blocked"
    )
    if run_result is None:
        completed_work = "本阶段未完成。"
    elif kind == "NEEDS_USER_DECISION":
        completed_work = "本阶段有审阅记录，但存在未解决事项，未生成可合并结果。"
    else:
        completed_work = "Phase 1 返回了结果，但项目交接检查未通过。"
    handoff = StageHandoff(
        project_goal=goal,
        stage_id=stage.stage_id,
        stage_objective=stage.objective,
        completed_work=completed_work,
        evidence=(f"local_deterministic_mock:{stage.stage_id}:status={status}",),
        active_artifacts=tuple(),
        unresolved=unresolved,
        next_decision=kind,
        acceptance_status=checks,
    )
    validate_stage_handoff(handoff, goal, criteria)
    records.append(
        StageRecord(
            status=status,
            stage=stage,
            run_result=run_result,
            decision=decision,
            handoff=handoff,
        )
    )
    events.extend((f"stage_stopped:{stage.stage_id}", status))
    return ProjectRunResult(
        project_goal=goal,
        status=status,
        execution_mode=LOCAL_EXECUTION_MODE,
        acceptance_criteria=criteria,
        stages=tuple(records),
        unresolved=unresolved,
        final_result=None,
        events=tuple(events),
    )


def _project_summary(goal: str, records: list[StageRecord]) -> str:
    lines = [
        "本地流程样例汇总（不代表真实 Agent 的专业结论或真实业务验收）",
        f"项目目标：{goal}",
        "",
        "已完成阶段：",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            (
                f"{index}. {record.stage.stage_id}｜{record.stage.objective}",
                record.handoff.completed_work,
            )
        )
    lines.extend(("", "本地样例检查结果："))
    for item in records[-1].handoff.acceptance_status:
        lines.append(f"- {item.criterion}：{item.status}；{item.evidence}")
    return "\n".join(lines)
