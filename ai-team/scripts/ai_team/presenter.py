"""User-facing renderer that hides workflow and backend internals."""

from __future__ import annotations

from .agents import CapabilityRouter
from dataclasses import replace

from .contracts import Artifact, RunResult


_CAPABILITY_LABELS = {
    "analysis": "分析",
    "research": "研究",
    "risk": "风险识别",
    "implementation": "实施",
    "delivery": "交付",
    "testing": "验证",
    "writing": "写作",
    "structure": "结构化",
    "quality": "质量检查",
    "review": "审阅",
    "arbitration": "分歧判断",
    "decision": "决策",
    "synthesis": "合并",
}

_SEVERITY_LABELS = {
    "critical": "严重",
    "major": "较重要",
    "minor": "一般",
}

_RESPONSE_LABELS = {
    "accept": "接受",
    "reject": "反驳",
    "clarify": "需要澄清",
}

_REASSESSMENT_LABELS = {
    "satisfied": "确认已解决",
    "withdrawn": "撤回原意见",
    "contested": "仍有分歧",
    "owner_required": "需要你决定",
}


def _latest_artifacts(result: RunResult) -> dict[str, Artifact]:
    latest: dict[str, Artifact] = {}
    for artifact in result.artifacts:
        current = latest.get(artifact.assignment_id)
        if current is None or artifact.version > current.version:
            latest[artifact.assignment_id] = artifact
    return latest


def render_markdown(
    result: RunResult, router: CapabilityRouter | None = None
) -> str:
    registry = router or CapabilityRouter()
    lines = ["# AI Team 执行结果", "", f"任务：{result.task}", "", "## 分工"]
    for assignment in result.assignments:
        agent = registry.get(assignment.agent_id)
        reason = "、".join(
            _CAPABILITY_LABELS.get(item, item) for item in assignment.matched_capabilities
        ) or "综合能力匹配"
        lines.append(
            f"- {agent.name}：{assignment.title}；负责{assignment.goal}；分配依据：{reason}。"
        )

    latest = _latest_artifacts(result)
    lines.extend(("", "## 结果"))
    for assignment in result.assignments:
        artifact = latest[assignment.id]
        agent = registry.get(artifact.author_id)
        lines.extend(
            (
                "",
                f"### {agent.name}｜{assignment.title}｜v{artifact.version}",
                "",
                artifact.content,
            )
        )

    lines.extend(("", "## 审阅意见"))
    for cycle in result.review_cycles:
        reviewer = registry.get(cycle.reviewer_id)
        author = registry.get(cycle.author_id)
        lines.append(
            f"- {reviewer.name} 审阅 {author.name} 的 v{cycle.reviewed_version}：{cycle.review_comment}"
        )
        responses = {
            item.finding_id: item
            for item in cycle.structured_author_response.finding_responses
        }
        reassessments = {
            item.finding_id: item
            for item in cycle.reviewer_reassessment_record.finding_reassessments
        }
        for finding in cycle.review_memo.findings:
            response = responses[finding.id]
            reassessment = reassessments[finding.id]
            lines.extend(
                (
                    f"  - 意见（{_SEVERITY_LABELS[finding.severity]}）：{finding.problem}",
                    f"  - 依据：{finding.evidence}",
                    f"  - 建议：{finding.suggestion}",
                    f"  - 作者回应（{_RESPONSE_LABELS[response.decision]}）：{response.rationale}",
                    f"  - 原审阅者复核 v{cycle.result_version}（{_REASSESSMENT_LABELS[reassessment.decision]}）：{reassessment.rationale}",
                )
            )
        if not cycle.review_memo.findings:
            lines.append(
                f"  - 原审阅者确认：{cycle.reviewer_reassessment_record.comment}"
            )

    for verification in result.delivery_verifications:
        title = next(item.title for item in result.assignments if item.id == verification.assignment_id)
        for check in verification.acceptance_status:
            label = {"passed": "通过", "pending": "未验证", "failed": "不通过"}[check.status]
            lines.append(
                f"- {title} v{verification.artifact_version}｜{check.criterion}：{label}；{check.evidence}"
            )

    lines.extend(("", "## 讨论结论"))
    closed_findings = sum(
        item.decision in {"satisfied", "withdrawn"}
        for cycle in result.review_cycles
        for item in cycle.reviewer_reassessment_record.finding_reassessments
    )
    no_findings_reviews = sum(
        cycle.review_memo.result == "no_findings"
        for cycle in result.review_cycles
    )
    if closed_findings:
        lines.append(f"- {closed_findings} 条审阅意见经回应和复核后关闭。")
    if no_findings_reviews:
        lines.append(
            f"- {no_findings_reviews} 项成果经审阅后未发现需要处理的问题。"
        )
    for arbitration in result.arbitrations:
        arbiter = registry.get(arbitration.arbiter_id)
        for decision in arbitration.finding_decisions:
            if decision.resolved:
                lines.append(
                    f"- {arbiter.name} 对 {decision.finding_id} 的判断："
                    f"{decision.decision} 理由：{decision.reason}"
                )
            else:
                lines.append(
                    f"- {arbiter.name} 未强行判断 {decision.finding_id}："
                    f"{decision.reason}"
                )
    if not closed_findings and not no_findings_reviews and not result.arbitrations:
        lines.append("- 所有审阅均已确认。")

    lines.extend(("", "## 未解决分歧"))
    blocking_checks = [
        check for verification in result.delivery_verifications
        for check in verification.acceptance_status if check.status != "passed"
    ]
    if not result.unresolved_disagreements and not blocking_checks:
        lines.append("无。")
    else:
        for item in result.unresolved_disagreements:
            lines.extend(
                (
                    f"- 分歧事项：{item.topic}",
                    f"  - 作者立场：{item.author_position}",
                    f"  - 审阅者立场：{item.reviewer_position}",
                    f"  - 无法自动解决的原因：{item.arbitration_reason}",
                    f"  - 需要你决定：{item.required_user_input}",
                )
            )
    for check in blocking_checks:
        kind = "检查未通过" if check.status == "failed" else "证据缺口"
        lines.append(f"- {kind}：{check.criterion}；{check.evidence} {check.impact}")

    lines.extend(("", "## 最终成果", ""))
    if result.final_result is None:
        if result.status == "needs_verification":
            lines.append("暂未生成；先补齐或更正上述必要检查，已有阶段成果保留。")
        else:
            lines.append("暂未生成；等待上述事项明确后再统一合并。")
    else:
        lines.append(result.final_result)
    return "\n".join(lines)


def render_feedback_markdown(outcome, router: CapabilityRouter | None = None) -> str:
    """Show the same six sections, with the user's amendment kept separate."""
    result = outcome.result or replace(
        outcome.previous, status="needs_verification", final_artifact_ids=(),
        final_result=None, integrator_id=None,
    )
    text = render_markdown(result, router)
    notes = [f"补充意见：{outcome.feedback.text}", outcome.message]
    for change in outcome.changes:
        notes.append(f"- v{change.before_version} → v{change.after_version}：{change.summary}")
    if outcome.reused_assignment_ids:
        notes.append("未受影响的成果保持原文，没有重新生成。")
    return text.replace("\n## 分工", "\n" + "\n".join(notes) + "\n\n## 分工", 1)
