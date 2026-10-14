"""The bounded, single-process AI Team workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, replace

from .agents import AgentBackend, CapabilityRouter, MockBackend
from .ccrl import (
    CCRLReviewCapability,
    ReviewCapability,
    validate_author_response,
    validate_change_refs,
    validate_reassessment,
    validate_review_memo,
    validate_review_package,
    artifact_sha256,
)
from .contracts import (
    ArbitrationDecision,
    Artifact,
    Assignment,
    DeliveryCheck,
    DeliveryVerification,
    IntegrationResult,
    ReviewCycle,
    ReviewPackage,
    RunResult,
    UnresolvedDisagreement,
)


class AITeam:
    def __init__(
        self,
        backend: AgentBackend | None = None,
        router: CapabilityRouter | None = None,
        review_capability: ReviewCapability | None = None,
    ) -> None:
        self.backend = backend or MockBackend()
        self.router = router or CapabilityRouter()
        self.review_capability = review_capability or CCRLReviewCapability()
        self._current_result = None
        self._current_requirements = None
        self._feedback_cache = {}
        self._feedback_busy = False

    def run(
        self,
        task: str,
        *,
        delivery_requirements: Mapping[str, tuple[str, ...]] | None = None,
        delivery_checker: Callable[[Artifact, ReviewPackage], list[dict]] | None = None,
    ) -> RunResult:
        if not task.strip():
            raise ValueError("任务不能为空")

        if self._feedback_busy:
            raise RuntimeError("修改接续期间不能另开任务")
        self._current_result = None
        self._feedback_cache.clear()
        events: list[str] = ["received"]
        assignments = self.router.plan(task)
        events.append("planned")
        requirements = _freeze_delivery_requirements(
            delivery_requirements, assignments, delivery_checker
        )

        artifacts: list[Artifact] = []
        latest_by_assignment: dict[str, Artifact] = {}
        for assignment in assignments:
            author = self.router.get(assignment.agent_id)
            content = self.backend.produce(author, assignment, task)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Agent 生成的成果不能为空")
            artifact = Artifact(
                id=f"{assignment.id}-v1",
                assignment_id=assignment.id,
                author_id=author.id,
                version=1,
                content=content,
            )
            artifacts.append(artifact)
            latest_by_assignment[assignment.id] = artifact
        events.append("produced")

        cycles: list[ReviewCycle] = []
        for index, assignment in enumerate(assignments):
            reviewer_assignment = assignments[(index + 1) % len(assignments)]
            cycle, revised, updates = self._review_artifact(
                task, assignment, latest_by_assignment[assignment.id],
                reviewer_assignment.agent_id, events,
            )
            artifacts.extend(updates)
            latest_by_assignment[assignment.id] = revised
            cycles.append(cycle)
        return self._finish(
            task, assignments, artifacts, latest_by_assignment, cycles,
            requirements, delivery_checker, events,
        )

    def continue_feedback(
        self, previous, feedback, *, revision_provider, delivery_checker=None,
    ):
        """Continue one completed local run; the host prepares internal scope."""
        from .feedback import continue_feedback
        return continue_feedback(
            self, previous, feedback, revision_provider=revision_provider,
            delivery_checker=delivery_checker,
        )

    def _review_artifact(
        self, task, assignment, artifact, reviewer_id, events, *,
        cycle_id=None, review_notes=None, revision_guard=None,
    ):
        author = self.router.get(assignment.agent_id)
        reviewer = self.router.get(reviewer_id)
        updates = []
        # Extra notes stay separate from the original user task and assignment.
        if review_notes is not None:
            assignment = replace(
                assignment, goal=assignment.goal + "\n补充修改要求：" + review_notes
            )
        if author.id == reviewer.id:
            raise RuntimeError("作者不能审阅自己的成果")

        package = self.review_capability.prepare(
            author, reviewer, artifact, assignment
        )
        validate_review_package(
            package, author, reviewer, artifact, assignment
        )
        memo = self.review_capability.review(
            self.backend,
            reviewer,
            artifact,
            assignment,
            task,
            package,
        )
        validate_review_memo(package, memo)
        events.append(f"reviewed:{assignment.id}:v{artifact.version}")
        response = self.review_capability.respond(
            self.backend, author, artifact, memo, task
        )
        validate_author_response(memo, response, author.id)
        events.append(f"responded:{assignment.id}")

        result_artifact = artifact
        if response.revised_content is not None:
            if not response.revised_content.strip():
                raise RuntimeError("作者修订内容不能为空")
            result_artifact = Artifact(
                id=f"{assignment.id}-v{artifact.version + 1}",
                assignment_id=artifact.assignment_id,
                author_id=artifact.author_id,
                version=artifact.version + 1,
                content=response.revised_content,
            )
            updates.append(result_artifact)

        if revision_guard is not None:
            revision_guard(result_artifact)
        validate_change_refs(response, artifact, result_artifact)

        reassessment = self.review_capability.reassess(
            self.backend,
            reviewer,
            memo,
            response,
            result_artifact,
        )
        validate_reassessment(
            memo, response, reassessment, reviewer.id, result_artifact
        )
        if (
            any(
                item.decision == "accept"
                for item in response.finding_responses
            )
            and reassessment.resolved
            and (
                response.revised_content is None
                or response.revised_content.strip() == artifact.content.strip()
            )
        ):
            raise RuntimeError("审阅要求修改时，必须基于实质修订后的新版本确认")
        events.append(
            f"rechecked:{assignment.id}:v{result_artifact.version}"
        )
        status = "confirmed" if reassessment.resolved else "disputed"
        first_finding = memo.findings[0] if memo.findings else None
        cycle = ReviewCycle(
                id=cycle_id or f"review-{assignment.id}",
                assignment_id=assignment.id,
                author_id=author.id,
                reviewer_id=reviewer.id,
                review_package=package,
                review_memo=memo,
                structured_author_response=response,
                reviewer_reassessment_record=reassessment,
                reviewed_artifact_id=artifact.id,
                reviewed_version=artifact.version,
                initial_verdict=memo.verdict,
                review_comment=memo.summary,
                author_response=response.message,
                result_artifact_id=result_artifact.id,
                result_version=result_artifact.version,
                reviewer_confirmation=reassessment.comment,
                status=status,
                disagreement_topic=(
                    first_finding.disagreement_topic if first_finding else None
                ),
                author_position=(
                    first_finding.author_position if first_finding else None
                ),
                reviewer_position=(
                    first_finding.reviewer_position if first_finding else None
                ),
            )
        return cycle, result_artifact, tuple(updates)

    def _finish(
        self, task, assignments, artifacts, latest_by_assignment, cycles,
        requirements, delivery_checker, events, existing_arbitrations=(),
    ):
        result = self._finish_result(
            task, assignments, artifacts, latest_by_assignment, cycles,
            requirements, delivery_checker, events, existing_arbitrations,
        )
        self._current_result = result
        self._current_requirements = None if requirements is None else dict(requirements)
        return result

    def _finish_result(
        self, task, assignments, artifacts, latest_by_assignment, cycles,
        requirements, delivery_checker, events, existing_arbitrations=(),
    ):
        arbitrations: list[ArbitrationDecision] = list(existing_arbitrations)
        unresolved: list[UnresolvedDisagreement] = []
        final_cycles: list[ReviewCycle] = []
        for cycle in cycles:
            if cycle.status != "disputed":
                final_cycles.append(cycle)
                continue

            arbiter = self.router.select(
                ("arbitration", "decision"),
                exclude={cycle.author_id, cycle.reviewer_id},
            )
            if arbiter.id in {cycle.author_id, cycle.reviewer_id}:
                raise RuntimeError("第三方判断 Agent 必须独立于争议双方")
            unresolved_finding_ids = tuple(
                item.finding_id
                for item in cycle.reviewer_reassessment_record.finding_reassessments
                if item.decision in {"contested", "owner_required"}
            )
            if not unresolved_finding_ids:
                raise RuntimeError("存在分歧的审阅必须列明未解决的具体意见")
            decision = self.backend.arbitrate(arbiter, cycle, task)
            if decision.arbiter_id != arbiter.id:
                raise RuntimeError("第三方判断结果的 Agent 身份与实际分配不一致")
            if decision.review_cycle_id != cycle.id:
                raise RuntimeError("第三方判断结果引用了错误的审阅事项")
            if len(decision.finding_ids) != len(set(decision.finding_ids)):
                raise RuntimeError("第三方判断包含重复的审阅意见编号")
            if set(decision.finding_ids) != set(unresolved_finding_ids):
                raise RuntimeError("第三方判断没有逐条覆盖全部未解决意见")
            for item in decision.finding_decisions:
                if not isinstance(item.resolved, bool):
                    raise RuntimeError(
                        f"{item.finding_id} 的第三方判断必须明确是否已解决"
                    )
                if not item.reason.strip():
                    raise RuntimeError(
                        f"第三方判断必须说明 {item.finding_id} 的理由"
                    )
                if item.resolved and (
                    not item.decision or not item.decision.strip()
                ):
                    raise RuntimeError(
                        f"第三方解决 {item.finding_id} 时必须给出明确结论"
                    )
                if item.resolved and (
                    item.required_user_input
                    and item.required_user_input.strip()
                ):
                    raise RuntimeError(
                        f"{item.finding_id} 已解决时不能再要求用户决定"
                    )
                if not item.resolved and item.decision and item.decision.strip():
                    raise RuntimeError(
                        f"{item.finding_id} 未解决时不能声称已有最终结论"
                    )
            arbitrations.append(decision)
            events.append(f"arbitrated:{cycle.id}")
            if decision.resolved:
                final_cycles.append(replace(cycle, status="arbitrated"))
            else:
                final_cycles.append(replace(cycle, status="unresolved"))
                findings = {
                    item.id: item for item in cycle.review_memo.findings
                }
                responses = {
                    item.finding_id: item
                    for item in cycle.structured_author_response.finding_responses
                }
                reassessments = {
                    item.finding_id: item
                    for item in cycle.reviewer_reassessment_record.finding_reassessments
                }
                for item in decision.finding_decisions:
                    if item.resolved:
                        continue
                    finding = findings[item.finding_id]
                    unresolved.append(
                        UnresolvedDisagreement(
                            topic=(finding.disagreement_topic or "").strip()
                            or finding.problem.strip()
                            or "未命名分歧",
                            author_position=(finding.author_position or "").strip()
                            or responses[item.finding_id].rationale.strip()
                            or "作者未说明",
                            reviewer_position=(
                                finding.reviewer_position or ""
                            ).strip()
                            or reassessments[item.finding_id].rationale.strip()
                            or "审阅者未说明",
                            arbitration_reason=item.reason.strip(),
                            required_user_input=(
                                item.required_user_input or ""
                            ).strip()
                            or "请提供判断所需信息。",
                        )
                    )

        final_artifacts = tuple(
            latest_by_assignment[assignment.id] for assignment in assignments
        )
        if unresolved:
            events.append("needs_user_decision")
            return RunResult(
                task=task,
                status="needs_user_decision",
                assignments=assignments,
                artifacts=tuple(artifacts),
                review_cycles=tuple(final_cycles),
                arbitrations=tuple(arbitrations),
                unresolved_disagreements=tuple(unresolved),
                final_artifact_ids=tuple(),
                final_result=None,
                integrator_id=None,
                events=tuple(events),
            )

        verifications = _verify_deliveries(
            final_artifacts, tuple(final_cycles), requirements, delivery_checker
        )
        if verifications:
            events.append("delivery_checked")
        if any(not item.checks_passed for item in verifications):
            events.append("needs_verification")
            return RunResult(
                task=task,
                status="needs_verification",
                assignments=assignments,
                artifacts=tuple(artifacts),
                review_cycles=tuple(final_cycles),
                arbitrations=tuple(arbitrations),
                unresolved_disagreements=tuple(),
                final_artifact_ids=tuple(),
                final_result=None,
                integrator_id=None,
                events=tuple(events),
                delivery_verifications=verifications,
            )

        integrator = self.router.select(("synthesis",), exclude=set())
        integration = self.backend.integrate(
            integrator,
            task,
            assignments,
            final_artifacts,
            tuple(arbitrations),
        )
        if not isinstance(integration, IntegrationResult):
            raise RuntimeError("合并 Agent 必须返回可核对的合并记录")
        if not integration.content.strip():
            raise RuntimeError("统一合并结果不能为空")
        expected_artifact_ids = tuple(item.id for item in final_artifacts)
        if (
            len(integration.artifact_ids) != len(set(integration.artifact_ids))
            or set(integration.artifact_ids) != set(expected_artifact_ids)
        ):
            raise RuntimeError("统一合并结果没有纳入全部已确认成果")
        compact_integration = " ".join(integration.content.split())
        if any(
            " ".join(item.content.split()) not in compact_integration
            for item in final_artifacts
        ):
            raise RuntimeError("统一合并正文遗漏了已确认成果")
        expected_arbitrations = tuple(
            item.review_cycle_id for item in arbitrations if item.resolved
        )
        if (
            len(integration.arbitration_review_cycle_ids)
            != len(set(integration.arbitration_review_cycle_ids))
            or set(integration.arbitration_review_cycle_ids)
            != set(expected_arbitrations)
        ):
            raise RuntimeError("统一合并结果没有纳入全部第三方判断")
        if any(
            finding.decision
            and finding.decision.strip() not in integration.content
            for arbitration in arbitrations
            if arbitration.resolved
            for finding in arbitration.finding_decisions
        ):
            raise RuntimeError("统一合并正文遗漏了第三方判断结论")
        events.extend(("merged", "completed"))
        return RunResult(
            task=task,
            status="completed",
            assignments=assignments,
            artifacts=tuple(artifacts),
            review_cycles=tuple(final_cycles),
            arbitrations=tuple(arbitrations),
            unresolved_disagreements=tuple(),
            final_artifact_ids=tuple(item.id for item in final_artifacts),
            final_result=integration.content,
            integrator_id=integrator.id,
            events=tuple(events),
            delivery_verifications=verifications,
        )



def _freeze_delivery_requirements(requirements, assignments, checker):
    if requirements is None:
        if checker is not None:
            raise ValueError("检查方法必须同时提供本次必要要求")
        return None
    if not isinstance(requirements, Mapping) or set(requirements) != {
        item.id for item in assignments
    }:
        raise ValueError("必须逐项说明每个子任务的必要检查要求")
    if checker is not None and not callable(checker):
        raise ValueError("成果检查方法必须可调用")
    frozen = {}
    for assignment_id, criteria in requirements.items():
        if (
            type(criteria) is not tuple or len(criteria) > 64
            or any(type(item) is not str or not item.strip() or "\0" in item for item in criteria)
            or len(set(criteria)) != len(criteria)
        ):
            raise ValueError("必要要求必须是无重复的非空文字；无客观检查时显式使用空元组")
        frozen[assignment_id] = criteria
    return frozen


def _verify_deliveries(artifacts, cycles, requirements, checker):
    if requirements is None:
        return tuple()
    # verification reuses the Phase 2 AcceptanceCheck type; import after modules
    # have loaded to preserve the existing project -> workflow dependency.
    from .verification import VerificationInputError, verify_delivery

    cycles_by_assignment = {item.assignment_id: item for item in cycles}
    results = []
    for artifact in artifacts:
        criteria = requirements[artifact.assignment_id]
        if not criteria:
            continue
        original = cycles_by_assignment[artifact.assignment_id].review_package
        package = replace(
            original,
            id=f"delivery-{artifact.id}",
            artifact_id=artifact.id,
            artifact_version=artifact.version,
            artifact_sha256=artifact_sha256(artifact),
            acceptance_criteria=criteria,
        )
        try:
            checks = [] if checker is None else checker(artifact, package)
            receipt = verify_delivery({
                "artifact": asdict(artifact),
                "review_package": asdict(package),
                "checks": checks,
            })
            checked = tuple(DeliveryCheck(**item) for item in receipt["acceptance_status"])
        except (VerificationInputError, OSError, ValueError, TypeError):
            # Do not echo callback exceptions, source bodies or stale receipts.
            checked = tuple(DeliveryCheck(
                criterion, "pending", "检查输入或执行不合要求，尚未取得有效检查结果。",
                "unverified", "未取得有效检查依据", None,
                "核对检查方法、依据和当前成果版本后再继续；不能宣布全部完成。",
            ) for criterion in criteria)
        results.append(DeliveryVerification(
            artifact.assignment_id, artifact.id, artifact.version,
            package.artifact_sha256, package.id, checked,
        ))
    return tuple(results)


def run(task: str) -> RunResult:
    """Run one task with safe local defaults."""

    return AITeam().run(task)
