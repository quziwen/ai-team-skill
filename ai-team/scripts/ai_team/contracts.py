"""Small, auditable data contracts for one AI Team run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RunStatus = Literal["completed", "needs_user_decision"]
InitialVerdict = Literal["no_findings", "changes_requested", "disputed"]
CycleStatus = Literal["confirmed", "disputed", "arbitrated", "unresolved"]
ReviewMemoResult = Literal["findings", "no_findings"]
ReviewSeverity = Literal["critical", "major", "minor"]
EvidenceType = Literal["local", "external", "inference"]
ResponseDecision = Literal["accept", "reject", "clarify"]
ReassessmentDecision = Literal[
    "satisfied", "withdrawn", "contested", "owner_required"
]


@dataclass(frozen=True)
class AgentProfile:
    id: str
    name: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class Assignment:
    id: str
    title: str
    goal: str
    required_capabilities: tuple[str, ...]
    agent_id: str
    matched_capabilities: tuple[str, ...]


@dataclass(frozen=True)
class Artifact:
    id: str
    assignment_id: str
    author_id: str
    version: int
    content: str


@dataclass(frozen=True)
class ReviewPackage:
    """Frozen review input for one artifact version."""

    id: str
    assignment_id: str
    artifact_id: str
    artifact_version: int
    artifact_sha256: str
    author_id: str
    reviewer_id: str
    scope: str
    acceptance_criteria: tuple[str, ...]


@dataclass(frozen=True)
class ReviewFinding:
    id: str
    severity: ReviewSeverity
    problem: str
    evidence_type: EvidenceType
    evidence: str
    suggestion: str
    disagreement_topic: str | None = None
    author_position: str | None = None
    reviewer_position: str | None = None


@dataclass(frozen=True)
class ReviewMemo:
    """Immutable initial review; later replies never overwrite it."""

    id: str
    review_package_id: str
    reviewer_id: str
    reviewed_artifact_id: str
    reviewed_version: int
    reviewed_sha256: str
    result: ReviewMemoResult
    verdict: InitialVerdict
    summary: str
    findings: tuple[ReviewFinding, ...]


@dataclass(frozen=True)
class FindingResponse:
    finding_id: str
    decision: ResponseDecision
    rationale: str
    evidence: tuple[str, ...]
    change_refs: tuple[str, ...]


@dataclass(frozen=True)
class StructuredAuthorResponse:
    id: str
    review_memo_id: str
    author_id: str
    message: str
    revised_content: str | None
    finding_responses: tuple[FindingResponse, ...]


@dataclass(frozen=True)
class FindingReassessment:
    finding_id: str
    reviewer_id: str
    decision: ReassessmentDecision
    rationale: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ReviewerReassessment:
    id: str
    review_memo_id: str
    author_response_id: str
    reviewer_id: str
    reviewed_artifact_id: str
    reviewed_version: int
    reviewed_sha256: str
    resolved: bool
    comment: str
    finding_reassessments: tuple[FindingReassessment, ...]


@dataclass(frozen=True)
class ReviewFeedback:
    verdict: InitialVerdict
    comment: str
    disagreement_topic: str | None = None
    author_position: str | None = None
    reviewer_position: str | None = None


@dataclass(frozen=True)
class AuthorResponse:
    message: str
    revised_content: str | None
    decision: ResponseDecision | None = None


@dataclass(frozen=True)
class Confirmation:
    resolved: bool
    comment: str


@dataclass(frozen=True)
class ReviewCycle:
    id: str
    assignment_id: str
    author_id: str
    reviewer_id: str
    review_package: ReviewPackage
    review_memo: ReviewMemo
    structured_author_response: StructuredAuthorResponse
    reviewer_reassessment_record: ReviewerReassessment
    reviewed_artifact_id: str
    reviewed_version: int
    initial_verdict: InitialVerdict
    review_comment: str
    author_response: str
    result_artifact_id: str
    result_version: int
    reviewer_confirmation: str
    status: CycleStatus
    disagreement_topic: str | None = None
    author_position: str | None = None
    reviewer_position: str | None = None


@dataclass(frozen=True)
class FindingArbitrationDecision:
    finding_id: str
    resolved: bool
    decision: str | None
    reason: str
    required_user_input: str | None = None


@dataclass(frozen=True)
class ArbitrationDecision:
    review_cycle_id: str
    arbiter_id: str
    finding_decisions: tuple[FindingArbitrationDecision, ...]

    @property
    def finding_ids(self) -> tuple[str, ...]:
        return tuple(item.finding_id for item in self.finding_decisions)

    @property
    def resolved(self) -> bool:
        return bool(self.finding_decisions) and all(
            item.resolved for item in self.finding_decisions
        )

    @property
    def decision(self) -> str | None:
        decisions = [
            item.decision.strip()
            for item in self.finding_decisions
            if item.resolved and item.decision and item.decision.strip()
        ]
        return "；".join(decisions) if decisions else None

    @property
    def reason(self) -> str:
        return "；".join(
            f"{item.finding_id}：{item.reason.strip()}"
            for item in self.finding_decisions
            if item.reason.strip()
        )

    @property
    def required_user_input(self) -> str | None:
        inputs = [
            item.required_user_input.strip()
            for item in self.finding_decisions
            if not item.resolved
            and item.required_user_input
            and item.required_user_input.strip()
        ]
        return "；".join(inputs) if inputs else None


@dataclass(frozen=True)
class UnresolvedDisagreement:
    topic: str
    author_position: str
    reviewer_position: str
    arbitration_reason: str
    required_user_input: str


@dataclass(frozen=True)
class IntegrationResult:
    content: str
    artifact_ids: tuple[str, ...]
    arbitration_review_cycle_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    task: str
    status: RunStatus
    assignments: tuple[Assignment, ...]
    artifacts: tuple[Artifact, ...]
    review_cycles: tuple[ReviewCycle, ...]
    arbitrations: tuple[ArbitrationDecision, ...]
    unresolved_disagreements: tuple[UnresolvedDisagreement, ...]
    final_artifact_ids: tuple[str, ...]
    final_result: str | None
    integrator_id: str | None
    events: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a serializable form for tests or future adapters."""

        return asdict(self)
