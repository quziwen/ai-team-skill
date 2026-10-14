"""A fixed, offline activity fixture; never presented as real Agent work."""
import argparse
import json
import sys

from ai_team import AITeam, FeedbackRequest, FeedbackRevision, FeedbackTarget
from ai_team.agents import MockBackend
from ai_team.ccrl import artifact_sha256
from ai_team.contracts import ReviewFeedback
from ai_team.presenter import render_feedback_markdown


ORIGINAL = "开场：20分钟\n交流：100分钟\n人员分工：主持甲，记录乙。\n20 + 100 = 120"
OTHER = "物品由丙准备，人员名单不变。"
TEXT = "把开场改为“开场：10分钟”，人员分工不变。"


class FixtureBackend(MockBackend):
    def produce(self, agent, assignment, task):
        return ORIGINAL if assignment.id == "work-1" else OTHER

    def review(self, reviewer, artifact, assignment, task):
        return ReviewFeedback("no_findings", "仅为本地预设审阅结果。")


def demonstrate(case="normal"):
    backend = FixtureBackend("normal")
    team = AITeam(backend)

    def check(artifact, package):
        checks = [{"criterion": "加法正确", "method": "sum", "expression": artifact.content.splitlines()[-1]}]
        if "开场符合补充意见" in package.acceptance_criteria:
            checks.append({"criterion": "开场符合补充意见", "method": "source_quote", "quote": "开场：10分钟",
                           "source_ref": "local-feedback-fixture", "source_text": TEXT})
        return checks

    previous = team.run("制定不超过120分钟的活动方案。",
                        delivery_requirements={"work-1": ("加法正确",), "work-2": ()}, delivery_checker=check)
    original = next(a for a in previous.artifacts if a.assignment_id == "work-1")
    feedback = FeedbackRequest("local-feedback-1", previous.task, TEXT, (FeedbackTarget(
        original.assignment_id, original.id, original.version, artifact_sha256(original),
        ("开场：20分钟", "20 + 100 = 120"), ("开场符合补充意见",),
    ),))
    calls = []

    def revise(author, artifact, request, target):
        calls.append(author.id)
        total = "111" if case == "wrong-sum" else "110"
        content = artifact.content.replace("开场：20分钟", "开场：10分钟").replace("20 + 100 = 120", "10 + 100 = " + total)
        return FeedbackRevision(author.id, artifact.id, artifact.version, artifact_sha256(artifact), content,
                                "开场缩短10分钟，人员及物品安排保留。")

    outcome = team.continue_feedback(previous, feedback, revision_provider=revise, delivery_checker=check)
    if case == "repeat":
        repeated = team.continue_feedback(previous, feedback, revision_provider=revise, delivery_checker=check)
        assert repeated is outcome
    return outcome, {
        "status": outcome.status, "author_revision_calls": len(calls),
        "total_merge_calls": backend.integrate_calls, "original_task_unchanged": outcome.result.task == previous.task,
        "unchanged_other_artifact": next(a for a in outcome.result.artifacts if a.assignment_id == "work-2").content == OTHER,
        "new_version": outcome.changes[0].after_version, "real_agents_started": False,
        "acceptance_boundary": "deterministic local feedback fixture only",
    }


def main():
    parser = argparse.ArgumentParser(description="本地 Mock 修改接续演示")
    parser.add_argument("--case", choices=("normal", "wrong-sum", "repeat"), default="normal")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    print("这是本地 Mock 修改接续示例，不调用真实模型，不是实际任务成果。", file=sys.stderr)
    outcome, receipt = demonstrate(args.case)
    print(json.dumps(receipt, ensure_ascii=False) if args.json else render_feedback_markdown(outcome))
    return 0 if outcome.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
