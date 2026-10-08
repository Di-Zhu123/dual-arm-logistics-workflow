"""Run the complete simulated workflow; never imports or contacts robot hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from robot_workflow.domain import PendingReview, TaskRequest  # noqa: E402
from robot_workflow.offline import (  # noqa: E402
    OfflineGraspService,
    OfflineMotionPlanner,
    OfflinePerception,
    OfflineVerifier,
    SimulatedExecutor,
    SimulationWorld,
    SyntheticThreeCameraSceneProvider,
)
from robot_workflow.workflow import PickPlaceWorkflow, WorkflowOutcome  # noqa: E402


def review_json(review: PendingReview) -> str:
    return json.dumps(
        {
            "phase": review.phase.value,
            "plan_id": review.plan_id,
            "plan_digest": review.plan_digest,
            "preview_uri": review.preview.uri,
            "preview_sha256": review.preview.sha256,
            "expires_at_s": review.expires_at_s,
        },
        indent=2,
        ensure_ascii=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="red block")
    parser.add_argument("--receptacle", default="blue box")
    parser.add_argument(
        "--approve-simulation",
        action="store_true",
        help="exercise both approval gates in the synthetic simulator only",
    )
    arguments = parser.parse_args()

    world = SimulationWorld()
    executor = SimulatedExecutor(world)
    workflow = PickPlaceWorkflow(
        scene_provider=SyntheticThreeCameraSceneProvider(world),
        perception=OfflinePerception(),
        grasp_service=OfflineGraspService(),
        planner=OfflineMotionPlanner(),
        executor=executor,
        verifier=OfflineVerifier(world),
    )
    review = workflow.start(TaskRequest(arguments.target, arguments.receptacle))
    print(review_json(review))
    if not arguments.approve_simulation:
        print("Stopped safely before simulated motion; pass --approve-simulation for offline E2E.")
        return 0

    while isinstance(review, PendingReview):
        token = workflow.approve_current(
            presented_digest=review.plan_digest,
            reviewer="explicit-offline-demo",
        )
        result = workflow.execute_current(token)
        if isinstance(result, PendingReview):
            review = result
            print(review_json(review))
            continue
        assert isinstance(result, WorkflowOutcome)
        print(
            json.dumps(
                {
                    "state": result.state.value,
                    "task_id": result.task_id,
                    "executed_plan_count": len(executor.history),
                    "placed_object_id": world.placed_object_id,
                },
                indent=2,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
