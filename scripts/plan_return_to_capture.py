#!/usr/bin/env python3
"""Plan an observation-only return to a previously captured wrist pose."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from robot_workflow.legacy_tcp import LegacyApiGateway


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    arguments = parse_args()
    source_plan_path = (
        arguments.capture / "plan.json"
        if arguments.capture.is_dir()
        else arguments.capture
    )
    source = json.loads(source_plan_path.read_text("utf-8"))
    require(source.get("arm") == "left", "best capture is not a left-wrist view")
    target_joints = [float(value) for value in source["current_joints"]]
    require(len(target_joints) == 7, "best capture has invalid target joints")

    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config = json.loads(
        (arguments.robotdata / "config/env_config.json").read_text("utf-8")
    )
    output = arguments.output_root / (
        "return_to_best_capture_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    environment = None
    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        left_status, current_left = environment.arm_left.get_joint_degree()
        right_status, current_right = environment.arm_right.get_joint_degree()
        require(
            left_status == 0 and len(current_left) == 7,
            "failed to read current left-arm joints",
        )
        require(
            right_status == 0 and len(current_right) == 7,
            "failed to read current right-arm joints",
        )
        current_left = [float(value) for value in current_left]
        current_right = [float(value) for value in current_right]
        target_camera_pose = [
            float(value)
            for value in environment.arm_left.camera_kinematics.get_camera_forward(
                target_joints
            )
        ]
        target_end_pose = [
            float(value)
            for value in environment.arm_left.gripper_kinematics.get_arm_end_forward(
                target_joints
            )
        ]
        simulation = LegacyApiGateway(
            arguments.api_host, timeout_s=120.0
        ).simulate(
            target_pose=target_end_pose,
            target_joints_deg=target_joints,
            arm="left",
            left_joints_deg=current_left,
            right_joints_deg=current_right,
            num_waypoints=25,
            recording=True,
        )
        require(bool(simulation["success"]), "no collision-free return to best capture")
        preview = output / "return_preview.mp4"
        preview.write_bytes(base64.b64decode(simulation["video"]))
        result: dict[str, Any] = {
            "ok": True,
            "planning_only": True,
            "real_robot_command_sent": False,
            "gripper_command_sent": False,
            "status": "awaiting_human_review",
            "arm": "left",
            "current_joints": current_left,
            "left_joints": current_left,
            "right_joints": current_right,
            "source_capture": str(source_plan_path.parent),
            "attempts": [
                {
                    "kind": "return_to_best_capture",
                    "target_joints": target_joints,
                    "target_camera_pose": target_camera_pose,
                    "target_end_pose": target_end_pose,
                    "simulation_solved": True,
                    "waypoints": simulation["path"],
                }
            ],
            "preview_video": str(preview),
        }
        plan_path = output / "plan.json"
        plan_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({"output": str(output), "plan": str(plan_path), **result}, ensure_ascii=False))
        return 0
    finally:
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
