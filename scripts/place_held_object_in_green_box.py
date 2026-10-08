#!/usr/bin/env python3
"""Continue from a closed, lifted left gripper and place into the green box."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from robot_workflow.hardware_video import HardwareVideoRecorder
from robot_workflow.legacy_tcp import LegacyApiGateway
from run_pistol_grasp_workflow import path_sha256, plan_box_transfer, require


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("grasp_api_tests"))
    parser.add_argument("--record-simulation-previews", action="store_true")
    parser.add_argument("--execute-confirmed-held-inert-prop", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = arguments.output_root / f"green_box_place_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config = json.loads(
        (arguments.robotdata / "config/env_config.json").read_text("utf-8")
    )
    environment = None
    recorder = None
    result = {
        "ok": False,
        "planning_only": not arguments.execute_confirmed_held_inert_prop,
        "real_robot_motion_sent": False,
        "gripper_release_sent": False,
        "output": str(output),
    }
    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        left_status, left = environment.arm_left.get_joint_degree()
        right_status, right = environment.arm_right.get_joint_degree()
        require(left_status == 0 and right_status == 0, "cannot read current joints")
        left = [float(value) for value in left]
        right = [float(value) for value in right]
        gripper_return, gripper_state = environment.arm_left.robot.rm_get_gripper_state()
        require(gripper_return == 0, "cannot read current left gripper")
        require(
            int(gripper_state.get("actpos", 0)) >= 100,
            "left gripper does not appear to contain the held prop",
        )
        result["initial_left_joints"] = left
        result["right_joints"] = right
        result["initial_gripper_state"] = gripper_state

        plan = plan_box_transfer(
            LegacyApiGateway("127.0.0.1", timeout_s=120.0),
            environment.arm_left.gripper_kinematics,
            left,
            right,
            (
                output
                if arguments.execute_confirmed_held_inert_prop
                or arguments.record_simulation_previews
                else None
            ),
        )
        result["plan"] = plan
        (output / "plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not arguments.execute_confirmed_held_inert_prop:
            result["ok"] = True
            result["status"] = "planned_only"
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0

        recorder = HardwareVideoRecorder(
            {
                "left": environment.arm_left.camera,
                "right": environment.arm_right.camera,
                "head": environment.head_camera,
            },
            output,
            fps=10.0,
        )
        recorder.start()
        recorder.wait_for_required(("left", "right"), timeout_s=10.0)

        transfer_records = []
        for segment in plan["pre_release_segments"]:
            phase = str(segment["phase"])
            path = segment["path"]
            require(path_sha256(path) == segment["path_sha256"], f"changed {phase} path")
            status, current = environment.arm_left.get_joint_degree()
            start_error = float(
                np.max(np.abs(np.asarray(current) - np.asarray(path[0])))
            )
            require(status == 0 and start_error <= 0.5, f"stale {phase} start")
            recorder.mark(f"green_box_{phase}")
            for index, waypoint in enumerate(path[1:], start=1):
                code = environment.arm_left.robot.rm_movej(
                    waypoint, v=2, r=0, connect=0, block=1
                )
                result["real_robot_motion_sent"] = True
                query, actual = environment.arm_left.get_joint_degree()
                error = float(
                    np.max(np.abs(np.asarray(actual) - np.asarray(waypoint)))
                )
                transfer_records.append(
                    {"phase": phase, "index": index, "return": code, "error_deg": error}
                )
                result["transfer_records"] = transfer_records
                require(code == 0 and query == 0 and error <= 1.0, f"failed {phase} waypoint {index}")

        recorder.mark("green_box_release")
        release = environment.arm_left.robot.rm_set_gripper_release(
            200, block=True, timeout=5
        )
        result["gripper_release_sent"] = True
        result["gripper_release_return"] = release
        require(release == 0, "green-box release failed")

        retreat_records = []
        for segment in plan["retreat_segments"]:
            phase = str(segment["phase"])
            path = segment["path"]
            require(path_sha256(path) == segment["path_sha256"], f"changed {phase} path")
            status, current = environment.arm_left.get_joint_degree()
            start_error = float(
                np.max(np.abs(np.asarray(current) - np.asarray(path[0])))
            )
            require(status == 0 and start_error <= 0.5, f"stale {phase} start")
            recorder.mark(f"green_box_{phase}")
            for index, waypoint in enumerate(path[1:], start=1):
                code = environment.arm_left.robot.rm_movej(
                    waypoint, v=2, r=0, connect=0, block=1
                )
                result["real_robot_motion_sent"] = True
                query, actual = environment.arm_left.get_joint_degree()
                error = float(
                    np.max(np.abs(np.asarray(actual) - np.asarray(waypoint)))
                )
                retreat_records.append(
                    {"phase": phase, "index": index, "return": code, "error_deg": error}
                )
                result["retreat_records"] = retreat_records
                require(code == 0 and query == 0 and error <= 1.0, f"failed {phase} waypoint {index}")

        result["hardware_video"] = recorder.stop()
        result["ok"] = True
        result["status"] = "placed_in_green_box_and_retreated_100mm"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        if environment is not None and result["real_robot_motion_sent"]:
            try:
                result["emergency_stop_return"] = environment.arm_left.robot.rm_set_arm_stop()
            except Exception as stop_error:
                result["emergency_stop_error"] = repr(stop_error)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1
    finally:
        output.mkdir(parents=True, exist_ok=True)
        if recorder is not None:
            result["hardware_video"] = recorder.stop()
        (output / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
