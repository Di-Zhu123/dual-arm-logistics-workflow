#!/usr/bin/env python3
"""Resume a magazine lift-and-place after a post-close visual-check failure."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from robot_workflow.hardware_video import HardwareVideoRecorder
from run_pistol_grasp_workflow import (
    path_sha256,
    require,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_result", type=Path)
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--execute-confirmed-held-inert-prop", action="store_true")
    return parser.parse_args()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_magazine_hold(gripper_state: dict) -> dict:
    """Use the physical non-empty closure signal for a regular-prism magazine."""
    actpos = float(gripper_state.get("actpos", -1.0))
    current_force = float(gripper_state.get("current_force", 0.0))
    return {
        "actpos": actpos,
        "current_force": current_force,
        "minimum_nonempty_actpos": 100.0,
        # The 4C2 controller intermittently reports -2 for current_force after
        # a successful lift while the non-empty aperture remains stable.  A
        # clearly non-zero aperture is the reliable retained-object signal.
        "passed": actpos >= 100.0,
    }


def main() -> int:
    arguments = parse_args()
    source_path = arguments.source_result.resolve()
    source = json.loads(source_path.read_text("utf-8"))
    output = source_path.parent / f"resume_magazine_transfer_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    result: dict = {
        "ok": False,
        "source_result": str(source_path),
        "output": str(output),
        "real_robot_motion_sent": False,
        "gripper_release_sent": False,
        "lift_completed": False,
        "box_transfer_completed": False,
        "box_retreat_completed": False,
    }
    environment = None
    recorder = None
    try:
        require(arguments.execute_confirmed_held_inert_prop, "execution flag is required")
        require(source.get("whole_object_grasp") is True, "source is not a whole-object magazine run")
        require(source.get("gripper_close_sent") is True, "source did not close the gripper")
        already_lifted = bool(source.get("lift_completed"))
        require(
            source.get("lift_sent") is False or already_lifted,
            "source started but did not complete its lift",
        )
        require(source.get("box_transfer_sent") is False, "source already started box transfer")
        require(source.get("box_release_sent") is False, "source already released the object")

        lift_plan = source["post_grasp_lift_plan"]
        box_plan = source["box_transfer_plan"]
        lift_path = lift_plan["path"]
        require(path_sha256(lift_path) == lift_plan["path_sha256"], "changed lift path")
        require(
            path_sha256(box_plan["pre_release_path"]) == box_plan["pre_release_path_sha256"],
            "changed box-transfer path",
        )
        require(
            path_sha256(box_plan["retreat_path"]) == box_plan["retreat_path_sha256"],
            "changed retreat path",
        )

        validation_path = Path(source["hardware_path_source"])
        validation = json.loads(validation_path.read_text("utf-8"))
        selected_rank = int(source["selected_rank"])
        selected = next(
            candidate for candidate in validation["candidates"]
            if int(candidate["rank"]) == selected_rank
        )
        expected_width_m = float(selected["cross_section_width_mm"]) / 1000.0
        result["selected_rank"] = selected_rank
        result["expected_cross_section_width_mm"] = expected_width_m * 1000.0

        sys.path.insert(0, str(arguments.robotdata))
        from utils.arm_environment import DualArmEnvironment  # type: ignore

        config = json.loads((arguments.robotdata / "config/env_config.json").read_text("utf-8"))
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        arm = environment.arm_left
        status, current = arm.get_joint_degree()
        gripper_return, gripper_state = arm.robot.rm_get_gripper_state()
        result["initial_left_joints"] = [float(value) for value in current]
        result["initial_gripper_state"] = gripper_state
        expected_start = lift_path[-1] if already_lifted else lift_path[0]
        start_error = float(
            np.max(np.abs(np.asarray(current) - np.asarray(expected_start)))
        )
        result["lift_start_error_deg"] = start_error
        require(status == 0 and start_error <= 0.75, f"stale lift start: {start_error:.3f} deg")
        require(gripper_return == 0, "cannot read gripper")
        hold = validate_magazine_hold(gripper_state)
        result["initial_hold_validation"] = hold
        require(bool(hold["passed"]), "gripper aperture does not indicate a held magazine")
        save(result_path, result)

        recorder = HardwareVideoRecorder(
            {"left": arm.camera, "right": environment.arm_right.camera, "head": environment.head_camera},
            output,
            fps=10.0,
        )
        recorder.start()
        recorder.wait_for_required(("left", "right"), timeout_s=10.0)

        lift_records = []
        if not already_lifted:
            recorder.mark("resume_post_grasp_vertical_lift")
            for index, waypoint in enumerate(lift_path[1:], start=1):
                code = arm.robot.rm_movej(waypoint, v=4, r=0, connect=0, block=1)
                result["real_robot_motion_sent"] = True
                query, actual = arm.get_joint_degree()
                error = float(np.max(np.abs(np.asarray(actual) - np.asarray(waypoint))))
                lift_records.append({"index": index, "return": code, "error_deg": error})
                result["lift_records"] = lift_records
                save(result_path, result)
                require(code == 0 and query == 0 and error <= 1.0, f"failed lift waypoint {index}")
        result["lift_completed"] = True

        retained_return, retained_state = arm.robot.rm_get_gripper_state()
        retained = validate_magazine_hold(retained_state)
        result["post_lift_gripper_state"] = retained_state
        result["post_lift_hold_validation"] = retained
        require(retained_return == 0 and bool(retained["passed"]), "magazine was not retained after lift")

        transfer_records = []
        for segment in box_plan["pre_release_segments"]:
            phase = str(segment["phase"])
            path = segment["path"]
            require(path_sha256(path) == segment["path_sha256"], f"changed {phase} path")
            segment_status, segment_current = arm.get_joint_degree()
            segment_error = float(np.max(np.abs(np.asarray(segment_current) - np.asarray(path[0]))))
            require(segment_status == 0 and segment_error <= 0.75, f"stale {phase} start")
            recorder.mark(f"green_box_{phase}")
            for index, waypoint in enumerate(path[1:], start=1):
                code = arm.robot.rm_movej(waypoint, v=4, r=0, connect=0, block=1)
                result["real_robot_motion_sent"] = True
                query, actual = arm.get_joint_degree()
                error = float(np.max(np.abs(np.asarray(actual) - np.asarray(waypoint))))
                transfer_records.append({"phase": phase, "index": index, "return": code, "error_deg": error})
                result["transfer_records"] = transfer_records
                save(result_path, result)
                require(code == 0 and query == 0 and error <= 1.0, f"failed {phase} waypoint {index}")
            if phase.startswith("extra_lift_"):
                check_return, check_state = arm.robot.rm_get_gripper_state()
                check = validate_magazine_hold(check_state)
                result.setdefault("transfer_hold_validations", []).append({"phase": phase, **check})
                require(check_return == 0 and bool(check["passed"]), f"magazine lost after {phase}")
        result["box_transfer_completed"] = True

        recorder.mark("green_box_release")
        release = arm.robot.rm_set_gripper_release(200, block=True, timeout=5)
        result["gripper_release_sent"] = True
        result["gripper_release_return"] = release
        save(result_path, result)
        require(release == 0, "green-box release failed")

        retreat_records = []
        for segment in box_plan["retreat_segments"]:
            phase = str(segment["phase"])
            path = segment["path"]
            require(path_sha256(path) == segment["path_sha256"], f"changed {phase} path")
            segment_status, segment_current = arm.get_joint_degree()
            segment_error = float(np.max(np.abs(np.asarray(segment_current) - np.asarray(path[0]))))
            require(segment_status == 0 and segment_error <= 0.75, f"stale {phase} start")
            recorder.mark(f"green_box_{phase}")
            for index, waypoint in enumerate(path[1:], start=1):
                code = arm.robot.rm_movej(waypoint, v=4, r=0, connect=0, block=1)
                result["real_robot_motion_sent"] = True
                query, actual = arm.get_joint_degree()
                error = float(np.max(np.abs(np.asarray(actual) - np.asarray(waypoint))))
                retreat_records.append({"phase": phase, "index": index, "return": code, "error_deg": error})
                result["retreat_records"] = retreat_records
                save(result_path, result)
                require(code == 0 and query == 0 and error <= 1.0, f"failed {phase} waypoint {index}")
        result["box_retreat_completed"] = True
        result["ok"] = True
        result["status"] = "resumed_lift_placed_in_green_box_and_retreated"
        return 0
    except Exception as error:
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        if environment is not None and result["real_robot_motion_sent"]:
            try:
                result["emergency_stop_return"] = environment.arm_left.robot.rm_set_arm_stop()
            except Exception as stop_error:
                result["emergency_stop_error"] = repr(stop_error)
        return 1
    finally:
        if recorder is not None:
            result["hardware_video"] = recorder.stop()
        save(result_path, result)
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
