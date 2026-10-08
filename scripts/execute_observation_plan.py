#!/usr/bin/env python3
"""Execute one previously simulated observation-only waypoint plan.

This adapter deliberately exposes no gripper operation.  It rejects stale
start states and plans whose final camera pose does not match the reviewed
observation target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


MAXIMUM_OBSERVATION_WAYPOINTS = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--robotdata",
        type=Path,
        default=Path("RobotDataCollection"),
    )
    parser.add_argument("--max-start-drift-deg", type=float, default=0.5)
    parser.add_argument("--max-endpoint-position-error-mm", type=float, default=10.0)
    parser.add_argument("--execute-approved-observation", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    arguments = parse_args()
    require(
        arguments.execute_approved_observation,
        "execution requires --execute-approved-observation",
    )
    require(
        0.0 < arguments.max_endpoint_position_error_mm <= 10.0,
        "observation endpoint error limit must be in (0, 10] mm",
    )
    plan: dict[str, Any] = json.loads(arguments.plan.read_text("utf-8"))
    require(plan.get("planning_only") is True, "not an observation planning artifact")
    require(plan.get("real_robot_command_sent") is False, "plan was already executed")
    require(plan.get("gripper_command_sent") is False, "plan contains gripper state")
    require(plan.get("status") == "awaiting_human_review", "plan is not reviewable")
    require(plan.get("arm") in ("left", "right"), "invalid arm")

    attempts = [item for item in plan.get("attempts", []) if item.get("simulation_solved")]
    require(len(attempts) == 1, "expected exactly one simulated observation path")
    waypoints = attempts[0].get("waypoints", [])
    require(
        1 <= len(waypoints) <= MAXIMUM_OBSERVATION_WAYPOINTS,
        "invalid observation waypoint count",
    )
    require(
        all(
            len(point) == 7 and all(math.isfinite(float(value)) for value in point)
            for point in waypoints
        ),
        "invalid waypoint",
    )
    for previous, current in zip(waypoints, waypoints[1:]):
        require(
            max(abs(float(a) - float(b)) for a, b in zip(previous, current)) <= 15.0,
            "waypoint jump exceeds 15 degrees",
        )

    sys.path.insert(0, str(arguments.robotdata))
    from utils.arm_environment import DualArmEnvironment  # type: ignore

    config = json.loads(
        (arguments.robotdata / "config" / "env_config.json").read_text("utf-8")
    )
    environment = None
    result: dict[str, Any] = {
        "ok": False,
        "observation_motion_only": True,
        "gripper_command_sent": False,
        "arm": plan["arm"],
    }
    try:
        environment = DualArmEnvironment(
            config["left_arm_config"],
            config["right_arm_config"],
            config["head_camera_config"],
        )
        selected = environment.arm_left if plan["arm"] == "left" else environment.arm_right
        status, current = selected.get_joint_degree()
        require(status == 0 and len(current) == 7, "failed to read current arm state")
        current = [float(value) for value in current]
        expected = [float(value) for value in plan["current_joints"]]
        drift = max(abs(a - b) for a, b in zip(current, expected))
        result["current_joints_before"] = current
        result["start_drift_deg"] = drift
        require(
            drift <= arguments.max_start_drift_deg,
            f"stale plan: start-state drift is {drift:.3f} degrees",
        )
        first_drift = max(abs(a - b) for a, b in zip(current, waypoints[0]))
        require(first_drift <= 0.5, "first waypoint does not match current state")

        target_camera = np.asarray(attempts[0]["target_camera_pose"], dtype=float)
        endpoint_camera = np.asarray(
            selected.camera_kinematics.get_camera_forward(waypoints[-1]), dtype=float
        )
        endpoint_error_mm = float(np.linalg.norm(endpoint_camera[:3] - target_camera[:3]) * 1000)
        result["planned_endpoint_camera_pose"] = endpoint_camera.tolist()
        result["planned_endpoint_position_error_mm"] = endpoint_error_mm
        result["maximum_endpoint_position_error_mm"] = (
            arguments.max_endpoint_position_error_mm
        )
        require(
            endpoint_error_mm <= arguments.max_endpoint_position_error_mm,
            "simulated endpoint misses camera target",
        )

        executed = 0
        for waypoint in waypoints:
            return_code = selected.movej([float(value) for value in waypoint])
            require(return_code == 0, f"robot rejected waypoint {executed}: {return_code}")
            executed += 1

        status, final_joints = selected.get_joint_degree()
        require(status == 0 and len(final_joints) == 7, "failed to read final arm state")
        final_joints = [float(value) for value in final_joints]
        final_error = max(abs(a - b) for a, b in zip(final_joints, waypoints[-1]))
        require(final_error <= 1.0, f"final joint error is {final_error:.3f} degrees")
        result.update(
            {
                "ok": True,
                "real_robot_observation_motion_sent": True,
                "executed_waypoints": executed,
                "final_joints": final_joints,
                "final_joint_error_deg": final_error,
                "final_camera_pose": [
                    float(value)
                    for value in selected.camera_kinematics.get_camera_forward(final_joints)
                ],
            }
        )
        arguments.plan.with_name("execution.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    finally:
        if environment is not None:
            if hasattr(environment, "head_camera"):
                environment.head_camera.close()
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
