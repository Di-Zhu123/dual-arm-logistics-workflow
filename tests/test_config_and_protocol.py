from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import threading
import unittest

from robot_workflow.errors import ProtocolError
from robot_workflow.legacy_tcp import (
    LegacyApiGateway,
    LengthPrefixedJsonClient,
    build_grasp_request,
    build_open_vocabulary_request,
    build_simulation_request,
    build_simulation_sequence_request,
    require_response_fields,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise RuntimeError("test peer closed unexpectedly")
        data += chunk
    return data


class OneShotServer:
    def __init__(self, response_parts: tuple[bytes, ...]) -> None:
        self.response_parts = response_parts
        self.received: object | None = None
        self.error: Exception | None = None
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        try:
            connection, _ = self.server.accept()
            with connection:
                size = struct.unpack(">I", recv_exact(connection, 4))[0]
                self.received = json.loads(recv_exact(connection, size).decode("utf-8"))
                for part in self.response_parts:
                    connection.sendall(part)
        except Exception as error:  # surfaced by close(), not silently swallowed
            self.error = error
        finally:
            self.server.close()

    def __enter__(self) -> "OneShotServer":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.server.close()
            raise RuntimeError("test server did not finish")
        if self.error is not None:
            raise self.error


class ConfigAuditTests(unittest.TestCase):
    def test_example_config_cannot_enable_hardware(self) -> None:
        path = REPOSITORY / "config" / "live.example.json"
        config = json.loads(path.read_text("utf-8"))
        self.assertEqual(config["execution_mode"], "offline")
        self.assertIs(config["hardware_execution_enabled"], False)
        self.assertIsNone(config["cameras"]["head"]["world_from_camera"])

    def test_new_source_does_not_import_real_robot_sdk(self) -> None:
        source = REPOSITORY / "src" / "robot_workflow"
        combined = "\n".join(path.read_text("utf-8") for path in source.glob("*.py"))
        self.assertNotIn("rm_robot_interface", combined)
        self.assertNotIn("from Robotic_Arm", combined)


class LegacyProtocolTests(unittest.TestCase):
    def test_fragmented_length_prefixed_json_response(self) -> None:
        encoded = json.dumps({"ok": True, "detections": []}).encode("utf-8")
        packet = struct.pack(">I", len(encoded)) + encoded
        with OneShotServer(tuple(packet[index : index + 1] for index in range(len(packet)))) as server:
            client = LengthPrefixedJsonClient("127.0.0.1", server.port, timeout_s=2)
            response = client.request({"task": "det", "image": "abc"})
        self.assertEqual(response, {"ok": True, "detections": []})
        self.assertEqual(server.received, {"task": "det", "image": "abc"})

    def test_truncated_response_fails_closed(self) -> None:
        with OneShotServer((struct.pack(">I", 10), b"{}")) as server:
            client = LengthPrefixedJsonClient("127.0.0.1", server.port, timeout_s=2)
            with self.assertRaisesRegex(ProtocolError, "missing"):
                client.request({"request": 1})

    def test_oversized_response_is_rejected_before_body_read(self) -> None:
        with OneShotServer((struct.pack(">I", 1000),)) as server:
            client = LengthPrefixedJsonClient(
                "127.0.0.1", server.port, timeout_s=2, max_message_bytes=100
            )
            with self.assertRaisesRegex(ProtocolError, "limit"):
                client.request({"request": 1})

    def test_false_debug_flag_is_omitted_for_server_compatibility(self) -> None:
        request = build_grasp_request(
            depth_images=[[[1]]],
            camera_poses=[[0, 0, 0, 0, 0, 0]],
            intrinsics=[{"fx": 1}],
            center_world_m=[0, 0, 0],
            debug=False,
        )
        self.assertNotIn("debug", request)

    def test_true_debug_flag_is_included(self) -> None:
        request = build_grasp_request(
            depth_images=[[[1]]],
            camera_poses=[[0, 0, 0, 0, 0, 0]],
            intrinsics=[{"fx": 1}],
            center_world_m=[0, 0, 0],
            debug=True,
        )
        self.assertIs(request["debug"], True)

    def test_detection_uses_supported_seg_open_task(self) -> None:
        request = build_open_vocabulary_request(
            image_bytes=b"raw-rgb", image_shape=(1, 2, 3), text="red block"
        )
        self.assertEqual(request["task"], "seg_open")

    def test_simulation_recording_defaults_on_for_human_review(self) -> None:
        request = build_simulation_request(
            target_pose=(0, 0, 0, 0, 0, 0),
            arm="left",
            left_joints_deg=(0,) * 7,
            right_joints_deg=(0,) * 7,
            num_waypoints=10,
        )
        self.assertIs(request["recording"], True)
        self.assertEqual(request["arm"], 0)

    def test_simulation_can_request_an_exact_joint_target(self) -> None:
        request = build_simulation_request(
            target_pose=(0, 0, 0, 0, 0, 0),
            target_joints_deg=(0, 0, 0, 0, 0, -70, 0),
            arm="left",
            left_joints_deg=(0,) * 7,
            right_joints_deg=(0,) * 7,
            num_waypoints=25,
            recording=False,
        )
        self.assertEqual(request["target_joints"], [0.0, 0.0, 0.0, 0.0, 0.0, -70.0, 0.0])

    def test_simulation_sequence_uses_one_request_for_all_targets(self) -> None:
        request = build_simulation_sequence_request(
            targets=(
                {
                    "phase": "lift",
                    "pose": (0, 0, 0.2, 0, 0, 0),
                    "target_joints": (0,) * 7,
                    "num_waypoints": 7,
                },
                {
                    "phase": "transfer",
                    "pose": (0.4, -0.2, 0.3, 0, 0, 0),
                    "target_joints": (1,) * 7,
                    "num_waypoints": 25,
                },
            ),
            arm="left",
            left_joints_deg=(0,) * 7,
            right_joints_deg=(0,) * 7,
            recording=True,
        )
        self.assertEqual(request["task"], "sim_sequence")
        self.assertEqual([item["phase"] for item in request["targets"]], ["lift", "transfer"])
        self.assertEqual(request["targets"][1]["num_waypoints"], 25)
        self.assertIs(request["recording"], True)

    def test_simulation_sequence_rejects_wrong_segment_count(self) -> None:
        class StaticClient:
            def request(self, payload):
                return {
                    "success": True,
                    "path": [[0.0] * 7],
                    "segments": [],
                    "video": "encoded",
                }

        gateway = LegacyApiGateway("127.0.0.1", timeout_s=1)
        gateway.simulation = StaticClient()
        with self.assertRaisesRegex(ProtocolError, "wrong segment count"):
            gateway.simulate_sequence(
                targets=(
                    {
                        "phase": "only",
                        "pose": (0,) * 6,
                        "target_joints": (0,) * 7,
                        "num_waypoints": 5,
                    },
                ),
                arm="left",
                left_joints_deg=(0,) * 7,
                right_joints_deg=(0,) * 7,
            )

    def test_missing_contract_field_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "omitted"):
            require_response_fields({"success": True}, ("success", "path"), service="simulation")

    def test_successful_simulation_without_review_video_fails_closed(self) -> None:
        class StaticClient:
            def request(self, payload):
                return {"success": True, "path": [[0.0] * 7], "video": None}

        gateway = LegacyApiGateway("127.0.0.1", timeout_s=1)
        gateway.simulation = StaticClient()
        with self.assertRaisesRegex(ProtocolError, "review video"):
            gateway.simulate(
                target_pose=(0, 0, 0, 0, 0, 0),
                arm="left",
                left_joints_deg=(0,) * 7,
                right_joints_deg=(0,) * 7,
                num_waypoints=10,
            )

    def test_inconsistent_grasp_arrays_fail_closed(self) -> None:
        class StaticClient:
            def request(self, payload):
                return {"pose": [[0.0] * 6], "gdepth": [], "gwidth": [0.1], "index": [0]}

        gateway = LegacyApiGateway("127.0.0.1", timeout_s=1)
        gateway.grasp = StaticClient()
        with self.assertRaisesRegex(ProtocolError, "inconsistent lengths"):
            gateway.propose_grasps(
                depth_images=[[[1]]],
                camera_poses=[[0, 0, 0, 0, 0, 0]],
                intrinsics=[{"fx": 1}],
                center_world_m=[0, 0, 0],
            )


if __name__ == "__main__":
    unittest.main()
