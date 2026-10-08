"""Read-only three-camera preview, with GUI and localhost HTTP modes.

This module opens only RealSense pipelines. It never imports the robot SDK and
never commands either arm or gripper. HTTP mode binds to localhost by default;
use an SSH local port-forward from the operator's PC to view it remotely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


ROLES = ("left", "head", "right")


@dataclass
class CameraStream:
    role: str
    serial: str
    pipeline: Any
    align: Any

    @classmethod
    def open(cls, role: str, serial: str) -> "CameraStream":
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        return cls(role, serial, pipeline, rs.align(rs.stream.color))

    def read(self) -> tuple[np.ndarray, int, int]:
        frames = self.pipeline.wait_for_frames(5000)
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f"{self.role} returned an incomplete frameset")
        return (
            np.asanyarray(color.get_data()).copy(),
            int(color.get_frame_number()),
            time.time_ns(),
        )

    def close(self) -> None:
        self.pipeline.stop()


class PreviewState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg_by_role: dict[str, bytes] = {}
        self.composite_jpeg: bytes | None = None
        self.last_error: str | None = None
        self.updated_at: str | None = None
        self.operator_status = "waiting for camera frames"
        self.stop = threading.Event()

    def update(self, images: dict[str, np.ndarray], frame_numbers: dict[str, int]) -> None:
        annotated: list[np.ndarray] = []
        encoded: dict[str, bytes] = {}
        for role in ROLES:
            image = images[role].copy()
            cv2.putText(
                image,
                f"{role}  frame={frame_numbers[role]}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise RuntimeError(f"failed to encode {role} preview")
            encoded[role] = bytes(buffer)
            annotated.append(image)
        composite = np.hstack(annotated)
        cv2.putText(
            composite,
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            (12, composite.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        ok, buffer = cv2.imencode(
            ".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            raise RuntimeError("failed to encode composite preview")
        with self.condition:
            self.jpeg_by_role = encoded
            self.composite_jpeg = bytes(buffer)
            self.last_error = None
            self.updated_at = datetime.now(timezone.utc).isoformat()
            self.condition.notify_all()

    def error(self, value: Exception) -> None:
        with self.condition:
            self.last_error = str(value)
            self.condition.notify_all()

    def set_operator_status(self, value: str) -> None:
        with self.condition:
            self.operator_status = str(value)
            self.condition.notify_all()

    def status(self) -> dict[str, str | None]:
        with self.condition:
            return {
                "operator_status": self.operator_status,
                "updated_at": self.updated_at,
                "last_error": self.last_error,
            }

    def get(self, role: str | None) -> bytes | None:
        with self.condition:
            return self.composite_jpeg if role is None else self.jpeg_by_role.get(role)


def load_serials(path: str | Path) -> dict[str, str]:
    document = json.loads(Path(path).read_text("utf-8"))
    cameras = document.get("cameras", {})
    try:
        serials = {role: str(cameras[role]["serial_number"]) for role in ROLES}
    except (KeyError, TypeError) as error:
        raise ValueError("identity file must contain left/head/right serial_number") from error
    if len(set(serials.values())) != len(ROLES):
        raise ValueError("camera serial numbers must be distinct")
    return serials


def open_streams(serials: dict[str, str]) -> dict[str, CameraStream]:
    available = {
        str(device.get_info(rs.camera_info.serial_number))
        for device in rs.context().query_devices()
    }
    missing = sorted(set(serials.values()) - available)
    if missing:
        raise RuntimeError(f"configured cameras are not connected: {missing}")
    streams: dict[str, CameraStream] = {}
    try:
        for role in ROLES:
            streams[role] = CameraStream.open(role, serials[role])
    except Exception:
        for stream in streams.values():
            stream.close()
        raise
    return streams


def capture_loop(
    streams: dict[str, CameraStream], state: PreviewState, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    while not state.stop.is_set():
        try:
            captures = {role: streams[role].read() for role in ROLES}
            images = {role: value[0] for role, value in captures.items()}
            frame_numbers = {role: value[1] for role, value in captures.items()}
            state.update(images, frame_numbers)
            # Keep a single operator-readable snapshot; do not create an
            # unbounded image archive during preview.
            composite = state.get(None)
            if composite is not None:
                (output_dir / "latest_all.jpg").write_bytes(composite)
        except Exception as error:  # surfaced in HTTP status/console
            state.error(error)
            time.sleep(0.2)


class PreviewHandler(BaseHTTPRequestHandler):
    server: "PreviewHTTPServer"

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/":
            body = (
                "<html><meta charset='utf-8'><title>three cameras</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee}"
                "img{width:31%;margin:0.5%}#status{padding:8px;color:#8f8}</style>"
                "<h3>Read-only RealSense calibration preview</h3>"
                "<div id='status'>loading status...</div>"
                "<img src='/mjpeg/left'><img src='/mjpeg/head'><img src='/mjpeg/right'>"
                "<p><a href='/snapshot/all.jpg'>composite snapshot</a></p>"
                "<script>setInterval(async()=>{try{const r=await fetch('/status.json',"
                "{cache:'no-store'});const s=await r.json();"
                "document.getElementById('status').textContent=s.operator_status+"
                "(s.last_error?' | ERROR: '+s.last_error:'');}catch(e){}},500);</script>"
                "</html>"
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/status.json":
            body = json.dumps(self.server.state.status()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/snapshot/all.jpg", "/snapshot/left.jpg", "/snapshot/head.jpg", "/snapshot/right.jpg"):
            role = None if self.path.endswith("all.jpg") else self.path.split("/")[2].split(".")[0]
            image = self.server.state.get(role)
            if image is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.end_headers()
            self.wfile.write(image)
            return
        if self.path.startswith("/mjpeg/"):
            role = self.path.split("/", 2)[2]
            if role not in ROLES:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not self.server.state.stop.is_set():
                    image = self.server.state.get(role)
                    if image is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(image)).encode("ascii")
                        + b"\r\n\r\n"
                        + image
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self.send_error(HTTPStatus.NOT_FOUND)


class PreviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: PreviewState):
        super().__init__(address, PreviewHandler)
        self.state = state


def run_http(state: PreviewState, host: str, port: int) -> None:
    server = PreviewHTTPServer((host, port), state)
    print(f"preview URL: http://{host}:{port}/")
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        state.stop.set()
        server.server_close()


def run_gui(state: PreviewState) -> None:
    cv2.namedWindow("RealSense left | head | right", cv2.WINDOW_AUTOSIZE)
    try:
        while not state.stop.is_set():
            image = state.get(None)
            if image is not None:
                decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    cv2.imshow("RealSense left | head | right", decoded)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                state.stop.set()
                break
            if key == ord("s"):
                print("latest_all.jpg is updated continuously in the output directory")
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-identities", required=True)
    parser.add_argument("--mode", choices=("http", "gui"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65090)
    parser.add_argument("--output-dir", default="calibration_runs/preview")
    arguments = parser.parse_args()
    if not 1024 <= arguments.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    serials = load_serials(arguments.camera_identities)
    streams = open_streams(serials)
    state = PreviewState()
    worker = threading.Thread(
        target=capture_loop,
        args=(streams, state, Path(arguments.output_dir)),
        daemon=True,
    )
    worker.start()
    try:
        if arguments.mode == "http":
            run_http(state, arguments.host, arguments.port)
        else:
            run_gui(state)
    finally:
        state.stop.set()
        worker.join(timeout=3.0)
        for stream in streams.values():
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
