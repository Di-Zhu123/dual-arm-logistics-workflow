"""Continuous, phase-labelled video recording for real-hardware execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np


WriterFactory = Callable[[Path, int, float, tuple[int, int]], Any]


def _default_writer_factory(
    path: Path, fourcc: int, fps: float, size: tuple[int, int]
) -> Any:
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_frame_data(frame_data: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in frame_data.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return copied


class HardwareVideoRecorder:
    """Record each camera on an independent thread while robot commands block.

    Independent workers are important because an unavailable optional head camera can
    block for several seconds.  It must not reduce the wrist-camera frame rate.
    """

    def __init__(
        self,
        cameras: Mapping[str, Any],
        output_directory: Path,
        *,
        fps: float = 10.0,
        codec: str = "mp4v",
        writer_factory: WriterFactory | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("video fps must be positive")
        if len(codec) != 4:
            raise ValueError("video codec must be four characters")
        self._cameras = dict(cameras)
        self._output_directory = Path(output_directory)
        self._fps = float(fps)
        self._codec = codec
        self._writer_factory = writer_factory or _default_writer_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._latest: dict[str, tuple[float, dict[str, Any]]] = {}
        self._phase = "initializing"
        self._started_monotonic: float | None = None
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._stop_called = False
        self._markers: list[dict[str, Any]] = []
        self._camera_state: dict[str, dict[str, Any]] = {
            name: {
                "path": str(self._output_directory / f"real_hardware_{name}.mp4"),
                "frame_count": 0,
                "error_count": 0,
                "last_error": None,
                "status": "not_started",
                "resolution": None,
            }
            for name in self._cameras
        }

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("hardware video recorder was already started")
        self._output_directory.mkdir(parents=True, exist_ok=True)
        self._started_monotonic = time.monotonic()
        self._started_at = _utc_now()
        self.mark("recording_started")
        for name, camera in self._cameras.items():
            thread = threading.Thread(
                target=self._record_camera,
                args=(name, camera),
                name=f"hardware-video-{name}",
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()

    def mark(self, phase: str) -> float:
        if self._started_monotonic is None:
            raise RuntimeError("hardware video recorder is not started")
        now = time.monotonic()
        with self._lock:
            self._phase = str(phase)
            self._markers.append(
                {
                    "phase": self._phase,
                    "elapsed_s": now - self._started_monotonic,
                    "utc": _utc_now(),
                }
            )
        return now

    def wait_for_required(
        self, names: Sequence[str] = ("left", "right"), *, timeout_s: float = 10.0
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if all(name in self._latest for name in names):
                    return
            time.sleep(0.01)
        missing = []
        with self._lock:
            missing = [name for name in names if name not in self._latest]
        raise RuntimeError(
            "required hardware video did not start for: " + ", ".join(missing)
        )

    def snapshot_after(
        self,
        requested_monotonic: float,
        *,
        required: Sequence[str] = ("left", "right"),
        timeout_s: float = 6.0,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        """Return frames captured after a phase marker; head remains optional."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                ready = all(
                    name in self._latest
                    and self._latest[name][0] >= requested_monotonic
                    for name in required
                )
                if ready:
                    observation = {
                        name: _copy_frame_data(frame_data)
                        for name, (captured, frame_data) in self._latest.items()
                        if captured >= requested_monotonic
                    }
                    errors = {
                        name: str(details["last_error"])
                        for name, details in self._camera_state.items()
                        if name not in observation and details["last_error"] is not None
                    }
                    return observation, errors
            time.sleep(0.01)
        missing = []
        with self._lock:
            missing = [
                name
                for name in required
                if name not in self._latest
                or self._latest[name][0] < requested_monotonic
            ]
        raise RuntimeError(
            "required hardware video snapshot did not arrive for: "
            + ", ".join(missing)
        )

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            cameras = {name: dict(details) for name, details in self._camera_state.items()}
            markers = [dict(marker) for marker in self._markers]
        return {
            "started_at": self._started_at,
            "stopped_at": self._stopped_at,
            "nominal_fps": self._fps,
            "codec": self._codec,
            "markers": markers,
            "cameras": cameras,
        }

    def stop(self) -> dict[str, Any]:
        if not self._stop_called:
            self._stop_called = True
            if self._started_monotonic is not None:
                self.mark("recording_stopping")
            self._stop.set()
            for thread in self._threads.values():
                thread.join(timeout=7.0)
            with self._lock:
                for name, thread in self._threads.items():
                    if thread.is_alive():
                        self._camera_state[name]["status"] = "stop_timeout"
                self._stopped_at = _utc_now()
        return self.metadata()

    def _record_camera(self, name: str, camera: Any) -> None:
        writer = None
        period_s = 1.0 / self._fps
        path = self._output_directory / f"real_hardware_{name}.mp4"
        with self._lock:
            self._camera_state[name]["status"] = "starting"
        try:
            while not self._stop.is_set():
                cycle_started = time.monotonic()
                try:
                    frame_data = camera.get_framedata()
                    rgb = np.asarray(frame_data["rgb"])
                    if rgb.ndim != 3 or rgb.shape[2] != 3:
                        raise RuntimeError(f"invalid RGB shape {rgb.shape}")
                    height, width = rgb.shape[:2]
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*self._codec)
                        writer = self._writer_factory(
                            path, fourcc, self._fps, (width, height)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot open video writer {path}")
                        with self._lock:
                            self._camera_state[name]["resolution"] = [width, height]
                            self._camera_state[name]["status"] = "recording"

                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    with self._lock:
                        phase = self._phase
                        started = self._started_monotonic
                    elapsed_s = cycle_started - float(started)
                    cv2.putText(
                        bgr,
                        f"{elapsed_s:07.2f}s {phase}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(bgr)
                    copied = _copy_frame_data(frame_data)
                    with self._lock:
                        self._latest[name] = (cycle_started, copied)
                        self._camera_state[name]["frame_count"] += 1
                        self._camera_state[name]["last_error"] = None
                except Exception as error:  # Camera timeouts are recorded, not hidden.
                    with self._lock:
                        self._camera_state[name]["error_count"] += 1
                        self._camera_state[name]["last_error"] = repr(error)
                        if writer is None:
                            self._camera_state[name]["status"] = "waiting_for_frame"
                    if self._stop.is_set():
                        break
                    time.sleep(0.05)
                remaining = period_s - (time.monotonic() - cycle_started)
                if remaining > 0:
                    self._stop.wait(remaining)
        finally:
            if writer is not None:
                writer.release()
            with self._lock:
                if self._camera_state[name]["status"] != "stop_timeout":
                    self._camera_state[name]["status"] = "stopped"
