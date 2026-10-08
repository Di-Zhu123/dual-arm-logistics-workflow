"""Read camera identities and calibrated stream metadata without robot motion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pyrealsense2 as rs

from .io import write_json


def inspect_device(serial: str) -> dict:
    pipeline = rs.pipeline()
    started = False
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
        started = True
        frames = pipeline.wait_for_frames(5000)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f"camera {serial} did not return color and depth")
        intrinsics = color.profile.as_video_stream_profile().get_intrinsics()
        device = profile.get_device()
        return {
            "serial_number": serial,
            "name": device.get_info(rs.camera_info.name),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "usb_type_descriptor": (
                device.get_info(rs.camera_info.usb_type_descriptor)
                if device.supports(rs.camera_info.usb_type_descriptor)
                else None
            ),
            "depth_scale_m_per_unit": float(depth.get_units()),
            "color": {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
                "distortion_model": str(intrinsics.model),
                "coeffs": [float(value) for value in intrinsics.coeffs],
                "format": str(color.profile.format()),
                "fps": int(color.profile.fps()),
            },
            "depth": {
                "width": int(depth.width),
                "height": int(depth.height),
                "format": str(depth.profile.format()),
                "fps": int(depth.profile.fps()),
            },
        }
    finally:
        if started:
            pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    context = rs.context()
    serials = [
        device.get_info(rs.camera_info.serial_number) for device in context.query_devices()
    ]
    if len(serials) != 3:
        raise RuntimeError(f"expected exactly three RealSense cameras, found {len(serials)}")
    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "warning": "device enumeration order is not a stable camera identity",
        "devices": [inspect_device(serial) for serial in serials],
    }
    write_json(Path(arguments.output), report)
    for index, device in enumerate(report["devices"]):
        print(
            f"enumeration_index={index} serial={device['serial_number']} "
            f"name={device['name']} depth_scale={device['depth_scale_m_per_unit']}"
        )
    print("No camera configuration was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
