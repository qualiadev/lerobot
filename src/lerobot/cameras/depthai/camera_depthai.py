"""DepthAI (Luxonis OAK) camera backend for LeRobot.

Supports OAK-D Lite, OAK-D, OAK-D Pro, and other Luxonis OAK cameras.
Uses the depthai v3 SDK to communicate over USB/XLink.

DepthAI v3 pipeline flow:
  1. Create pipeline and Camera node
  2. Build the node (connects to device hardware)
  3. Request output streams with desired size/format
  4. Create output queues from the streams
  5. Start pipeline and read frames from queues
"""

import logging
import time
from datetime import timedelta
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.depthai.configuration_depthai import DepthAICameraConfig

logger = logging.getLogger(__name__)

# Board socket mapping for stream types
_STREAM_TO_SOCKET = {
    "color": "CAM_A",   # Main RGB camera
    "left": "CAM_B",    # Left mono/stereo
    "right": "CAM_C",   # Right mono/stereo
}


# ──────────────────────────────────────────────────────────────────────────────
# DepthAI Camera (v3 API)
# ──────────────────────────────────────────────────────────────────────────────


class DepthAICamera(Camera):
    """DepthAI (Luxonis OAK) camera implementation for LeRobot.

    Uses depthai v3 SDK to access OAK cameras over USB/XLink. The camera
    runs an on-device pipeline that captures frames and sends them to
    the host via output queues.

    Supports:
      - Color (RGB) camera: OAK-D Lite IMX214 (up to 4K)
      - Left/Right mono cameras: OAK-D Lite OV7251 (up to 800p)

    Frames are returned as numpy arrays in the configured color mode,
    matching the interface expected by LeRobot's observation pipeline.
    """

    def __init__(self, config: DepthAICameraConfig):
        super().__init__(config)
        self.config: DepthAICameraConfig = config
        self._pipeline = None
        self._queue = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the camera is currently connected.

        Returns:
            bool: True if the camera is connected and ready to capture frames.
        """
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        """Connect to the OAK camera and start the pipeline.

        Uses the depthai v3 API:
          - Pipeline() as the top-level container
          - Camera node with .build() for hardware detection
          - requestOutput() for sized/typed output streams
          - createOutputQueue() for host-side frame access

        Args:
            warmup: If True, read and discard frames for warmup_s seconds
                to let auto-exposure and white balance stabilize.
        """
        if self.is_connected:
            logger.warning("DepthAI camera is already connected.")
            return

        import depthai as dai

        width = self.config.width if self.config.width is not None else 640
        height = self.config.height if self.config.height is not None else 480

        # ── Create pipeline and camera node ──
        pipeline = dai.Pipeline()

        # Determine board socket
        socket_name = _STREAM_TO_SOCKET.get(self.config.stream)
        if socket_name is None:
            raise ValueError(
                f"Invalid stream '{self.config.stream}'. "
                f"Choose from: {list(_STREAM_TO_SOCKET.keys())}"
            )
        socket = getattr(dai.CameraBoardSocket, socket_name)

        # Create and build Camera node (v3: must build before requestOutput)
        if self.config.mx_id is not None:
            # Connect to specific device by MX ID
            device = pipeline.getDefaultDevice()
            # Verify the device matches
            # We'll check after build

        cam_node = pipeline.create(dai.node.Camera, socket)
        cam_node.build()

        # ── Determine output frame type ──
        if self.config.stream == "color":
            frame_type = dai.ImgFrame.Type.BGR888i
        else:
            frame_type = None  # mono default

        # ── Request output with desired size ──
        output_kwargs = {"size": (width, height)}
        if frame_type is not None:
            output_kwargs["type"] = frame_type
        if self.config.fps is not None:
            output_kwargs["fps"] = float(self.config.fps)

        out = cam_node.requestOutput(**output_kwargs)

        # ── Create output queue ──
        self._queue = out.createOutputQueue()
        self._queue.setMaxSize(self.config.queue_size)
        self._queue.setBlocking(self.config.queue_blocking)

        # ── Rotation (v3: set on Camera node if supported) ──
        if self.config.rotation in (90, 180, 270):
            rotation_map = {
                90: dai.CameraImageOrientation.ROTATE_90_DEG
                    if hasattr(dai.CameraImageOrientation, "ROTATE_90_DEG") else None,
                180: dai.CameraImageOrientation.ROTATE_180_DEG,
                270: dai.CameraImageOrientation.ROTATE_270_DEG
                    if hasattr(dai.CameraImageOrientation, "ROTATE_270_DEG") else None,
            }
            orientation = rotation_map.get(self.config.rotation)
            if orientation is not None:
                try:
                    cam_node.setImageOrientation(orientation)
                except Exception:
                    logger.debug(f"Pipeline rotation {self.config.rotation}° not supported, will rotate in software.")

        # ── Start pipeline ──
        pipeline.start()
        self._pipeline = pipeline
        self._is_connected = True

        # Get device ID for logging
        try:
            device = pipeline.getDefaultDevice()
            mx_id = device.getDeviceId()
        except Exception:
            mx_id = self.config.mx_id or "unknown"

        logger.info(
            f"Connected to OAK camera (MX ID: {mx_id}, "
            f"stream: {self.config.stream}, "
            f"{width}x{height}@{self.config.fps or 'auto'}fps, "
            f"USB: {self.config.usb_speed})"
        )

        # ── Warmup ──
        if warmup and self.config.warmup_s > 0:
            logger.debug(f"Warming up camera for {self.config.warmup_s}s...")
            start = time.perf_counter()
            while time.perf_counter() - start < self.config.warmup_s:
                try:
                    self._queue.get(timeout=timedelta(milliseconds=500))
                except Exception:
                    break

    def disconnect(self) -> None:
        """Disconnect from the OAK camera and release resources."""
        if not self.is_connected:
            return

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping OAK pipeline: {e}")
            self._pipeline = None

        self._queue = None
        self._is_connected = False
        logger.info("Disconnected from OAK camera.")

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """Return the most recent frame, waiting up to timeout_ms for a new one.

        For DepthAI, the XLink output queue already buffers frames on-device.
        With queue_blocking=False (default), old frames are dropped so the
        latest frame in the queue is always relatively fresh.

        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds.

        Returns:
            Frame as numpy array.

        Raises:
            RuntimeError: If camera is not connected.
            TimeoutError: If no frame arrives within timeout_ms.
        """
        if not self.is_connected:
            raise RuntimeError("DepthAI camera is not connected.")

        in_frame = self._queue.tryGet()
        if in_frame is None:
            # No frame buffered — wait up to timeout_ms
            try:
                in_frame = self._queue.get(timeout=timedelta(milliseconds=timeout_ms))
            except Exception:
                in_frame = None
        if in_frame is None:
            raise TimeoutError(
                f"Timed out waiting for frame from DepthAI camera after {timeout_ms}ms."
            )
        return self._process_frame(in_frame)

    def _process_frame(self, in_frame) -> np.ndarray:
        """Convert a depthai ImgFrame to a numpy array with configured color mode."""
        if self.config.stream == "color":
            frame = in_frame.getCvFrame()
            if self.config.color_mode == ColorMode.RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = in_frame.getFrame()
            if self.config.color_mode == ColorMode.RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif self.config.color_mode == ColorMode.BGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.config.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return frame

    def read(self) -> np.ndarray:
        """Read a frame from the camera (blocking).

        Returns:
            Frame as numpy array, shape (H, W, 3) for color or (H, W) for grayscale.
        """
        if not self.is_connected:
            raise RuntimeError("DepthAI camera is not connected.")

        try:
            in_frame = self._queue.get(timeout=timedelta(milliseconds=5000))
        except Exception:
            in_frame = None
        if in_frame is None:
            raise RuntimeError("Timeout reading frame from OAK camera.")

        return self._process_frame(in_frame)

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Find all available OAK cameras connected via USB.

        Returns:
            List of dicts with keys: type, id, name, default_stream_profile
        """
        import depthai as dai

        cameras = []
        available_devices = dai.Device.getAllAvailableDevices()

        for device_info in available_devices:
            mx_id = device_info.getDeviceId()
            state = str(device_info.state).split(".")[-1]

            camera_info = {
                "type": "DepthAI",
                "id": mx_id,
                "name": f"OAK Device {mx_id[:8]}... ({state})",
                "default_stream_profile": {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "format": "BGR",
                },
            }
            cameras.append(camera_info)

        return cameras
