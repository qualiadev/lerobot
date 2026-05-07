"""Configuration for DepthAI (Luxonis OAK) cameras."""

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("depthai")
@dataclass
class DepthAICameraConfig(CameraConfig):
    """Configuration class for DepthAI (Luxonis OAK) camera devices.

    The OAK-D Lite has:
      - 1x 4K color camera (IMX214)
      - 2x mono cameras (OV7251) for stereo depth

    Example configurations:
    ```python
    # Basic color camera
    DepthAICameraConfig(fps=30, width=640, height=480)

    # Specific device by MX ID
    DepthAICameraConfig(fps=30, width=640, height=480, mx_id="14442C1001F5C5D700")

    # Mono camera (left stereo)
    DepthAICameraConfig(fps=30, width=640, height=480, stream="left")

    # USB2 mode (limited bandwidth)
    DepthAICameraConfig(fps=30, width=640, height=480, usb_speed="usb2")
    ```

    Attributes:
        mx_id: Device MX ID (serial number). None = use first available device.
        stream: Which camera stream to use ("color", "left", "right").
        sensor_resolution: Native sensor resolution before preview scaling.
            Color: "1080p" (default), "4k", "12mp", "13mp"
            Mono: "400p", "480p" (default), "720p", "800p"
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        interleaved: Whether to use interleaved frame layout (False = planar).
        usb_speed: USB speed limit. "usb3" for full bandwidth, "usb2" for
            compatibility mode (lower resolution/fps).
        rotation: Image rotation in degrees (0, 90, 180, 270).
        manual_focus: Manual focus value (0-255). None = autofocus.
        auto_exposure: Enable auto exposure (default True).
        brightness: Brightness adjustment (-10 to 10). None = default.
        saturation: Saturation adjustment (-10 to 10). None = default.
        warmup_s: Time reading frames before returning from connect (in seconds).
        queue_size: XLink output queue size. Larger = more latency tolerance,
            smaller = fresher frames.
        queue_blocking: If True, queue blocks when full. If False, drops oldest
            frames (preferred for real-time inference).
    """

    mx_id: str | None = None
    stream: str = "color"  # "color", "left", "right"
    sensor_resolution: str | None = None  # None = auto (1080p color, 480p mono)
    color_mode: ColorMode = ColorMode.RGB
    interleaved: bool = False
    usb_speed: str = "usb3"
    rotation: int = 0
    manual_focus: int | None = None
    auto_exposure: bool = True
    brightness: int | None = None
    saturation: int | None = None
    warmup_s: int = 1
    queue_size: int = 4
    queue_blocking: bool = False

    def __post_init__(self):
        self.color_mode = ColorMode(self.color_mode)

        if self.stream not in ("color", "left", "right"):
            raise ValueError(
                f"`stream` must be 'color', 'left', or 'right', but '{self.stream}' was provided."
            )

        if self.rotation not in (0, 90, 180, 270):
            raise ValueError(
                f"`rotation` must be 0, 90, 180, or 270, but {self.rotation} was provided."
            )

        if self.manual_focus is not None and not (0 <= self.manual_focus <= 255):
            raise ValueError(
                f"`manual_focus` must be 0-255, but {self.manual_focus} was provided."
            )

        if self.brightness is not None and not (-10 <= self.brightness <= 10):
            raise ValueError(
                f"`brightness` must be -10 to 10, but {self.brightness} was provided."
            )

        if self.saturation is not None and not (-10 <= self.saturation <= 10):
            raise ValueError(
                f"`saturation` must be -10 to 10, but {self.saturation} was provided."
            )

        if self.usb_speed not in ("usb2", "usb3"):
            raise ValueError(
                f"`usb_speed` must be 'usb2' or 'usb3', but '{self.usb_speed}' was provided."
            )
