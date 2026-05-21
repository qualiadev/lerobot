# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic foot pedal listener using evdev.

Callers supply a callback receiving the pressed key code (e.g. ``"KEY_A"``)
and an optional device path.  The listener runs in a daemon thread and
silently no-ops when :mod:`evdev` is not installed or the device is
unavailable.  Strategy-specific key mapping logic lives in the caller.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_PEDAL_DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"


def start_pedal_listener(
    on_press: Callable[[str], None],
    device_path: str = DEFAULT_PEDAL_DEVICE,
) -> threading.Thread | None:
    """Spawn a daemon thread that forwards pedal key-press codes to ``on_press``.

    Parameters
    ----------
    on_press:
        Callback invoked with the pressed key code string (e.g. ``"KEY_A"``)
        on each pedal press event.  The callback runs in the listener thread
        and must be thread-safe.
    device_path:
        Linux input device path (e.g. ``/dev/input/by-id/...``).

    Returns
    -------
    The started daemon :class:`threading.Thread`, or ``None`` when
    :mod:`evdev` is not installed (optional dependency; silent no-op).
    """
    try:
        from evdev import InputDevice, categorize, ecodes
    except ImportError:
        logger.warning(
            "Pedal listener disabled: 'evdev' is not installed in this environment. "
            "Install it (e.g. `pip install evdev`) to enable foot-pedal input."
        )
        return None

    def pedal_reader() -> None:
        try:
            dev = InputDevice(device_path)
            logger.info(
                "Pedal connected: name=%r device=%s (listening for key-down events)",
                dev.name,
                device_path,
            )
            for ev in dev.read_loop():
                if ev.type != ecodes.EV_KEY:
                    continue
                key = categorize(ev)
                code = key.keycode
                if isinstance(code, (list, tuple)):
                    code = code[0]
                if key.keystate != 1:  # only key-down events
                    continue
                logger.debug("Pedal key-down: %s", code)
                try:
                    on_press(code)
                except Exception as cb_err:  # pragma: no cover - defensive
                    logger.warning("Pedal callback error: %s", cb_err)
        except FileNotFoundError:
            logger.warning(
                "Pedal device not found: %s. The foot-pedal listener will not "
                "receive any input. Confirm the pedal is plugged in and check "
                "`ls /dev/input/by-id/` for the actual device path; pass it via "
                "--pedal.device_path=... if it differs from the default.",
                device_path,
            )
        except PermissionError:
            logger.warning(
                "Pedal device permission denied: %s. Grant the current user "
                "read access (e.g. `sudo setfacl -m u:$USER:rw %s`, or add the "
                "user to the 'input' group and re-login). Foot-pedal input will "
                "be ignored until permission is granted.",
                device_path,
                device_path,
            )
        except Exception as e:
            logger.warning("Pedal error: %s", e)

    thread = threading.Thread(target=pedal_reader, daemon=True, name="PedalListener")
    thread.start()
    return thread
