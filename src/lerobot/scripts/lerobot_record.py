# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.camera_encoder.vcodec=h264 \\
    --dataset.camera_encoder.preset=fast \\
    --dataset.camera_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

# Patch: force DepthAICameraConfig registration with draccus ChoiceClass
from lerobot.cameras.depthai.configuration_depthai import DepthAICameraConfig  # noqa: F401

import logging
import time
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.pedal import DEFAULT_PEDAL_DEVICE, start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# State used by the slow-loop warning rate-limiter in ``record_loop``.  Kept at
# module scope so a re-entry into ``record_loop`` (e.g. episode -> reset ->
# next episode) preserves the "are we currently in the slow regime?" bit and
# avoids spamming a fresh WARN on every loop entry.
_slow_warn_state = {
    "slow": False,           # True while we believe the loop is slow
    "slow_since_s": 0.0,     # perf_counter time when we entered the slow regime
    "last_emit_s": 0.0,      # perf_counter time of the last WARN emit
}
# Minimum seconds between repeated "loop is slow" WARNs once we're in the slow
# regime.  One WARN every ~5s is enough to surface a real problem without
# burying the rest of the logs.
_SLOW_WARN_REMIND_S = 5.0


@dataclass
class RecordPedalConfig:
    """Foot pedal configuration for recording controls.

    Default mapping for the PCsensor 3-key foot pedal:
      * ``toggle_pause`` (KEY_C):  pause/resume recording during an episode,
        and "done resetting, start next episode" during the reset window.
        Robot keeps executing teleop while paused; only ``add_frame`` is
        skipped, so resume produces a continuous in-distribution recording
        with the operator's chosen gap segments excluded.
      * ``end_episode`` (KEY_B):   end the current episode now and save it
        normally.  Pairs with ``--dataset.episode_wait_for_pedal=true`` for
        an indefinite-length episode where pedal-B is the only way out.
      * ``discard_episode`` (KEY_A): discard the current episode (clear its
        frame buffer) and immediately enter the reset window so the operator
        can re-record.  Equivalent to the existing Escape-key behaviour but
        on the foot pedal.

    Pedal codes are evdev key code strings (e.g. ``"KEY_A"``) as reported by
    the underlying USB HID device.
    """

    # Enable pedal listener.  When false, no pedal thread is spawned.
    enabled: bool = False
    # Linux input device path.  Defaults to the PCsensor single-pedal device.
    device_path: str = DEFAULT_PEDAL_DEVICE
    # Evdev key code emitted by the pedal that toggles pause/resume during an
    # episode, or "start next episode" during the reset window.
    toggle_pause: str = "KEY_C"
    # Evdev key code that ends the current episode and saves it normally.
    end_episode: str = "KEY_B"
    # Evdev key code that discards the current episode and re-runs it after the
    # reset window.  Wires into the existing ``events['rerecord_episode']``
    # signal used elsewhere in lerobot (the Escape key default).
    discard_episode: str = "KEY_A"


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # Foot pedal pause/resume controls for dataset frame writes.
    pedal: RecordPedalConfig = dataclass_field(default_factory=RecordPedalConfig)

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    is_reset_loop: bool = False,
    wait_for_pedal_to_end: bool = False,
    slow_loop_warn_below_hz: float = 0.0,
):
    # When ``is_reset_loop`` is True the loop ignores ``control_time_s`` and
    # waits indefinitely for ``events['exit_early']`` to flip (toggled by the
    # foot pedal, space-bar, or the right-arrow key).  This is useful when the
    # operator needs an unbounded amount of time to reset the environment
    # between episodes.
    #
    # When ``wait_for_pedal_to_end`` is True (during an actual recording
    # episode), the loop also ignores ``control_time_s`` and runs until the
    # operator triggers ``exit_early`` (pedal-B / pedal-A / Escape / Enter)
    # — i.e. "infinite" episode duration.
    #
    # ``slow_loop_warn_below_hz`` filters the "Record loop is running slower"
    # warning so it only fires when the measured rate drops below the given
    # threshold.  0 (the default) keeps the old per-tick behaviour.  The
    # warning is also edge-triggered with periodic reminders to avoid flooding
    # logs at the loop rate.
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    # Track pause-state transitions so we only log once per edge instead of every tick,
    # and accumulate paused wall-clock time so it does NOT count against control_time_s.
    # The effective timestamp is `now - start - total_paused`, so the budget
    # ('episode_time_s' / 'reset_time_s') only ticks while actively recording.
    last_paused_state = events.get("paused", False)
    total_paused_s = 0.0
    paused_segment_start: float | None = (
        time.perf_counter() if last_paused_state else None
    )
    if last_paused_state and dataset is not None:
        logging.info("Recording starts in PAUSED state — frames will not be appended until pedal/key resume.")
    if is_reset_loop:
        logging.info(
            "Reset window started — waiting for pedal/space/right-arrow press to "
            "start the next episode (reset_time_s is ignored)."
        )
    elif wait_for_pedal_to_end:
        logging.info(
            "Episode started in indefinite mode — will record until pedal-end "
            "is pressed (episode_time_s is ignored)."
        )
    while is_reset_loop or wait_for_pedal_to_end or timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Edge-trigger logging on pause/resume.  The pause flag is mutated by the
        # pedal listener thread (and the keyboard listener) — we do not clear it
        # here so the state persists across loop iterations until toggled again.
        paused = events.get("paused", False)
        if paused != last_paused_state:
            if paused:
                paused_segment_start = start_loop_t
                logging.info("Recording PAUSED — robot still executing teleop, but frames are NOT being saved.")
            else:
                if paused_segment_start is not None:
                    segment_s = start_loop_t - paused_segment_start
                    total_paused_s += segment_s
                    paused_segment_start = None
                    logging.info(
                        "Recording RESUMED — appending frames to dataset "
                        "(paused %.2fs; total paused this episode %.2fs).",
                        segment_s,
                        total_paused_s,
                    )
                else:
                    logging.info("Recording RESUMED — appending frames to dataset.")
            last_paused_state = paused

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset (skipped while paused: robot keeps moving, but no frames
        # are appended; this produces a continuous in-distribution recording with
        # the operator's chosen gap segments excluded).
        if dataset is not None and not paused:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            measured_hz = 1 / dt_s if dt_s > 0 else float("inf")
            # Only emit when below the user-specified threshold (0 = always),
            # and rate-limit log lines.  We use closure-local state attached
            # to the function so we don't pollute the outer scope.
            should_emit = (
                slow_loop_warn_below_hz <= 0.0
                or measured_hz < slow_loop_warn_below_hz
            )
            if should_emit:
                state = _slow_warn_state
                now_warn = time.perf_counter()
                if not state["slow"] or (now_warn - state["last_emit_s"]) >= _SLOW_WARN_REMIND_S:
                    if not state["slow"]:
                        state["slow"] = True
                        state["slow_since_s"] = now_warn
                    logging.warning(
                        "Record loop is running slower (%.1f Hz) than the target FPS (%d Hz). "
                        "Dataset frames might be dropped and robot control might be unstable. "
                        "Common causes are: 1) Camera FPS not keeping up 2) Policy inference "
                        "taking too long 3) CPU starvation",
                        measured_hz,
                        fps,
                    )
                    state["last_emit_s"] = now_warn
        else:
            # Loop met the target this tick.  If we were previously in the
            # slow regime, log a single recovery line and reset the state.
            state = _slow_warn_state
            if state["slow"]:
                slow_for = time.perf_counter() - state["slow_since_s"]
                logging.info(
                    "Record loop recovered to target FPS (was slow for %.1fs).",
                    slow_for,
                )
                state["slow"] = False

        precise_sleep(max(sleep_time_s, 0.0))

        # Subtract accumulated paused wall-clock time so the episode timer only
        # advances while we are actually recording.  If currently paused, also
        # subtract the in-progress paused segment so the loop doesn't tick out
        # halfway through a long pause.
        now = time.perf_counter()
        live_pause_s = (now - paused_segment_start) if paused_segment_start is not None else 0.0
        timestamp = now - start_episode_t - total_paused_s - live_pause_s


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    pedal_thread = None
    space_listener = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()
        # Augment the events dict with a pause flag toggled by the foot pedal or
        # the space-bar.  The dict is shared by reference with all listener
        # threads and the recording loop, so flipping the flag from any thread
        # takes effect on the next loop iteration.
        events["paused"] = False

        # Track whether we're currently in the reset window so the pedal /
        # space-bar can act as "start next episode" instead of pause/resume.
        events["reset_active"] = False

        if cfg.pedal.enabled:
            toggle_code = cfg.pedal.toggle_pause
            end_code = cfg.pedal.end_episode
            discard_code = cfg.pedal.discard_episode

            def _on_pedal_press(code: str) -> None:
                # toggle_pause (default KEY_C): pause/resume during episode,
                # "start next episode" during the reset window.
                if code == toggle_code:
                    if events.get("reset_active", False):
                        events["exit_early"] = True
                        logging.info("Pedal (%s) pressed during reset — starting next episode.", code)
                    else:
                        events["paused"] = not events.get("paused", False)
                        state = "PAUSED" if events["paused"] else "RESUMED"
                        logging.info("Pedal (%s) toggled recording: %s", code, state)
                    return
                # end_episode (default KEY_B): end current episode and save
                # normally.  No-op during the reset window.
                if code == end_code:
                    if events.get("reset_active", False):
                        logging.info(
                            "Pedal (%s) end-episode pressed during reset — ignored "
                            "(press %s to start the next episode).",
                            code,
                            toggle_code,
                        )
                    else:
                        events["exit_early"] = True
                        # Clear paused so the reset window starts cleanly
                        # (otherwise a leftover PAUSED state would carry over).
                        events["paused"] = False
                        logging.info(
                            "Pedal (%s) end-episode pressed — saving episode and "
                            "entering reset.",
                            code,
                        )
                    return
                # discard_episode (default KEY_A): drop current episode, enter
                # reset window, re-record.  No-op during reset (you already are
                # between episodes; the previous discard already cleared it).
                if code == discard_code:
                    if events.get("reset_active", False):
                        logging.info(
                            "Pedal (%s) discard pressed during reset — ignored "
                            "(already between episodes).",
                            code,
                        )
                    else:
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                        events["paused"] = False
                        logging.info(
                            "Pedal (%s) discard pressed — dropping episode, "
                            "entering reset.",
                            code,
                        )
                    return
                # Unknown code: log at DEBUG to aid pedal-mapping discovery.
                logging.debug("Pedal key-down ignored (no mapping): %s", code)

            pedal_thread = start_pedal_listener(
                _on_pedal_press, device_path=cfg.pedal.device_path
            )
            if pedal_thread is None:
                logging.warning(
                    "Pedal listener could not start (evdev missing or device %s unavailable). "
                    "Falling back to keyboard space-bar for pause toggle.",
                    cfg.pedal.device_path,
                )
            else:
                logging.info(
                    "Foot pedal listener started on %s. Active mapping: "
                    "%s = pause/resume (or 'start next episode' during reset); "
                    "%s = end current episode and save; "
                    "%s = discard current episode and re-record.",
                    cfg.pedal.device_path,
                    toggle_code,
                    end_code,
                    discard_code,
                )

        # Always also bind space-bar as a pause toggle when not headless: it is
        # useful both as a fallback when the pedal is unavailable and as an
        # alternative input.  ``init_keyboard_listener`` already owns the arrow
        # / ESC keys, so we register a second pynput listener for space only.
        if not is_headless():
            try:
                from pynput import keyboard as _kb

                def _on_space(key):
                    if key != _kb.Key.space:
                        return
                    if events.get("reset_active", False):
                        events["exit_early"] = True
                        logging.info("Space-bar pressed during reset — starting next episode.")
                    else:
                        events["paused"] = not events.get("paused", False)
                        state = "PAUSED" if events["paused"] else "RESUMED"
                        logging.info("Space-bar toggled recording: %s", state)

                space_listener = _kb.Listener(on_press=_on_space)
                space_listener.start()
                logging.info("Space-bar pause toggle armed.")
            except Exception as _e:  # pragma: no cover - defensive
                logging.debug("Space-bar pause listener unavailable: %s", _e)

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.camera_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                # Clear pedal-driven flags entering the episode so a stale True
                # from the previous reset window can't immediately end it.
                events["exit_early"] = False
                events["rerecord_episode"] = False
                events["paused"] = False
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    wait_for_pedal_to_end=cfg.dataset.episode_wait_for_pedal,
                    slow_loop_warn_below_hz=cfg.dataset.slow_loop_warn_below_hz,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)

                    events["reset_active"] = True
                    # Make sure exit_early is fresh entering the reset window so
                    # a stale True doesn't immediately end it, and clear paused
                    # so the next episode starts unpaused.
                    events["exit_early"] = False
                    events["paused"] = False
                    try:
                        record_loop(
                            robot=robot,
                            events=events,
                            fps=cfg.dataset.fps,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            teleop=teleop,
                            control_time_s=cfg.dataset.reset_time_s,
                            single_task=cfg.dataset.single_task,
                            display_data=cfg.display_data,
                            is_reset_loop=cfg.dataset.reset_wait_for_pedal,
                            slow_loop_warn_below_hz=cfg.dataset.slow_loop_warn_below_hz,
                        )
                    finally:
                        events["reset_active"] = False

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()
        if space_listener is not None:
            try:
                space_listener.stop()
            except Exception:  # pragma: no cover - defensive
                pass
        # pedal_thread is a daemon evdev reader — it will exit on process teardown;
        # we do not need to join() it here.

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
