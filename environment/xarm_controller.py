"""
Synchronous xArm controller with optional tactile-driven gripper safety.

Adapted from tactile-ril-env/ril_env/xarm_controller.py (legacy XArm class):
  - Continuous grasp mapping in [0, 1] -> gripper SDK units via _apply_grasp.
  - Optional tactile=TactileSensors hook. When the contact metric exceeds
    the configured threshold (or readings are stale), the controller
    refuses to close further: it clamps `grasp` to `previous_grasp`.
    Opening is always allowed.

Primary API is `step_abs(new_position, new_orientation, grasp)` so the
existing teleop loop (which derives an absolute pose from the phone) can
keep its current shape.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from xarm.wrapper import XArmAPI

from environment.tactile import TactileSensors


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class XArmConfig:
    robot_ip: str = "192.168.1.223"
    home_pos: List[int] = field(default_factory=lambda: [0, 0, 0, 70, 0, 70, 0])
    home_speed: float = 50.0
    tcp_maxacc: int = 5000

    # Gripper bounds in xArm SDK units. grasp=0.0 -> open_pos, grasp=1.0 -> close_pos.
    gripper_open_pos: int = 850
    gripper_close_pos: int = 0
    # Minimum change in grasp (in [0,1]) before re-issuing a gripper command.
    # Prevents spamming the gripper API every tick from tiny numeric noise.
    gripper_eps: float = 0.01

    use_gripper: bool = True
    verbose: bool = False


def _apply_grasp(arm, grasp: float, previous_grasp: float, config: XArmConfig) -> float:
    """Map grasp in [0,1] to a continuous gripper position and command it if changed.

    Returns the new previous_grasp value (unchanged if below epsilon).
    """
    grasp = float(np.clip(grasp, 0.0, 1.0))
    if abs(grasp - previous_grasp) < config.gripper_eps:
        return previous_grasp
    open_pos = config.gripper_open_pos
    close_pos = config.gripper_close_pos
    target = int(round(open_pos + grasp * (close_pos - open_pos)))
    code = arm.set_gripper_position(target, wait=False)
    if code != 0:
        logger.error(f"Error in set_gripper_position({target}): {code}")
        raise RuntimeError(f"Error in set_gripper_position({target}): {code}")
    return grasp


class XArm:
    """Synchronous xArm wrapper. Use via `with XArm(cfg) as arm:` or call
    `initialize()` / `shutdown()` directly.

    The optional `tactile` argument enables the gripper safety wrapper —
    every `step_abs` call queries TactileSensors.safety() and clamps a
    closing command to `previous_grasp` when contact is over threshold or
    data is stale.
    """

    def __init__(
        self,
        xarm_config: XArmConfig,
        tactile: Optional[TactileSensors] = None,
    ):
        self.config = xarm_config
        self.tactile = tactile
        self.init = False
        self._safety_active = False

        self.current_position: Optional[np.ndarray] = None
        self.current_orientation: Optional[np.ndarray] = None
        self.previous_grasp = 0.0
        self.use_gripper = xarm_config.use_gripper

        if self.config.verbose:
            logger.setLevel(logging.DEBUG)

    @property
    def is_ready(self) -> bool:
        return self.init

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self):
        self.arm = XArmAPI(self.config.robot_ip)
        arm = self.arm

        arm.connect()
        arm.clean_error()
        arm.clean_warn()

        code = arm.motion_enable(enable=True)
        if code != 0:
            raise RuntimeError(f"Error in motion_enable: {code}")

        arm.set_tcp_maxacc(self.config.tcp_maxacc)

        code = arm.set_mode(1)
        if code != 0:
            raise RuntimeError(f"Error in set_mode: {code}")

        code = arm.set_state(0)
        if code != 0:
            raise RuntimeError(f"Error in set_state: {code}")

        code, state = arm.get_state()
        if code != 0:
            raise RuntimeError(f"Error getting robot state: {code}")
        if state != 0:
            raise RuntimeError(f"Robot is not ready to move. Current state: {state}")
        logger.info(f"Robot is ready to move. Current state: {state}")

        err_code, warn_code = arm.get_err_warn_code()
        if err_code != 0 or warn_code != 0:
            logger.error(
                f"Error code: {err_code}, Warning code: {warn_code}. "
                "Cleaning error and warning."
            )
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(enable=True)
            arm.set_state(0)

        if self.use_gripper:
            code = arm.set_gripper_mode(0)
            if code != 0:
                raise RuntimeError(f"Error in set_gripper_mode: {code}")
            code = arm.set_gripper_enable(True)
            if code != 0:
                raise RuntimeError(f"Error in set_gripper_enable: {code}")
            code = arm.set_gripper_speed(1000)
            if code != 0:
                raise RuntimeError(f"Error in set_gripper_speed: {code}")

        self.init = True
        time.sleep(3)
        self.home()
        time.sleep(3)
        logger.info("Successfully initialized xArm.")

    def shutdown(self):
        if not self.init:
            logger.error("shutdown() called on an uninitialized xArm.")
            return
        self.home()
        self.arm.disconnect()
        logger.info("xArm shutdown complete.")

    def home(self):
        if not self.init:
            raise RuntimeError("xArm not initialized.")
        logger.info("Homing robot.")
        arm = self.arm
        arm.set_mode(0)
        arm.set_state(0)
        if self.use_gripper:
            code = arm.set_gripper_position(self.config.gripper_open_pos, wait=False)
            if code != 0:
                raise RuntimeError(
                    f"Error in set_gripper_position (open, homing): {code}"
                )
        code = arm.set_servo_angle(
            angle=self.config.home_pos, speed=self.config.home_speed, wait=True
        )
        if code != 0:
            raise RuntimeError(f"Error in set_servo_angle (homing): {code}")
        arm.set_mode(1)
        arm.set_state(0)

        code, pose = arm.get_position()
        if code != 0:
            raise RuntimeError(f"Failed to query initial pose: {code}")
        self.current_position = np.array(pose[:3])
        self.current_orientation = np.array(pose[3:])
        # Gripper is now open after homing; mirror that in our cached state
        # so the next non-zero grasp command isn't silently swallowed by the
        # epsilon gate in _apply_grasp.
        self.previous_grasp = 0.0
        self._safety_active = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def step_abs(self, new_position=None, new_orientation=None, grasp: float = 0.0):
        """Command an absolute Cartesian pose. Tactile safety, if wired,
        clamps `grasp` to `previous_grasp` when contact exceeds threshold.
        """
        if not self.init:
            raise RuntimeError("xArm not initialized. Use it in a 'with' block")

        if new_position is not None:
            self.current_position = np.asarray(new_position, dtype=np.float64)
        if new_orientation is not None:
            self.current_orientation = np.asarray(new_orientation, dtype=np.float64)

        code = self.arm.set_servo_cartesian(
            np.concatenate((self.current_position, self.current_orientation)),
            is_radian=False,
        )
        if code != 0:
            raise RuntimeError(f"Error in set_servo_cartesian in step_abs(): {code}")

        if self.use_gripper:
            grasp = self._apply_tactile_safety(grasp)
            self.previous_grasp = _apply_grasp(
                self.arm, grasp, self.previous_grasp, self.config
            )

    def _apply_tactile_safety(self, grasp: float) -> float:
        """If tactile is wired and metric is over threshold (or stale),
        clamp a closing command to the previous grasp value. Opening is
        always allowed.
        """
        if self.tactile is None:
            self._safety_active = False
            return grasp
        try:
            metric_val, is_safe = self.tactile.safety()
        except Exception as e:
            logger.error(f"[XArm] tactile read failed: {e}")
            metric_val, is_safe = float("nan"), False

        closing = grasp > self.previous_grasp
        if not is_safe and closing:
            if not self._safety_active:
                logger.warning(
                    f"[XArm] tactile safety engaged "
                    f"(metric={metric_val:.2f}, "
                    f"threshold={self.tactile.config.safety_threshold:.2f}); "
                    f"holding grasp at {self.previous_grasp:.3f}"
                )
            self._safety_active = True
            return self.previous_grasp
        self._safety_active = False
        return grasp

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        state: dict = {}
        code, actual_pose = self.arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError(f"Error getting TCP pose: code {code}")
        state["ActualTCPPose"] = actual_pose

        tcp_speed_attr = self.arm.realtime_tcp_speed
        state["ActualTCPSpeed"] = tcp_speed_attr() if callable(tcp_speed_attr) else tcp_speed_attr

        code, actual_angles = self.arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise RuntimeError(f"Error getting joint angles: code {code}")
        state["ActualQ"] = actual_angles

        joint_speeds_attr = self.arm.realtime_joint_speeds
        state["ActualQd"] = joint_speeds_attr() if callable(joint_speeds_attr) else joint_speeds_attr

        state["TactileSafetyActive"] = 1.0 if self._safety_active else 0.0
        return state
