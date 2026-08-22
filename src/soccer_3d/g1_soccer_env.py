from __future__ import annotations

from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from src.soccer_3d.g1_locomotion import (
    COMMAND_HIGH,
    COMMAND_LOW,
    PHYSICS_STEPS_PER_CONTROL,
    G1LocomotionController,
    reset_g1_for_locomotion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = Path(__file__).parent / "assets" / "g1" / "soccer_scene.xml"
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "models" / "g1_locomotion" / "policy.onnx"
)

CONTROL_TIMESTEP = 0.1
MAX_EPISODE_STEPS = 100
SETTLE_DURATION = 1.0
GOAL_REWARD = 1.0
MINIMUM_PELVIS_HEIGHT = 0.45
MINIMUM_UPRIGHT_ALIGNMENT = 0.5
RENDER_HEIGHT = 400
RENDER_WIDTH = 640
FIXED_G1_XY = np.array([0.0, 0.0], dtype=np.float64)
FIXED_BALL_XY = np.array([1.0, 0.0], dtype=np.float64)
RANDOM_BALL_XY_LOW = np.array([0.8, -0.7], dtype=np.float64)
RANDOM_BALL_XY_HIGH = np.array([1.6, 0.7], dtype=np.float64)
RANDOM_BEHIND_DISTANCE = (0.8, 1.4)
RANDOM_LATERAL_OFFSET = (-0.6, 0.6)


def normalized_action_to_command(action: np.ndarray) -> np.ndarray:
    """Map [-1, 1] actions to controller commands while preserving zero."""
    normalized_action = np.asarray(action, dtype=np.float32)
    if normalized_action.shape != (3,):
        raise ValueError(
            f"Expected action shape (3,), received {normalized_action.shape}"
        )
    if not np.all(np.isfinite(normalized_action)):
        raise ValueError("Action values must be finite")

    normalized_action = np.clip(normalized_action, -1.0, 1.0)
    negative_scale = -COMMAND_LOW
    return np.where(
        normalized_action < 0.0,
        normalized_action * negative_scale,
        normalized_action * COMMAND_HIGH,
    ).astype(np.float32)


class G1SoccerEnv(gym.Env):
    """Soccer task controlled through a G1 locomotion policy.

    Action order: forward velocity, lateral velocity, and yaw rate.

    Observation order: robot-frame ball XY, robot-frame ball-to-goal XY,
    robot-frame pelvis linear XY velocity, pelvis yaw rate, robot-frame ball
    linear XYZ velocity, and robot-frame ball angular XYZ velocity.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": round(1 / CONTROL_TIMESTEP),
    }

    def __init__(
        self,
        policy_path: Path = DEFAULT_POLICY_PATH,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        render_mode: str | None = None,
        randomize_initial_positions: bool = False,
    ):
        super().__init__()
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        policy_path = Path(policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(
                f"Missing locomotion policy {policy_path}; run "
                "`make download-g1-locomotion-policy` first"
            )

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.controller = G1LocomotionController(self.model, policy_path)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.randomize_initial_positions = randomize_initial_positions

        high_level_ratio = CONTROL_TIMESTEP / self.model.opt.timestep
        self.physics_steps_per_action = round(high_level_ratio)
        if not np.isclose(
            self.physics_steps_per_action * self.model.opt.timestep,
            CONTROL_TIMESTEP,
        ):
            raise ValueError(
                "CONTROL_TIMESTEP must be an integer multiple of the MuJoCo "
                "physics timestep"
            )
        if self.physics_steps_per_action % PHYSICS_STEPS_PER_CONTROL != 0:
            raise ValueError(
                "Each high-level action must contain a whole number of "
                "locomotion-policy updates"
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

        self._ball_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "ball",
        )
        self._g1_freejoint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "floating_base_joint",
        )
        self._ball_freejoint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "ball_free",
        )
        self._ball_geom_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "ball_geom",
        )
        self._goal_line_site_id = self._name_id(
            mujoco.mjtObj.mjOBJ_SITE,
            "goal_line",
        )
        self._goal_crossbar_geom_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "goal_crossbar",
        )
        self._overview_camera_id = self._name_id(
            mujoco.mjtObj.mjOBJ_CAMERA,
            "soccer_overview",
        )
        self._foot_body_ids = {
            self._name_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "left_ankle_roll_link",
            ),
            self._name_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "right_ankle_roll_link",
            ),
        }
        self._ball_radius = self.model.geom_size[self._ball_geom_id, 0]
        self._g1_qpos_address = self.model.jnt_qposadr[
            self._g1_freejoint_id
        ]
        self._ball_qpos_address = self.model.jnt_qposadr[
            self._ball_freejoint_id
        ]

        self._elapsed_steps = 0
        self._ball_contact_occurred = False
        self._renderer = None
        self._viewer = None
        self._last_human_render_time = None

    def _name_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _body_velocity(self, body_id: int, local: bool) -> np.ndarray:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            int(local),
        )
        return velocity

    def _get_observation(self) -> np.ndarray:
        pelvis_id = self.controller.pelvis_id
        pelvis_position = self.data.xpos[pelvis_id]
        pelvis_rotation = self.data.xmat[pelvis_id].reshape(3, 3)
        world_to_robot = pelvis_rotation.T

        ball_position = self.data.xpos[self._ball_body_id]
        goal_position = self.data.site_xpos[self._goal_line_site_id]
        ball_relative = world_to_robot @ (ball_position - pelvis_position)
        goal_from_ball = world_to_robot @ (goal_position - ball_position)

        pelvis_velocity = self._body_velocity(pelvis_id, local=True)
        ball_world_velocity = self._body_velocity(
            self._ball_body_id,
            local=False,
        )
        ball_angular_velocity = world_to_robot @ ball_world_velocity[:3]
        ball_linear_velocity = world_to_robot @ ball_world_velocity[3:]

        observation = np.concatenate(
            [
                ball_relative[:2],
                goal_from_ball[:2],
                pelvis_velocity[3:5],
                pelvis_velocity[2:3],
                ball_linear_velocity,
                ball_angular_velocity,
            ]
        ).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("The G1 soccer observation is not finite")
        return observation

    def _ball_has_scored(self) -> bool:
        ball_position = self.data.xpos[self._ball_body_id]
        goal_position = self.data.site_xpos[self._goal_line_site_id]
        goal_half_width = self.model.site_size[self._goal_line_site_id, 1]
        crossbar_position = self.data.geom_xpos[
            self._goal_crossbar_geom_id
        ]
        crossbar_radius = self.model.geom_size[
            self._goal_crossbar_geom_id,
            0,
        ]

        crossed_goal_line = (
            ball_position[0] - self._ball_radius >= goal_position[0]
        )
        between_posts = (
            abs(ball_position[1] - goal_position[1]) + self._ball_radius
            <= goal_half_width
        )
        below_crossbar = (
            ball_position[2] + self._ball_radius
            <= crossbar_position[2] - crossbar_radius
        )
        return bool(crossed_goal_line and between_posts and below_crossbar)

    def _g1_has_fallen(self) -> bool:
        pelvis_id = self.controller.pelvis_id
        pelvis_rotation = self.data.xmat[pelvis_id].reshape(3, 3)
        return bool(
            self.data.xpos[pelvis_id, 2] < MINIMUM_PELVIS_HEIGHT
            or pelvis_rotation[2, 2] < MINIMUM_UPRIGHT_ALIGNMENT
        )

    def _foot_touches_ball(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == self._ball_geom_id:
                other_geom_id = contact.geom2
            elif contact.geom2 == self._ball_geom_id:
                other_geom_id = contact.geom1
            else:
                continue

            if self.model.geom_bodyid[other_geom_id] in self._foot_body_ids:
                return True
        return False

    def _sample_initial_xy_positions(
        self,
        difficulty: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("Initial-state difficulty must be in [0, 1]")

        final_ball_xy = self.np_random.uniform(
            low=RANDOM_BALL_XY_LOW,
            high=RANDOM_BALL_XY_HIGH,
        )
        behind_distance = self.np_random.uniform(*RANDOM_BEHIND_DISTANCE)
        lateral_offset = self.np_random.uniform(*RANDOM_LATERAL_OFFSET)
        final_g1_xy = np.array(
            [
                final_ball_xy[0] - behind_distance,
                final_ball_xy[1] + lateral_offset,
            ],
            dtype=np.float64,
        )

        g1_xy = FIXED_G1_XY + difficulty * (
            final_g1_xy - FIXED_G1_XY
        )
        ball_xy = FIXED_BALL_XY + difficulty * (
            final_ball_xy - FIXED_BALL_XY
        )
        return g1_xy, ball_xy

    def _set_initial_xy_positions(
        self,
        g1_xy: np.ndarray,
        ball_xy: np.ndarray,
    ):
        g1_xy = np.asarray(g1_xy, dtype=np.float64)
        ball_xy = np.asarray(ball_xy, dtype=np.float64)
        if g1_xy.shape != (2,) or ball_xy.shape != (2,):
            raise ValueError("Initial XY positions must each have shape (2,)")
        if not np.all(np.isfinite(np.concatenate([g1_xy, ball_xy]))):
            raise ValueError("Initial XY positions must be finite")
        if g1_xy[0] >= ball_xy[0]:
            raise ValueError("The G1 must start behind the ball")

        self.data.qpos[
            self._g1_qpos_address : self._g1_qpos_address + 2
        ] = g1_xy
        self.data.qpos[
            self._ball_qpos_address : self._ball_qpos_address + 2
        ] = ball_xy
        mujoco.mj_forward(self.model, self.data)

    def _advance_physics(self, command: np.ndarray, physics_steps: int):
        goal = False
        fell = False
        for physics_step in range(physics_steps):
            if physics_step % PHYSICS_STEPS_PER_CONTROL == 0:
                self.controller.policy_step(self.data, command)

            self.data.ctrl[:] = self.controller.torques(self.data)
            mujoco.mj_step(self.model, self.data)

            if not np.all(np.isfinite(self.data.qpos)) or not np.all(
                np.isfinite(self.data.qvel)
            ):
                raise RuntimeError("G1 produced a non-finite physical state")

            if self._foot_touches_ball():
                self._ball_contact_occurred = True
            goal = self._ball_has_scored()
            fell = self._g1_has_fallen()
            if goal or fell:
                break
        return goal, fell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)

        reset_g1_for_locomotion(self.model, self.data, self.controller)
        difficulty = options.get("initial_state_difficulty")
        if difficulty is None:
            difficulty = 1.0 if self.randomize_initial_positions else 0.0
        g1_xy, ball_xy = self._sample_initial_xy_positions(float(difficulty))
        self._set_initial_xy_positions(g1_xy, ball_xy)

        zero_command = np.zeros(3, dtype=np.float32)
        self.controller.reset(self.data, zero_command)
        self._elapsed_steps = 0
        self._ball_contact_occurred = False

        settle_steps = round(SETTLE_DURATION / self.model.opt.timestep)
        goal, fell = self._advance_physics(zero_command, settle_steps)
        if goal or fell:
            raise RuntimeError("The fixed initial state did not stabilize")
        self._ball_contact_occurred = False

        if self.render_mode == "human":
            self.render()
        info = {
            "initial_g1_xy": g1_xy.copy(),
            "initial_ball_xy": ball_xy.copy(),
            "initial_state_difficulty": float(difficulty),
        }
        return self._get_observation(), info

    def step(self, action):
        command = normalized_action_to_command(action)
        goal, fell = self._advance_physics(
            command,
            self.physics_steps_per_action,
        )
        self._elapsed_steps += 1

        terminated = goal or fell
        truncated = (
            not terminated and self._elapsed_steps >= self.max_episode_steps
        )
        reward = GOAL_REWARD if goal else 0.0
        observation = self._get_observation()
        info = {
            "goal": goal,
            "fell": fell,
            "ball_contact_occurred": self._ball_contact_occurred,
            "command": command.copy(),
            "elapsed_steps": self._elapsed_steps,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=RENDER_HEIGHT,
                    width=RENDER_WIDTH,
                )
            self._renderer.update_scene(self.data, camera="soccer_overview")
            return self._renderer.render().copy()

        if self._viewer is None:
            from mujoco import viewer as mujoco_viewer

            self._viewer = mujoco_viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._viewer.cam.fixedcamid = self._overview_camera_id
        elif not self._viewer.is_running():
            return None

        self._viewer.sync()
        now = time.perf_counter()
        if self._last_human_render_time is not None:
            remaining = CONTROL_TIMESTEP - (
                now - self._last_human_render_time
            )
            if remaining > 0.0:
                time.sleep(remaining)
        self._last_human_render_time = time.perf_counter()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            self._last_human_render_time = None
