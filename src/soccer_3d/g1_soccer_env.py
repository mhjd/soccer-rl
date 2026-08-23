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
APPROACH_PROGRESS_WEIGHT = 0.2
BALL_PROGRESS_WEIGHT = 0.5
APPROACH_DISTANCE = 0.75
APPROACH_LATERAL_OFFSET = -0.08
STAND_TRANSLATION_THRESHOLD = 0.05
MINIMUM_WALKING_TRANSLATION = np.array(
    [0.2, 0.3],
    dtype=np.float32,
)
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
RECOVERY_G1_X_OFFSET = (-0.1, 0.4)
RECOVERY_G1_LATERAL_DISTANCE = (0.6, 0.9)
EASY_RECOVERY_G1_X_OFFSET = (-1.0, -0.7)
EASY_RECOVERY_G1_LATERAL_DISTANCE = (0.0, 0.2)
MINIMUM_INITIAL_SEPARATION = 0.5
TASK_OBSERVATION_SIZE = 13
SOCCER_STATE_OBSERVATION_SIZE = 30
OBSERVATION_MODES = ("task", "soccer_state")
REWARD_MODES = ("goal", "approach_progress")


def normalized_action_to_command(action: np.ndarray) -> np.ndarray:
    """Map normalized actions to commands the locomotion policy can execute."""
    normalized_action = np.asarray(action, dtype=np.float32)
    if normalized_action.shape != (3,):
        raise ValueError(
            f"Expected action shape (3,), received {normalized_action.shape}"
        )
    if not np.all(np.isfinite(normalized_action)):
        raise ValueError("Action values must be finite")

    normalized_action = np.clip(normalized_action, -1.0, 1.0)
    negative_scale = -COMMAND_LOW
    command = np.where(
        normalized_action < 0.0,
        normalized_action * negative_scale,
        normalized_action * COMMAND_HIGH,
    ).astype(np.float32)
    translation_speed = float(np.linalg.norm(command[:2]))
    if translation_speed <= STAND_TRANSLATION_THRESHOLD:
        command[:] = 0.0
    else:
        activation_level = float(
            np.max(np.abs(command[:2]) / MINIMUM_WALKING_TRANSLATION)
        )
        if activation_level < 1.0:
            command[:2] /= activation_level
    return command


class G1SoccerEnv(gym.Env):
    """Soccer task controlled through a G1 locomotion policy.

    Action order: forward velocity, lateral velocity, and yaw rate.

    The task observation contains robot-frame ball XY, robot-frame ball-to-goal
    XY, robot-frame pelvis linear XY velocity, pelvis yaw rate, robot-frame ball
    linear XYZ velocity, and robot-frame ball angular XYZ velocity. The
    soccer-state mode appends the approach target, foot-ball kinematics, and
    the most recent locomotion command.
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
        recovery_start_probability: float = 0.0,
        observation_mode: str = "task",
        reward_mode: str = "goal",
    ):
        super().__init__()
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if not 0.0 <= recovery_start_probability <= 1.0:
            raise ValueError("recovery_start_probability must be in [0, 1]")
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {OBSERVATION_MODES}"
            )
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")

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
        self.recovery_start_probability = recovery_start_probability
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode

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
            shape=(
                {
                    "task": TASK_OBSERVATION_SIZE,
                    "soccer_state": SOCCER_STATE_OBSERVATION_SIZE,
                }[observation_mode],
            ),
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
        self._ordered_foot_body_ids = (
            self._name_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "left_ankle_roll_link",
            ),
            self._name_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "right_ankle_roll_link",
            ),
        )
        self._foot_body_ids = set(self._ordered_foot_body_ids)
        self._ball_radius = self.model.geom_size[self._ball_geom_id, 0]
        self._g1_qpos_address = self.model.jnt_qposadr[
            self._g1_freejoint_id
        ]
        self._ball_qpos_address = self.model.jnt_qposadr[
            self._ball_freejoint_id
        ]

        self._elapsed_steps = 0
        self._ball_contact_occurred = False
        self._recovery_start = False
        self._renderer = None
        self._viewer = None
        self._last_human_render_time = None
        self._last_command = np.zeros(3, dtype=np.float32)

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

        task_observation = np.concatenate(
            [
                ball_relative[:2],
                goal_from_ball[:2],
                pelvis_velocity[3:5],
                pelvis_velocity[2:3],
                ball_linear_velocity,
                ball_angular_velocity,
            ]
        ).astype(np.float32)
        if self.observation_mode == "task":
            observation = task_observation
        else:
            approach_position = self._approach_position()
            approach_relative = world_to_robot @ (
                approach_position - pelvis_position
            )
            foot_ball_positions = []
            foot_ball_velocities = []
            for foot_body_id in self._ordered_foot_body_ids:
                foot_ball_positions.append(
                    world_to_robot
                    @ (self.data.xpos[foot_body_id] - ball_position)
                )
                foot_world_velocity = self._body_velocity(
                    foot_body_id,
                    local=False,
                )[3:]
                foot_ball_velocities.append(
                    world_to_robot
                    @ (foot_world_velocity - ball_world_velocity[3:])
                )

            observation = np.concatenate(
                [
                    task_observation,
                    approach_relative[:2],
                    *foot_ball_positions,
                    *foot_ball_velocities,
                    self._last_command,
                ]
            ).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("The G1 soccer observation is not finite")
        return observation

    def _approach_position(self) -> np.ndarray:
        ball_position = self.data.xpos[self._ball_body_id]
        goal_position = self.data.site_xpos[self._goal_line_site_id]
        shot_direction_xy = goal_position[:2] - ball_position[:2]
        shot_distance = np.linalg.norm(shot_direction_xy)
        if shot_distance <= np.finfo(np.float64).eps:
            shot_direction_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            shot_direction_xy /= shot_distance
        shot_lateral_xy = np.array(
            [-shot_direction_xy[1], shot_direction_xy[0]],
            dtype=np.float64,
        )
        approach_xy = (
            ball_position[:2]
            - APPROACH_DISTANCE * shot_direction_xy
            + APPROACH_LATERAL_OFFSET * shot_lateral_xy
        )
        return np.array(
            [approach_xy[0], approach_xy[1], ball_position[2]],
            dtype=np.float64,
        )

    def _progress_distances(self) -> tuple[float, float]:
        pelvis_xy = self.data.xpos[self.controller.pelvis_id, :2]
        ball_xy = self.data.xpos[self._ball_body_id, :2]
        goal_xy = self.data.site_xpos[self._goal_line_site_id, :2]
        approach_distance = np.linalg.norm(
            self._approach_position()[:2] - pelvis_xy
        )
        ball_goal_distance = np.linalg.norm(goal_xy - ball_xy)
        return float(approach_distance), float(ball_goal_distance)

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
        recovery_start: bool,
        recovery_difficulty: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("Initial-state difficulty must be in [0, 1]")
        if not 0.0 <= recovery_difficulty <= 1.0:
            raise ValueError("Recovery-state difficulty must be in [0, 1]")

        final_ball_xy = self.np_random.uniform(
            low=RANDOM_BALL_XY_LOW,
            high=RANDOM_BALL_XY_HIGH,
        )
        ball_xy = FIXED_BALL_XY + difficulty * (
            final_ball_xy - FIXED_BALL_XY
        )

        if recovery_start:
            x_offset_range = np.asarray(EASY_RECOVERY_G1_X_OFFSET) + (
                recovery_difficulty
                * (
                    np.asarray(RECOVERY_G1_X_OFFSET)
                    - np.asarray(EASY_RECOVERY_G1_X_OFFSET)
                )
            )
            lateral_distance_range = np.asarray(
                EASY_RECOVERY_G1_LATERAL_DISTANCE
            ) + (
                recovery_difficulty
                * (
                    np.asarray(RECOVERY_G1_LATERAL_DISTANCE)
                    - np.asarray(EASY_RECOVERY_G1_LATERAL_DISTANCE)
                )
            )
            lateral_sign = self.np_random.choice((-1.0, 1.0))
            for _ in range(100):
                x_offset = self.np_random.uniform(*x_offset_range)
                lateral_distance = self.np_random.uniform(
                    *lateral_distance_range
                )
                offset = np.array(
                    [x_offset, lateral_sign * lateral_distance],
                    dtype=np.float64,
                )
                if np.linalg.norm(offset) >= MINIMUM_INITIAL_SEPARATION:
                    return ball_xy + offset, ball_xy
            raise RuntimeError("Could not sample a valid recovery start")

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
        return g1_xy, ball_xy

    def _set_initial_pose(
        self,
        g1_xy: np.ndarray,
        ball_xy: np.ndarray,
        g1_yaw: float = 0.0,
    ):
        g1_xy = np.asarray(g1_xy, dtype=np.float64)
        ball_xy = np.asarray(ball_xy, dtype=np.float64)
        g1_yaw = float(g1_yaw)
        if g1_xy.shape != (2,) or ball_xy.shape != (2,):
            raise ValueError("Initial XY positions must each have shape (2,)")
        if not np.all(
            np.isfinite(np.concatenate([g1_xy, ball_xy, [g1_yaw]]))
        ):
            raise ValueError("Initial pose values must be finite")
        if np.linalg.norm(g1_xy - ball_xy) < MINIMUM_INITIAL_SEPARATION:
            raise ValueError("The G1 and ball initial positions are too close")

        self.data.qpos[
            self._g1_qpos_address : self._g1_qpos_address + 2
        ] = g1_xy
        half_yaw = 0.5 * g1_yaw
        self.data.qpos[
            self._g1_qpos_address + 3 : self._g1_qpos_address + 7
        ] = (
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )
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
        recovery_start_probability = float(
            options.get(
                "recovery_start_probability",
                self.recovery_start_probability,
            )
        )
        recovery_difficulty = float(
            options.get("recovery_state_difficulty", 1.0)
        )
        if not 0.0 <= recovery_difficulty <= 1.0:
            raise ValueError("Recovery-state difficulty must be in [0, 1]")
        if not 0.0 <= recovery_start_probability <= 1.0:
            raise ValueError("recovery_start_probability must be in [0, 1]")
        if recovery_start_probability == 0.0:
            recovery_start = False
        elif recovery_start_probability == 1.0:
            recovery_start = True
        else:
            recovery_start = bool(
                self.np_random.random() < recovery_start_probability
            )
        explicit_g1_xy = options.get("initial_g1_xy")
        explicit_ball_xy = options.get("initial_ball_xy")
        if (explicit_g1_xy is None) != (explicit_ball_xy is None):
            raise ValueError(
                "initial_g1_xy and initial_ball_xy must be provided together"
            )
        if explicit_g1_xy is None:
            g1_xy, ball_xy = self._sample_initial_xy_positions(
                float(difficulty),
                recovery_start,
                recovery_difficulty,
            )
        else:
            g1_xy = np.asarray(explicit_g1_xy, dtype=np.float64)
            ball_xy = np.asarray(explicit_ball_xy, dtype=np.float64)
        g1_yaw = float(options.get("initial_g1_yaw", 0.0))
        self._set_initial_pose(g1_xy, ball_xy, g1_yaw)

        zero_command = np.zeros(3, dtype=np.float32)
        self.controller.reset(self.data, zero_command)
        self._last_command[:] = zero_command
        self._elapsed_steps = 0
        self._ball_contact_occurred = False
        self._recovery_start = recovery_start

        settle_steps = round(SETTLE_DURATION / self.model.opt.timestep)
        goal, fell = self._advance_physics(zero_command, settle_steps)
        if goal or fell:
            raise RuntimeError("The initial state did not stabilize")
        self._ball_contact_occurred = False

        if self.render_mode == "human":
            self.render()
        info = {
            "initial_g1_xy": g1_xy.copy(),
            "initial_ball_xy": ball_xy.copy(),
            "initial_g1_yaw": g1_yaw,
            "initial_state_difficulty": float(difficulty),
            "recovery_start": recovery_start,
            "recovery_start_probability": recovery_start_probability,
            "recovery_state_difficulty": recovery_difficulty,
        }
        return self._get_observation(), info

    def step(self, action):
        previous_approach_distance, previous_ball_goal_distance = (
            self._progress_distances()
        )
        command = normalized_action_to_command(action)
        goal, fell = self._advance_physics(
            command,
            self.physics_steps_per_action,
        )
        self._elapsed_steps += 1
        self._last_command[:] = command

        terminated = goal or fell
        truncated = (
            not terminated and self._elapsed_steps >= self.max_episode_steps
        )
        reward = GOAL_REWARD if goal else 0.0
        if self.reward_mode == "approach_progress":
            approach_distance, ball_goal_distance = (
                self._progress_distances()
            )
            reward += APPROACH_PROGRESS_WEIGHT * (
                previous_approach_distance - approach_distance
            )
            reward += BALL_PROGRESS_WEIGHT * (
                previous_ball_goal_distance - ball_goal_distance
            )
        observation = self._get_observation()
        info = {
            "goal": goal,
            "fell": fell,
            "ball_contact_occurred": self._ball_contact_occurred,
            "recovery_start": self._recovery_start,
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
