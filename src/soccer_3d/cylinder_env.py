from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


SCENE_PATH = Path(__file__).parent / "assets" / "cylinder_scene.xml"

CONTROL_TIMESTEP = 0.1
MAX_EPISODE_STEPS = 100

GOAL_LINE_Y = -2.0
GOAL_HALF_WIDTH = 1.0
GOAL_HEIGHT = 0.8
GOAL_TARGET_XY = np.array([0.0, -2.4])
GOAL_REWARD = 1.0
AGENT_BALL_PROGRESS_WEIGHT = 0.1
BALL_GOAL_PROGRESS_WEIGHT = 0.2
WARMUP_AGENT_BALL_PROGRESS_WEIGHT = 0.1
AGENT_BALL_REWARD_WARMUP_STEPS = 80_000
REWARD_STRATEGIES = {
    "combined",
    "contact_phased",
    "approach_warmup",
    "ball_goal_only",
    "goal_only",
}
RANDOM_INITIAL_XY_LOW = np.array([-1.5, -0.5])
RANDOM_INITIAL_XY_HIGH = np.array([1.5, 1.0])
INITIAL_CLEARANCE = 0.25
MAX_INITIAL_POSITION_ATTEMPTS = 1000
RENDER_HEIGHT = 400
RENDER_WIDTH = 640


class CylinderSoccerEnv(gym.Env):
    """State-based Gymnasium environment for the 3D cylinder soccer task.

    Observation order:
    agent X/Y position, agent X/Y velocity, ball X/Y/Z position,
    ball X/Y/Z linear velocity, and ball X/Y/Z angular velocity.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": round(1 / CONTROL_TIMESTEP),
    }

    def __init__(
        self,
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        reward_strategy="combined",
        randomize_initial_positions=False,
    ):
        super().__init__()
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if reward_strategy not in REWARD_STRATEGIES:
            raise ValueError(
                f"Unsupported reward strategy: {reward_strategy}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_strategy = reward_strategy
        self.randomize_initial_positions = randomize_initial_positions
        self._renderer = None
        self._viewer = None
        self._last_human_render_time = None

        timestep_ratio = CONTROL_TIMESTEP / self.model.opt.timestep
        self.physics_steps_per_action = round(timestep_ratio)
        if not np.isclose(
            self.physics_steps_per_action * self.model.opt.timestep,
            CONTROL_TIMESTEP,
        ):
            raise ValueError(
                "CONTROL_TIMESTEP must be an integer multiple of the MuJoCo "
                "physics timestep"
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

        self._agent_x_joint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "agent_x",
        )
        self._agent_y_joint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "agent_y",
        )
        self._ball_joint_id = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "ball_free",
        )
        self._overview_camera_id = self._name_id(
            mujoco.mjtObj.mjOBJ_CAMERA,
            "overview",
        )
        self._agent_geom_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "agent_geom",
        )
        self._ball_geom_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "ball_geom",
        )
        self._actuator_ids = np.array(
            [
                self._name_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "agent_x_velocity",
                ),
                self._name_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    "agent_y_velocity",
                ),
            ]
        )

        self._agent_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[self._agent_x_joint_id],
                self.model.jnt_qposadr[self._agent_y_joint_id],
            ]
        )
        self._agent_qvel_addresses = np.array(
            [
                self.model.jnt_dofadr[self._agent_x_joint_id],
                self.model.jnt_dofadr[self._agent_y_joint_id],
            ]
        )
        self._ball_qpos_address = self.model.jnt_qposadr[self._ball_joint_id]
        self._ball_qvel_address = self.model.jnt_dofadr[self._ball_joint_id]
        self._agent_radius = self.model.geom_size[self._agent_geom_id, 0]
        self._ball_radius = self.model.geom_size[self._ball_geom_id, 0]

        self._elapsed_steps = 0
        self._total_steps = 0
        self._ball_contact_occurred = False

    def _name_id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _get_observation(self):
        ball_position_start = self._ball_qpos_address
        ball_velocity_start = self._ball_qvel_address

        return np.concatenate(
            [
                self.data.qpos[self._agent_qpos_addresses],
                self.data.qvel[self._agent_qvel_addresses],
                self.data.qpos[ball_position_start : ball_position_start + 3],
                self.data.qvel[ball_velocity_start : ball_velocity_start + 3],
                self.data.qvel[
                    ball_velocity_start + 3 : ball_velocity_start + 6
                ],
            ]
        ).astype(np.float32)

    def _apply_action(self, action):
        normalized_action = np.asarray(action, dtype=np.float64)
        if normalized_action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"received {normalized_action.shape}"
            )
        if not np.all(np.isfinite(normalized_action)):
            raise ValueError("Action values must be finite")

        normalized_action = np.clip(
            normalized_action,
            self.action_space.low,
            self.action_space.high,
        )
        control_ranges = self.model.actuator_ctrlrange[self._actuator_ids]
        control_low = control_ranges[:, 0]
        control_high = control_ranges[:, 1]
        physical_action = control_low + (
            (normalized_action + 1.0) * 0.5 * (control_high - control_low)
        )
        self.data.ctrl[self._actuator_ids] = physical_action

    def _ball_has_scored(self):
        ball_position = self.data.qpos[
            self._ball_qpos_address : self._ball_qpos_address + 3
        ]
        ball_x, ball_y, ball_z = ball_position

        crossed_goal_line = ball_y + self._ball_radius <= GOAL_LINE_Y
        between_posts = abs(ball_x) + self._ball_radius <= GOAL_HALF_WIDTH
        below_crossbar = ball_z + self._ball_radius <= GOAL_HEIGHT
        return bool(crossed_goal_line and between_posts and below_crossbar)

    def _shaping_distances(self):
        agent_xy = self.data.qpos[self._agent_qpos_addresses]
        ball_xy = self.data.qpos[
            self._ball_qpos_address : self._ball_qpos_address + 2
        ]
        return (
            float(np.linalg.norm(agent_xy - ball_xy)),
            float(np.linalg.norm(ball_xy - GOAL_TARGET_XY)),
        )

    def _agent_touches_ball(self):
        expected_geom_ids = {self._agent_geom_id, self._ball_geom_id}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if {contact.geom1, contact.geom2} == expected_geom_ids:
                return True
        return False

    def _sample_initial_xy_positions(self):
        minimum_separation = (
            self._agent_radius + self._ball_radius + INITIAL_CLEARANCE
        )
        for _ in range(MAX_INITIAL_POSITION_ATTEMPTS):
            agent_xy, ball_xy = self.np_random.uniform(
                low=RANDOM_INITIAL_XY_LOW,
                high=RANDOM_INITIAL_XY_HIGH,
                size=(2, 2),
            )
            if np.linalg.norm(agent_xy - ball_xy) >= minimum_separation:
                return agent_xy, ball_xy
        raise RuntimeError("Could not sample valid initial positions")

    def _set_initial_xy_positions(self, agent_xy, ball_xy):
        agent_xy = np.asarray(agent_xy, dtype=np.float64)
        ball_xy = np.asarray(ball_xy, dtype=np.float64)
        if agent_xy.shape != (2,) or ball_xy.shape != (2,):
            raise ValueError("Initial XY positions must each have shape (2,)")
        if not np.all(np.isfinite(np.concatenate([agent_xy, ball_xy]))):
            raise ValueError("Initial XY positions must be finite")

        minimum_separation = (
            self._agent_radius + self._ball_radius + INITIAL_CLEARANCE
        )
        if np.linalg.norm(agent_xy - ball_xy) < minimum_separation:
            raise ValueError("Initial agent and ball positions are too close")

        self.data.qpos[self._agent_qpos_addresses] = agent_xy
        self.data.qpos[
            self._ball_qpos_address : self._ball_qpos_address + 2
        ] = ball_xy

    def _agent_ball_reward_weight(self):
        if self.reward_strategy == "combined":
            return AGENT_BALL_PROGRESS_WEIGHT
        if self.reward_strategy == "contact_phased":
            return (
                AGENT_BALL_PROGRESS_WEIGHT
                if not self._ball_contact_occurred
                else 0.0
            )
        if self.reward_strategy == "approach_warmup":
            return (
                WARMUP_AGENT_BALL_PROGRESS_WEIGHT
                if self._total_steps < AGENT_BALL_REWARD_WARMUP_STEPS
                else 0.0
            )
        return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options

        mujoco.mj_resetData(self.model, self.data)
        initial_xy_positions = options.get("initial_xy_positions")
        if initial_xy_positions is not None:
            agent_xy, ball_xy = initial_xy_positions
            self._set_initial_xy_positions(agent_xy, ball_xy)
        elif self.randomize_initial_positions:
            agent_xy, ball_xy = self._sample_initial_xy_positions()
            self._set_initial_xy_positions(agent_xy, ball_xy)
        mujoco.mj_forward(self.model, self.data)
        self._elapsed_steps = 0
        self._ball_contact_occurred = False

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), {}

    def step(self, action):
        agent_ball_distance_before, ball_goal_distance_before = (
            self._shaping_distances()
        )
        agent_ball_reward_weight = self._agent_ball_reward_weight()
        self._apply_action(action)
        self._elapsed_steps += 1
        self._total_steps += 1

        terminated = False
        for _ in range(self.physics_steps_per_action):
            mujoco.mj_step(self.model, self.data)
            if self._agent_touches_ball():
                self._ball_contact_occurred = True
            if self._ball_has_scored():
                terminated = True
                break

        truncated = (
            not terminated and self._elapsed_steps >= self.max_episode_steps
        )
        agent_ball_distance_after, ball_goal_distance_after = (
            self._shaping_distances()
        )
        agent_ball_progress = (
            agent_ball_distance_before - agent_ball_distance_after
        )
        ball_goal_progress = (
            ball_goal_distance_before - ball_goal_distance_after
        )
        agent_ball_reward = agent_ball_reward_weight * agent_ball_progress
        ball_goal_reward = (
            0.0
            if self.reward_strategy == "goal_only"
            else BALL_GOAL_PROGRESS_WEIGHT * ball_goal_progress
        )
        shaping_reward = agent_ball_reward + ball_goal_reward
        goal_reward = GOAL_REWARD if terminated else 0.0
        reward = goal_reward + shaping_reward
        observation = self._get_observation()
        info = {
            "goal": terminated,
            "goal_reward": goal_reward,
            "shaping_reward": shaping_reward,
            "agent_ball_reward": agent_ball_reward,
            "ball_goal_reward": ball_goal_reward,
            "agent_ball_progress": agent_ball_progress,
            "ball_goal_progress": ball_goal_progress,
            "agent_ball_reward_active": agent_ball_reward_weight > 0.0,
            "agent_ball_reward_weight": agent_ball_reward_weight,
            "ball_contact_occurred": self._ball_contact_occurred,
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
            self._renderer.update_scene(self.data, camera="overview")
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
            if remaining > 0:
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
