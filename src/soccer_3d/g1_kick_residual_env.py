from __future__ import annotations

from gymnasium import spaces
import mujoco
import numpy as np

from src.soccer_3d.g1_locomotion import (
    DEFAULT_JOINT_POSITION_HARDWARE,
    LEG_JOINT_COUNT,
    MAX_LEG_JOINT_RESIDUAL,
    PHYSICS_STEPS_PER_CONTROL,
)
from src.soccer_3d.g1_soccer_env import (
    GOAL_REWARD,
    SOCCER_STATE_OBSERVATION_SIZE,
    G1SoccerEnv,
)


RESIDUAL_CONTROL_TIMESTEP = 0.02
DEFAULT_RESIDUAL_EPISODE_STEPS = 200
BASE_FORWARD_COMMAND = 0.6
WARMUP_DURATION_RANGE = (0.3, 1.0)
CONTACT_DISTANCE_RANGE = (0.5, 0.7)
CONTACT_LATERAL_RANGE = (-0.16, 0.16)
RESIDUAL_OBSERVATION_SIZE = (
    SOCCER_STATE_OBSERVATION_SIZE + 4 * LEG_JOINT_COUNT
)


def get_leg_residual_observation(
    env: G1SoccerEnv,
    last_residual_action: np.ndarray,
) -> np.ndarray:
    task_observation = env._get_observation()
    leg_qpos = env.data.qpos[
        env.controller.joint_qpos_addresses[:LEG_JOINT_COUNT]
    ]
    leg_qvel = env.data.qvel[
        env.controller.joint_qvel_addresses[:LEG_JOINT_COUNT]
    ]
    default_leg_position = DEFAULT_JOINT_POSITION_HARDWARE[:LEG_JOINT_COUNT]
    base_leg_target = env.controller.base_target_joint_position_hardware[
        :LEG_JOINT_COUNT
    ]
    observation = np.concatenate(
        [
            task_observation,
            leg_qpos - default_leg_position,
            0.05 * leg_qvel,
            base_leg_target - default_leg_position,
            last_residual_action,
        ]
    ).astype(np.float32)
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("The residual observation is not finite")
    return observation


class G1KickResidualEnv(G1SoccerEnv):
    """Short-horizon leg-residual task around an imminent ball contact."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": round(1 / RESIDUAL_CONTROL_TIMESTEP),
    }

    def __init__(
        self,
        max_episode_steps: int = DEFAULT_RESIDUAL_EPISODE_STEPS,
        render_mode: str | None = None,
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            randomize_initial_positions=False,
            recovery_start_probability=0.0,
            observation_mode="soccer_state",
            reward_mode="goal",
        )
        self.control_timestep = RESIDUAL_CONTROL_TIMESTEP
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(LEG_JOINT_COUNT,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(RESIDUAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._base_command = np.array(
            [BASE_FORWARD_COMMAND, 0.0, 0.0],
            dtype=np.float32,
        )
        self._last_residual_action = np.zeros(
            LEG_JOINT_COUNT,
            dtype=np.float32,
        )
        self._ball_qvel_address = self.model.jnt_dofadr[
            self._ball_freejoint_id
        ]

    def _get_residual_observation(self) -> np.ndarray:
        return get_leg_residual_observation(
            self,
            self._last_residual_action,
        )

    def _move_ball_out_of_warmup_path(self):
        ball_qpos = self.data.qpos[
            self._ball_qpos_address : self._ball_qpos_address + 7
        ]
        ball_qpos[:3] = (2.5, 1.5, self._ball_radius)
        ball_qpos[3:] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[
            self._ball_qvel_address : self._ball_qvel_address + 6
        ] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _place_contact_ball(self, distance: float, lateral: float):
        pelvis_id = self.controller.pelvis_id
        pelvis_xy = self.data.xpos[pelvis_id, :2]
        pelvis_rotation = self.data.xmat[pelvis_id].reshape(3, 3)
        offset_world = pelvis_rotation[:2, :2] @ np.array(
            [distance, lateral],
            dtype=np.float64,
        )
        ball_qpos = self.data.qpos[
            self._ball_qpos_address : self._ball_qpos_address + 7
        ]
        ball_qpos[:3] = (
            pelvis_xy[0] + offset_world[0],
            pelvis_xy[1] + offset_world[1],
            self._ball_radius,
        )
        ball_qpos[3:] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[
            self._ball_qvel_address : self._ball_qvel_address + 6
        ] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        options = {} if options is None else dict(options)
        super().reset(seed=seed, options=options)
        self._move_ball_out_of_warmup_path()

        warmup_duration = float(
            options.get(
                "warmup_duration",
                self.np_random.uniform(*WARMUP_DURATION_RANGE),
            )
        )
        if not np.isfinite(warmup_duration) or warmup_duration < 0.0:
            raise ValueError("warmup_duration must be finite and non-negative")
        warmup_steps = round(warmup_duration / self.model.opt.timestep)
        goal, fell = self._advance_physics(
            self._base_command,
            warmup_steps,
        )
        if goal or fell:
            raise RuntimeError("The residual warm-up did not remain stable")

        contact_distance = float(
            options.get(
                "contact_distance",
                self.np_random.uniform(*CONTACT_DISTANCE_RANGE),
            )
        )
        contact_lateral = float(
            options.get(
                "contact_lateral",
                self.np_random.uniform(*CONTACT_LATERAL_RANGE),
            )
        )
        if not np.isfinite(contact_distance) or contact_distance <= 0.0:
            raise ValueError("contact_distance must be finite and positive")
        if not np.isfinite(contact_lateral):
            raise ValueError("contact_lateral must be finite")
        self._place_contact_ball(contact_distance, contact_lateral)
        self._elapsed_steps = 0
        self._ball_contact_occurred = False
        self._last_command[:] = self._base_command
        self._last_residual_action.fill(0.0)
        self._last_human_render_time = None

        if self.render_mode == "human":
            self.render()
        info = {
            "warmup_duration": warmup_duration,
            "contact_distance": contact_distance,
            "contact_lateral": contact_lateral,
            "command": self._base_command.copy(),
        }
        return self._get_residual_observation(), info

    def step(self, action):
        normalized_residual = np.asarray(action, dtype=np.float32)
        if normalized_residual.shape != (LEG_JOINT_COUNT,):
            raise ValueError(
                "Expected residual action shape "
                f"({LEG_JOINT_COUNT},), received "
                f"{normalized_residual.shape}"
            )
        if not np.all(np.isfinite(normalized_residual)):
            raise ValueError("Residual action values must be finite")
        normalized_residual = np.clip(
            normalized_residual,
            -1.0,
            1.0,
        )
        physical_residual = (
            MAX_LEG_JOINT_RESIDUAL * normalized_residual
        )
        self.controller.policy_step(
            self.data,
            self._base_command,
            physical_residual,
        )

        goal = False
        fell = False
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
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

        self._elapsed_steps += 1
        self._last_command[:] = self._base_command
        self._last_residual_action[:] = normalized_residual
        terminated = goal or fell
        truncated = (
            not terminated and self._elapsed_steps >= self.max_episode_steps
        )
        reward = GOAL_REWARD if goal else 0.0
        observation = self._get_residual_observation()
        info = {
            "goal": goal,
            "fell": fell,
            "ball_contact_occurred": self._ball_contact_occurred,
            "command": self._base_command.copy(),
            "residual": physical_residual.copy(),
            "elapsed_steps": self._elapsed_steps,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info
