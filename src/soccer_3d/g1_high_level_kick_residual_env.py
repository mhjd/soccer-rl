from __future__ import annotations

from gymnasium import spaces
import mujoco
import numpy as np

from src.soccer_3d.g1_locomotion import COMMAND_HIGH, COMMAND_LOW
from src.soccer_3d.g1_soccer_env import (
    GOAL_REWARD,
    SOCCER_STATE_OBSERVATION_SIZE,
    G1SoccerEnv,
)


HIGH_LEVEL_RESIDUAL_SIZE = 3
HIGH_LEVEL_RESIDUAL_OBSERVATION_SIZE = (
    SOCCER_STATE_OBSERVATION_SIZE + 2 * HIGH_LEVEL_RESIDUAL_SIZE
)
MAX_HIGH_LEVEL_COMMAND_RESIDUAL = np.array(
    [0.15, 0.08, 0.06],
    dtype=np.float32,
)
DEFAULT_HIGH_LEVEL_RESIDUAL_EPISODE_STEPS = 40
BASE_FORWARD_COMMAND = 0.6
WARMUP_DURATION_RANGE = (0.3, 1.0)
CONTACT_DISTANCE_RANGE = (0.5, 0.7)
CONTACT_LATERAL_RANGE = (-0.16, 0.16)


def get_high_level_residual_observation(
    env: G1SoccerEnv,
    base_command: np.ndarray,
    last_residual_action: np.ndarray,
) -> np.ndarray:
    base_command = np.asarray(base_command, dtype=np.float32)
    last_residual_action = np.asarray(
        last_residual_action,
        dtype=np.float32,
    )
    if base_command.shape != (HIGH_LEVEL_RESIDUAL_SIZE,):
        raise ValueError("Base command must have shape (3,)")
    if last_residual_action.shape != (HIGH_LEVEL_RESIDUAL_SIZE,):
        raise ValueError("Last residual action must have shape (3,)")
    observation = np.concatenate(
        [
            env._get_observation(),
            base_command,
            last_residual_action,
        ]
    ).astype(np.float32)
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("The high-level residual observation is not finite")
    return observation


def apply_high_level_command_residual(
    base_command: np.ndarray,
    normalized_residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_command = np.asarray(base_command, dtype=np.float32)
    normalized_residual = np.asarray(
        normalized_residual,
        dtype=np.float32,
    )
    if base_command.shape != (HIGH_LEVEL_RESIDUAL_SIZE,):
        raise ValueError("Base command must have shape (3,)")
    if normalized_residual.shape != (HIGH_LEVEL_RESIDUAL_SIZE,):
        raise ValueError("Residual action must have shape (3,)")
    if not np.all(np.isfinite(base_command)) or not np.all(
        np.isfinite(normalized_residual)
    ):
        raise ValueError("Command and residual values must be finite")
    normalized_residual = np.clip(normalized_residual, -1.0, 1.0)
    physical_residual = (
        MAX_HIGH_LEVEL_COMMAND_RESIDUAL * normalized_residual
    )
    final_command = np.clip(
        base_command + physical_residual,
        COMMAND_LOW,
        COMMAND_HIGH,
    ).astype(np.float32)
    return final_command, physical_residual


class G1HighLevelKickResidualEnv(G1SoccerEnv):
    """Short-horizon command residual around an imminent ball contact."""

    def __init__(
        self,
        max_episode_steps: int = DEFAULT_HIGH_LEVEL_RESIDUAL_EPISODE_STEPS,
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
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(HIGH_LEVEL_RESIDUAL_SIZE,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(HIGH_LEVEL_RESIDUAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._base_command = np.array(
            [BASE_FORWARD_COMMAND, 0.0, 0.0],
            dtype=np.float32,
        )
        self._last_residual_action = np.zeros(
            HIGH_LEVEL_RESIDUAL_SIZE,
            dtype=np.float32,
        )
        self._ball_qvel_address = self.model.jnt_dofadr[
            self._ball_freejoint_id
        ]

    def _get_residual_observation(self) -> np.ndarray:
        return get_high_level_residual_observation(
            self,
            self._base_command,
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
        super().reset(seed=seed, options=options)
        self._move_ball_out_of_warmup_path()

        warmup_duration = self.np_random.uniform(*WARMUP_DURATION_RANGE)
        warmup_steps = round(warmup_duration / self.model.opt.timestep)
        goal, fell = self._advance_physics(
            self._base_command,
            warmup_steps,
        )
        if goal or fell:
            raise RuntimeError("The residual warm-up did not remain stable")

        contact_distance = self.np_random.uniform(*CONTACT_DISTANCE_RANGE)
        contact_lateral = self.np_random.uniform(*CONTACT_LATERAL_RANGE)
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
            "base_command": self._base_command.copy(),
        }
        return self._get_residual_observation(), info

    def step(self, action):
        final_command, physical_residual = (
            apply_high_level_command_residual(
                self._base_command,
                action,
            )
        )
        goal, fell = self._advance_physics(
            final_command,
            self.physics_steps_per_action,
        )
        self._elapsed_steps += 1
        self._last_command[:] = final_command
        self._last_residual_action[:] = np.clip(action, -1.0, 1.0)

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
            "base_command": self._base_command.copy(),
            "command": final_command.copy(),
            "residual": physical_residual.copy(),
            "elapsed_steps": self._elapsed_steps,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info
