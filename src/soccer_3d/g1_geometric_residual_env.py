from __future__ import annotations

from gymnasium import Env, spaces
import numpy as np

from src.soccer_3d.g1_broad_pose import sample_broad_pose
from src.soccer_3d.g1_geometric_state_machine import (
    GEOMETRIC_PHASES,
    GeometricCommandStateMachine,
    physical_to_normalized,
)
from src.soccer_3d.g1_high_level_kick_residual_env import (
    HIGH_LEVEL_RESIDUAL_SIZE,
    apply_high_level_command_residual,
)
from src.soccer_3d.g1_soccer_env import (
    SOCCER_STATE_OBSERVATION_SIZE,
    G1SoccerEnv,
)


DEFAULT_AIM_Y_OFFSET = 0.25
GEOMETRIC_RESIDUAL_OBSERVATION_SIZE = (
    SOCCER_STATE_OBSERVATION_SIZE
    + HIGH_LEVEL_RESIDUAL_SIZE
    + len(GEOMETRIC_PHASES)
    + HIGH_LEVEL_RESIDUAL_SIZE
)


class G1GeometricResidualEnv(G1SoccerEnv):
    """Full geometric controller with a residual action at every step."""

    def __init__(
        self,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        aim_y_offset: float = DEFAULT_AIM_Y_OFFSET,
        disabled_residual_phases: tuple[str, ...] = (),
        hard_start_poses: list[dict] | None = None,
        hard_start_probability: float = 0.0,
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            randomize_initial_positions=False,
            recovery_start_probability=0.0,
            observation_mode="soccer_state",
            reward_mode="goal",
        )
        self.aim_y_offset = float(aim_y_offset)
        unknown_phases = set(disabled_residual_phases).difference(
            GEOMETRIC_PHASES
        )
        if unknown_phases:
            raise ValueError(
                "Unknown disabled residual phases: "
                + ", ".join(sorted(unknown_phases))
            )
        if not 0.0 <= hard_start_probability <= 1.0:
            raise ValueError("hard_start_probability must be in [0, 1]")
        self.disabled_residual_phases = frozenset(
            disabled_residual_phases
        )
        self.hard_start_poses = list(hard_start_poses or [])
        self.hard_start_probability = float(hard_start_probability)
        if self.hard_start_probability > 0.0 and not self.hard_start_poses:
            raise ValueError(
                "hard_start_poses cannot be empty when their probability "
                "is positive"
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
            shape=(GEOMETRIC_RESIDUAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._geometric_controller = None
        self._base_command = np.zeros(
            HIGH_LEVEL_RESIDUAL_SIZE,
            dtype=np.float32,
        )
        self._last_residual_action = np.zeros(
            HIGH_LEVEL_RESIDUAL_SIZE,
            dtype=np.float32,
        )

    def _phase_one_hot(self) -> np.ndarray:
        phase = self._geometric_controller.phase
        phase_one_hot = np.zeros(len(GEOMETRIC_PHASES), dtype=np.float32)
        phase_one_hot[GEOMETRIC_PHASES.index(phase)] = 1.0
        return phase_one_hot

    def _get_residual_observation(self) -> np.ndarray:
        observation = np.concatenate(
            [
                self._get_observation(),
                self._base_command,
                self._phase_one_hot(),
                self._last_residual_action,
            ]
        ).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                "The geometric residual observation is not finite"
            )
        return observation

    def reset(self, seed=None, options=None):
        Env.reset(self, seed=seed)
        options = {} if options is None else dict(options)
        explicit_pose = "initial_g1_xy" in options
        hard_start = False
        if explicit_pose:
            required = {
                "initial_g1_xy",
                "initial_ball_xy",
                "initial_g1_yaw",
            }
            missing = required.difference(options)
            if missing:
                raise ValueError(
                    "Explicit geometric reset is missing: "
                    + ", ".join(sorted(missing))
                )
        else:
            use_hard_start = (
                self.hard_start_poses
                and self.np_random.random() < self.hard_start_probability
            )
            if use_hard_start:
                hard_start = True
                pose = self.hard_start_poses[
                    int(self.np_random.integers(len(self.hard_start_poses)))
                ]
                g1_xy = np.asarray(pose["initial_g1_xy"])
                ball_xy = np.asarray(pose["initial_ball_xy"])
                g1_yaw = float(pose["initial_g1_yaw"])
            else:
                g1_xy, ball_xy, g1_yaw = sample_broad_pose(
                    self.np_random,
                    self.aim_y_offset,
                )
            options.update(
                {
                    "initial_g1_xy": g1_xy,
                    "initial_ball_xy": ball_xy,
                    "initial_g1_yaw": g1_yaw,
                }
            )

        _, info = G1SoccerEnv.reset(self, seed=None, options=options)
        self._geometric_controller = GeometricCommandStateMachine(
            self,
            aim_y_offset=self.aim_y_offset,
        )
        base_command = self._geometric_controller.next_command()
        if base_command is None:
            raise RuntimeError(
                "The geometric controller produced no initial command"
            )
        self._base_command[:] = base_command
        self._last_residual_action.fill(0.0)
        info = dict(info)
        info.update(
            {
                "geometric_phase": self._geometric_controller.phase,
                "base_command": self._base_command.copy(),
                "hard_start": hard_start,
            }
        )
        return self._get_residual_observation(), info

    def step(self, action):
        executed_phase = self._geometric_controller.phase
        applied_action = np.asarray(action, dtype=np.float32)
        if executed_phase in self.disabled_residual_phases:
            applied_action = np.zeros(
                HIGH_LEVEL_RESIDUAL_SIZE,
                dtype=np.float32,
            )
        final_command, physical_residual = (
            apply_high_level_command_residual(
                self._base_command,
                applied_action,
            )
        )
        observation, reward, terminated, truncated, info = (
            G1SoccerEnv.step(
                self,
                physical_to_normalized(final_command),
            )
        )
        self._last_residual_action[:] = np.clip(
            applied_action,
            -1.0,
            1.0,
        )

        controller_exhausted = False
        if not (terminated or truncated):
            next_command = self._geometric_controller.next_command()
            if next_command is None:
                controller_exhausted = True
                truncated = True
                self._base_command.fill(0.0)
            else:
                self._base_command[:] = next_command

        info = dict(info)
        info.update(
            {
                "geometric_phase": executed_phase,
                "next_geometric_phase": self._geometric_controller.phase,
                "base_command": self._base_command.copy(),
                "residual": physical_residual.copy(),
                "controller_exhausted": controller_exhausted,
            }
        )
        if terminated or truncated:
            observation = self._get_observation()
            phase = np.zeros(len(GEOMETRIC_PHASES), dtype=np.float32)
            phase[
                GEOMETRIC_PHASES.index(self._geometric_controller.phase)
            ] = 1.0
            observation = np.concatenate(
                [
                    observation,
                    self._base_command,
                    phase,
                    self._last_residual_action,
                ]
            ).astype(np.float32)
        else:
            observation = self._get_residual_observation()
        return observation, reward, terminated, truncated, info
