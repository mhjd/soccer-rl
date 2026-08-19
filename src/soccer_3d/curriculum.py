import gymnasium as gym
import numpy as np

from .cylinder_env import (
    GOAL_LINE_Y,
    INITIAL_CLEARANCE,
    MAX_INITIAL_POSITION_ATTEMPTS,
    RANDOM_INITIAL_XY_HIGH,
    RANDOM_INITIAL_XY_LOW,
)


TARGET_SUCCESS_RATE = 0.5
DIFFICULTY_STEP = 0.02
EASY_BALL_GOAL_CLEARANCE = 0.35


class AdaptiveStartCurriculum(gym.Wrapper):
    """Adapt start-state difficulty toward a 50% episode success rate."""

    def __init__(
        self,
        env,
        target_success_rate=TARGET_SUCCESS_RATE,
        difficulty_step=DIFFICULTY_STEP,
    ):
        super().__init__(env)
        if not 0.0 < target_success_rate < 1.0:
            raise ValueError("target_success_rate must be between 0 and 1")
        if difficulty_step <= 0.0:
            raise ValueError("difficulty_step must be positive")

        self.target_success_rate = target_success_rate
        self.difficulty_step = difficulty_step
        self.difficulty = 0.0
        self.completed_episodes = 0
        self.successes = 0
        self._curriculum_rng = np.random.default_rng()

        ball_y = (
            GOAL_LINE_Y
            + self.env._ball_radius
            + EASY_BALL_GOAL_CLEARANCE
        )
        agent_y = (
            ball_y
            + self.env._agent_radius
            + self.env._ball_radius
            + INITIAL_CLEARANCE
        )
        self._easy_agent_xy = np.array([0.0, agent_y])
        self._easy_ball_xy = np.array([0.0, ball_y])

    @property
    def success_rate(self):
        if self.completed_episodes == 0:
            return 0.0
        return self.successes / self.completed_episodes

    def _sample_positions(self):
        minimum_separation = (
            self.env._agent_radius
            + self.env._ball_radius
            + INITIAL_CLEARANCE
        )
        for _ in range(MAX_INITIAL_POSITION_ATTEMPTS):
            final_agent_xy, final_ball_xy = self._curriculum_rng.uniform(
                low=RANDOM_INITIAL_XY_LOW,
                high=RANDOM_INITIAL_XY_HIGH,
                size=(2, 2),
            )
            agent_xy = self._easy_agent_xy + self.difficulty * (
                final_agent_xy - self._easy_agent_xy
            )
            ball_xy = self._easy_ball_xy + self.difficulty * (
                final_ball_xy - self._easy_ball_xy
            )
            if np.linalg.norm(agent_xy - ball_xy) >= minimum_separation:
                return agent_xy, ball_xy
        raise RuntimeError("Could not sample a valid curriculum start state")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._curriculum_rng = np.random.default_rng(seed)
        agent_xy, ball_xy = self._sample_positions()
        reset_options = {} if options is None else dict(options)
        reset_options["initial_xy_positions"] = (agent_xy, ball_xy)
        return self.env.reset(seed=seed, options=reset_options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        if terminated or truncated:
            success = float(info["goal"])
            self.completed_episodes += 1
            self.successes += int(success)
            adjustment = self.difficulty_step * (
                success - self.target_success_rate
            )
            self.difficulty = float(
                np.clip(self.difficulty + adjustment, 0.0, 1.0)
            )

        info["curriculum_difficulty"] = self.difficulty
        info["curriculum_success_rate"] = self.success_rate
        return observation, reward, terminated, truncated, info
