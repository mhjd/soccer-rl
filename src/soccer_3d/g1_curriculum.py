import gymnasium as gym
import numpy as np


TARGET_SUCCESS_RATE = 0.5
DIFFICULTY_STEP = 0.02


class G1AdaptiveStartCurriculum(gym.Wrapper):
    """Adapt G1 start-position variation toward a 50% success rate."""

    def __init__(
        self,
        env,
        target_success_rate: float = TARGET_SUCCESS_RATE,
        difficulty_step: float = DIFFICULTY_STEP,
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

    @property
    def success_rate(self) -> float:
        if self.completed_episodes == 0:
            return 0.0
        return self.successes / self.completed_episodes

    def reset(self, *, seed=None, options=None):
        reset_options = {} if options is None else dict(options)
        reset_options["initial_state_difficulty"] = self.difficulty
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
