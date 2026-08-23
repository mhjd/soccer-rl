import gymnasium as gym
import numpy as np


TARGET_SUCCESS_RATE = 0.5
DIFFICULTY_STEP = 0.02
MAX_RECOVERY_START_PROBABILITY = 0.5


class G1AdaptiveStartCurriculum(gym.Wrapper):
    """Adapt G1 start-position variation toward a 50% success rate."""

    def __init__(
        self,
        env,
        target_success_rate: float = TARGET_SUCCESS_RATE,
        difficulty_step: float = DIFFICULTY_STEP,
        recovery_start_curriculum: bool = False,
        max_recovery_start_probability: float = (
            MAX_RECOVERY_START_PROBABILITY
        ),
        initial_difficulty: float = 0.0,
    ):
        super().__init__(env)
        if not 0.0 < target_success_rate < 1.0:
            raise ValueError("target_success_rate must be between 0 and 1")
        if difficulty_step <= 0.0:
            raise ValueError("difficulty_step must be positive")
        if not 0.0 <= max_recovery_start_probability <= 1.0:
            raise ValueError(
                "max_recovery_start_probability must be in [0, 1]"
            )
        if not 0.0 <= initial_difficulty <= 1.0:
            raise ValueError("initial_difficulty must be in [0, 1]")

        self.target_success_rate = target_success_rate
        self.difficulty_step = difficulty_step
        self.recovery_start_curriculum = recovery_start_curriculum
        self.max_recovery_start_probability = (
            max_recovery_start_probability
        )
        self.difficulty = initial_difficulty
        self.completed_episodes = 0
        self.successes = 0
        self.recovery_episodes = 0
        self.recovery_successes = 0
        self._current_recovery_start = False

    @property
    def success_rate(self) -> float:
        if self.completed_episodes == 0:
            return 0.0
        return self.successes / self.completed_episodes

    @property
    def recovery_success_rate(self) -> float:
        if self.recovery_episodes == 0:
            return 0.0
        return self.recovery_successes / self.recovery_episodes

    @property
    def recovery_start_probability(self) -> float:
        if not self.recovery_start_curriculum:
            return 0.0
        return self.max_recovery_start_probability * self.difficulty

    def reset(self, *, seed=None, options=None):
        reset_options = {} if options is None else dict(options)
        reset_options["initial_state_difficulty"] = (
            1.0 if self.recovery_start_curriculum else self.difficulty
        )
        reset_options["recovery_start_probability"] = (
            self.recovery_start_probability
        )
        reset_options["recovery_state_difficulty"] = self.difficulty
        observation, info = self.env.reset(seed=seed, options=reset_options)
        self._current_recovery_start = bool(info["recovery_start"])
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        if terminated or truncated:
            success = float(info["goal"])
            self.completed_episodes += 1
            self.successes += int(success)
            if self._current_recovery_start:
                self.recovery_episodes += 1
                self.recovery_successes += int(success)
            adjustment = self.difficulty_step * (
                success - self.target_success_rate
            )
            self.difficulty = float(
                np.clip(self.difficulty + adjustment, 0.0, 1.0)
            )

        info["curriculum_difficulty"] = self.difficulty
        info["curriculum_success_rate"] = self.success_rate
        info["curriculum_recovery_start_probability"] = (
            self.recovery_start_probability
        )
        info["curriculum_recovery_success_rate"] = (
            self.recovery_success_rate
        )
        return observation, reward, terminated, truncated, info
