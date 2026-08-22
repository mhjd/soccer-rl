import argparse

from gymnasium.utils.env_checker import check_env
import numpy as np

from src.soccer_3d.g1_soccer_env import (
    CONTROL_TIMESTEP,
    FIXED_BALL_XY,
    FIXED_G1_XY,
    G1SoccerEnv,
    RANDOM_BALL_XY_HIGH,
    RANDOM_BALL_XY_LOW,
    normalized_action_to_command,
)
from src.soccer_3d.g1_curriculum import G1AdaptiveStartCurriculum


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check the fixed-start G1 soccer environment.",
    )
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def check_action_mapping():
    actions = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [-0.5, -0.3, -0.2],
            [-0.25, -0.15, -0.1],
            [0.0, 0.0, 0.0],
            [0.5, 0.15, 0.1],
            [1.0, 0.3, 0.2],
        ],
        dtype=np.float32,
    )
    actual = np.stack(
        [normalized_action_to_command(action) for action in actions]
    )
    if not np.allclose(actual, expected):
        raise RuntimeError(
            f"Unexpected normalized-action mapping:\n{actual}"
        )


def check_gymnasium_contract():
    env = G1SoccerEnv(max_episode_steps=2)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def check_control_timestep():
    env = G1SoccerEnv(max_episode_steps=2)
    try:
        observation, _ = env.reset(seed=0)
        if not env.observation_space.contains(observation):
            raise RuntimeError("Reset returned an invalid observation")
        initial_time = env.data.time
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        elapsed_time = env.data.time - initial_time
        if not np.isclose(elapsed_time, CONTROL_TIMESTEP):
            raise RuntimeError(
                f"One action advanced {elapsed_time:.6f} s instead of "
                f"{CONTROL_TIMESTEP:.6f} s"
            )
        if reward != 0.0 or terminated or truncated:
            raise RuntimeError("Neutral first step ended the episode")
        if info["goal"] or info["fell"]:
            raise RuntimeError("Neutral first step reported a terminal event")
        if not env.observation_space.contains(observation):
            raise RuntimeError("Step returned an invalid observation")
    finally:
        env.close()


def check_truncation():
    env = G1SoccerEnv(max_episode_steps=1)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(
            np.zeros(3, dtype=np.float32)
        )
        if terminated or not truncated:
            raise RuntimeError("Episode horizon did not cause truncation")
    finally:
        env.close()


def check_randomized_resets():
    env = G1SoccerEnv(randomize_initial_positions=True)
    try:
        observation_a, info_a = env.reset(seed=123)
        observation_b, info_b = env.reset(seed=123)
        if not np.allclose(observation_a, observation_b):
            raise RuntimeError("Equal reset seeds produced different states")
        if not np.allclose(info_a["initial_g1_xy"], info_b["initial_g1_xy"]):
            raise RuntimeError("Equal seeds produced different G1 positions")
        if not np.allclose(
            info_a["initial_ball_xy"],
            info_b["initial_ball_xy"],
        ):
            raise RuntimeError("Equal seeds produced different ball positions")

        sampled_positions = []
        for seed in range(20):
            _, info = env.reset(seed=seed)
            g1_xy = info["initial_g1_xy"]
            ball_xy = info["initial_ball_xy"]
            if g1_xy[0] >= ball_xy[0]:
                raise RuntimeError("Randomized G1 did not start behind ball")
            if np.any(ball_xy < RANDOM_BALL_XY_LOW) or np.any(
                ball_xy > RANDOM_BALL_XY_HIGH
            ):
                raise RuntimeError("Randomized ball position is out of range")
            sampled_positions.append(np.concatenate([g1_xy, ball_xy]))
        if np.allclose(sampled_positions[0], sampled_positions[1]):
            raise RuntimeError("Different seeds produced the same positions")

        _, fixed_info = env.reset(
            seed=0,
            options={"initial_state_difficulty": 0.0},
        )
        if not np.allclose(fixed_info["initial_g1_xy"], FIXED_G1_XY):
            raise RuntimeError("Difficulty zero changed the fixed G1 position")
        if not np.allclose(fixed_info["initial_ball_xy"], FIXED_BALL_XY):
            raise RuntimeError("Difficulty zero changed the fixed ball position")
    finally:
        env.close()


def check_adaptive_curriculum():
    base_env = G1SoccerEnv(max_episode_steps=1)
    env = G1AdaptiveStartCurriculum(base_env)
    try:
        env.difficulty = 0.5
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        if terminated or not truncated:
            raise RuntimeError("Curriculum failure check did not truncate")
        if not np.isclose(env.difficulty, 0.49):
            raise RuntimeError("Curriculum did not lower failed difficulty")
        if not np.isclose(info["curriculum_difficulty"], 0.49):
            raise RuntimeError("Curriculum info reported wrong difficulty")
    finally:
        env.close()


def check_scripted_goal(render: bool):
    env = G1SoccerEnv(
        max_episode_steps=100,
        render_mode="human" if render else None,
    )
    try:
        env.reset(seed=0)
        action = np.array([0.6, 0.0, 0.0], dtype=np.float32)
        info = {}
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(env.max_episode_steps):
            _, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        if not terminated or truncated:
            raise RuntimeError("Scripted command did not terminate on a goal")
        if not info["goal"] or info["fell"]:
            raise RuntimeError(f"Unexpected terminal result: {info}")
        if not info["ball_contact_occurred"]:
            raise RuntimeError("Goal occurred without recorded foot contact")
        if reward != 1.0:
            raise RuntimeError(f"Goal reward was {reward}, expected 1.0")
        print(
            "scripted goal: "
            f"{info['elapsed_steps']} high-level steps, "
            f"simulation time {env.data.time:.3f} s"
        )
    finally:
        env.close()


def main():
    args = parse_args()
    check_action_mapping()
    check_gymnasium_contract()
    check_control_timestep()
    check_truncation()
    check_randomized_resets()
    check_adaptive_curriculum()
    check_scripted_goal(args.render)
    print("g1 soccer environment: passed")


if __name__ == "__main__":
    main()
