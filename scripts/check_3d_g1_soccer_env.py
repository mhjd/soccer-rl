import argparse

from gymnasium.utils.env_checker import check_env
import numpy as np

from src.soccer_3d.g1_soccer_env import (
    CONTROL_TIMESTEP,
    FIXED_BALL_XY,
    FIXED_G1_XY,
    G1SoccerEnv,
    MINIMUM_INITIAL_SEPARATION,
    MINIMUM_WALKING_TRANSLATION,
    RANDOM_BALL_XY_HIGH,
    RANDOM_BALL_XY_LOW,
    RECOVERY_G1_LATERAL_DISTANCE,
    RECOVERY_G1_X_OFFSET,
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

    stand_command = normalized_action_to_command(
        np.array([0.01, 0.01, 1.0], dtype=np.float32)
    )
    if not np.allclose(stand_command, 0.0):
        raise RuntimeError("The walking deadband did not select stand")

    weak_action = np.array([-0.2, -0.2, 1.0], dtype=np.float32)
    walking_command = normalized_action_to_command(weak_action)
    if not np.isclose(
        np.max(
            np.abs(walking_command[:2])
            / MINIMUM_WALKING_TRANSLATION
        ),
        1.0,
    ):
        raise RuntimeError("A weak walking action stayed in the dead zone")
    if not np.isclose(walking_command[2], 0.2):
        raise RuntimeError("Walking activation changed the yaw command")


def check_gymnasium_contract():
    for observation_mode, reward_mode in (
        ("task", "goal"),
        ("soccer_state", "approach_progress"),
    ):
        env = G1SoccerEnv(
            max_episode_steps=2,
            observation_mode=observation_mode,
            reward_mode=reward_mode,
        )
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
        if info_a["recovery_start"] or info_b["recovery_start"]:
            raise RuntimeError("Regular randomization produced a recovery start")

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


def check_recovery_resets():
    env = G1SoccerEnv(
        randomize_initial_positions=True,
        recovery_start_probability=1.0,
    )
    try:
        sampled_positions = []
        for seed in range(20):
            _, info = env.reset(seed=seed)
            if not info["recovery_start"]:
                raise RuntimeError("Forced recovery reset used a regular start")

            g1_xy = info["initial_g1_xy"]
            ball_xy = info["initial_ball_xy"]
            relative_xy = g1_xy - ball_xy
            if not RECOVERY_G1_X_OFFSET[0] <= relative_xy[0] <= (
                RECOVERY_G1_X_OFFSET[1]
            ):
                raise RuntimeError("Recovery X offset is out of range")
            if not RECOVERY_G1_LATERAL_DISTANCE[0] <= abs(
                relative_xy[1]
            ) <= RECOVERY_G1_LATERAL_DISTANCE[1]:
                raise RuntimeError("Recovery lateral offset is out of range")
            if np.linalg.norm(relative_xy) < MINIMUM_INITIAL_SEPARATION:
                raise RuntimeError("Recovery positions are too close")
            sampled_positions.append(np.concatenate([g1_xy, ball_xy]))

        if np.allclose(sampled_positions[0], sampled_positions[1]):
            raise RuntimeError("Different seeds produced identical recoveries")
    finally:
        env.close()


def check_explicit_initial_pose():
    env = G1SoccerEnv()
    try:
        requested_g1_xy = np.array([1.8, -0.8])
        requested_ball_xy = np.array([0.6, 0.4])
        requested_yaw = 1.2
        _, info = env.reset(
            seed=0,
            options={
                "initial_g1_xy": requested_g1_xy,
                "initial_ball_xy": requested_ball_xy,
                "initial_g1_yaw": requested_yaw,
            },
        )
        if not np.allclose(info["initial_g1_xy"], requested_g1_xy):
            raise RuntimeError("Explicit G1 XY was not preserved")
        if not np.allclose(info["initial_ball_xy"], requested_ball_xy):
            raise RuntimeError("Explicit ball XY was not preserved")
        if not np.isclose(info["initial_g1_yaw"], requested_yaw):
            raise RuntimeError("Explicit G1 yaw was not preserved")

        pelvis_rotation = env.data.xmat[env.controller.pelvis_id].reshape(3, 3)
        actual_yaw = np.arctan2(
            pelvis_rotation[1, 0],
            pelvis_rotation[0, 0],
        )
        yaw_error = (actual_yaw - requested_yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(yaw_error) > 0.05:
            raise RuntimeError("Settled G1 yaw differs from the requested yaw")
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

    recovery_base_env = G1SoccerEnv(max_episode_steps=1)
    recovery_env = G1AdaptiveStartCurriculum(
        recovery_base_env,
        recovery_start_curriculum=True,
    )
    try:
        recovery_env.difficulty = 0.8
        _, reset_info = recovery_env.reset(seed=0)
        if not np.isclose(reset_info["initial_state_difficulty"], 1.0):
            raise RuntimeError("Recovery curriculum reduced position coverage")
        if not np.isclose(reset_info["recovery_start_probability"], 0.4):
            raise RuntimeError("Recovery probability did not track difficulty")
        if not np.isclose(reset_info["recovery_state_difficulty"], 0.8):
            raise RuntimeError("Recovery geometry did not track difficulty")
        _, _, _, truncated, info = recovery_env.step(
            np.zeros(3, dtype=np.float32)
        )
        if not truncated:
            raise RuntimeError("Recovery curriculum check did not truncate")
        if not np.isclose(
            info["curriculum_recovery_start_probability"],
            0.395,
        ):
            raise RuntimeError("Recovery probability did not follow update")
    finally:
        recovery_env.close()


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
    check_recovery_resets()
    check_explicit_initial_pose()
    check_adaptive_curriculum()
    check_scripted_goal(args.render)
    print("g1 soccer environment: passed")


if __name__ == "__main__":
    main()
