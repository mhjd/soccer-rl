import argparse

import numpy as np
from gymnasium.utils.env_checker import check_env

from scripts.evaluate_3d_g1_geometric_controller import (
    PilotTools,
    initial_episode_info,
)
from src.soccer_3d import G1GeometricResidualEnv, G1SoccerEnv
from src.soccer_3d.g1_broad_pose import sample_broad_pose
from src.soccer_3d.g1_soccer_env import normalized_action_to_command


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check stepwise zero-residual equivalence with the blocking "
            "geometric controller."
        )
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main():
    args = parse_args()
    aim_y_offset = 0.25
    pose_rng = np.random.default_rng(args.seed)
    blocking_env = G1SoccerEnv(
        observation_mode="soccer_state",
        max_episode_steps=500,
    )
    stepwise_env = G1GeometricResidualEnv(
        aim_y_offset=aim_y_offset,
        max_episode_steps=500,
    )
    zero_residual = np.zeros(3, dtype=np.float32)
    try:
        check_env(stepwise_env, skip_render_check=True)
        for episode_index in range(args.episodes):
            g1_xy, ball_xy, g1_yaw = sample_broad_pose(
                pose_rng,
                aim_y_offset,
            )
            options = {
                "initial_g1_xy": g1_xy,
                "initial_ball_xy": ball_xy,
                "initial_g1_yaw": g1_yaw,
            }

            observation, _ = blocking_env.reset(
                seed=args.seed + episode_index,
                options=options,
            )
            blocking_actions = []
            pilot = PilotTools(
                blocking_env,
                observation,
                initial_episode_info(),
                aim_y_offset=aim_y_offset,
                recorded_actions=blocking_actions,
            )
            reached, aligned, blocking_info = pilot.solve()

            _, _ = stepwise_env.reset(
                seed=args.seed + episode_index,
                options=options,
            )
            stepwise_actions = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, stepwise_info = (
                    stepwise_env.step(zero_residual)
                )
                stepwise_actions.append(
                    stepwise_env._last_command.copy()
                )

            if len(blocking_actions) != len(stepwise_actions):
                raise AssertionError(
                    "Stepwise and blocking action counts differ: "
                    f"{len(stepwise_actions)} != {len(blocking_actions)}"
                )
            for blocking_action, stepwise_command in zip(
                blocking_actions,
                stepwise_actions,
            ):
                np.testing.assert_array_equal(
                    normalized_action_to_command(blocking_action),
                    stepwise_command,
                )

            if (
                reached
                != stepwise_env._geometric_controller.reached_approach
                or aligned
                != stepwise_env._geometric_controller.aligned_shot
            ):
                raise AssertionError("Geometric completion flags differ")
            for key in (
                "goal",
                "fell",
                "ball_contact_occurred",
                "elapsed_steps",
            ):
                if blocking_info[key] != stepwise_info[key]:
                    raise AssertionError(
                        f"Episode {episode_index} differs for {key}: "
                        f"{blocking_info[key]} != {stepwise_info[key]}"
                    )
            np.testing.assert_array_equal(
                blocking_env.data.qpos,
                stepwise_env.data.qpos,
            )
            np.testing.assert_array_equal(
                blocking_env.data.qvel,
                stepwise_env.data.qvel,
            )
    finally:
        blocking_env.close()
        stepwise_env.close()

    print(
        "Stepwise zero-residual controller exactly matched "
        f"{args.episodes} blocking-controller episodes"
    )


if __name__ == "__main__":
    main()
