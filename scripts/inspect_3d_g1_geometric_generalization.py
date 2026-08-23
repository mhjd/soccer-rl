import argparse
import time

import numpy as np

from scripts.evaluate_3d_g1_geometric_controller import (
    GeometricTrace,
    PilotTools,
    initial_episode_info,
)
from scripts.evaluate_3d_g1_geometric_generalization import (
    episode_diagnostics,
    position_category,
    sample_broad_pose,
)
from src.soccer_3d import G1SoccerEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one reproducible broad geometric-controller case."
    )
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aim-y-offset", type=float, default=0.25)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("--episode must be non-negative")
    return args


def sampled_pose(seed, episode, aim_y_offset):
    rng = np.random.default_rng(seed)
    pose = None
    for _ in range(episode + 1):
        pose = sample_broad_pose(rng, aim_y_offset)
    return pose


def main():
    args = parse_args()
    g1_xy, ball_xy, g1_yaw = sampled_pose(
        args.seed,
        args.episode,
        args.aim_y_offset,
    )
    env = G1SoccerEnv(
        render_mode="human",
        observation_mode="soccer_state",
        max_episode_steps=args.max_episode_steps,
    )
    try:
        observation, _ = env.reset(
            seed=args.seed + args.episode,
            options={
                "initial_g1_xy": g1_xy,
                "initial_ball_xy": ball_xy,
                "initial_g1_yaw": g1_yaw,
            },
        )
        trace = GeometricTrace()
        pilot = PilotTools(
            env,
            observation,
            initial_episode_info(),
            aim_y_offset=args.aim_y_offset,
            verbose=True,
            trace=trace,
        )
        reached, aligned, info = pilot.solve()
        category = position_category(
            g1_xy,
            ball_xy,
            args.aim_y_offset,
        )
        diagnostic = episode_diagnostics(
            args.episode,
            g1_xy,
            ball_xy,
            g1_yaw,
            category,
            reached,
            aligned,
            info,
            trace,
        )
        print(
            f"episode={args.episode} "
            f"signature={diagnostic['signature']} "
            f"goal={diagnostic['goal']} "
            f"steps={diagnostic['elapsed_steps']}",
            flush=True,
        )
        time.sleep(2.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
