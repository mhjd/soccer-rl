import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.soccer_3d import G1GeometricResidualEnv
from src.soccer_3d.g1_geometric_state_machine import GEOMETRIC_PHASES


DEFAULT_OUTPUT = Path("models/ppo_3d_g1_geometric_residual.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a bounded residual over every geometric-controller phase."
        )
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--disable-residual-phase",
        action="append",
        choices=GEOMETRIC_PHASES,
        default=[],
    )
    parser.add_argument(
        "--hard-start-report",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--hard-start-probability", type=float, default=0.0)
    parser.add_argument("--checkpoint-frequency", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    if not 0.0 <= args.hard_start_probability <= 1.0:
        parser.error("--hard-start-probability must be in [0, 1]")
    if args.hard_start_probability > 0.0 and not args.hard_start_report:
        parser.error(
            "--hard-start-report is required when hard starts are enabled"
        )
    if args.checkpoint_frequency <= 0:
        parser.error("--checkpoint-frequency must be positive")
    for report_path in args.hard_start_report:
        if not report_path.exists():
            parser.error(f"report not found: {report_path}")
    return args


def load_failed_poses(report_paths):
    failed_poses = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        for episode in report["episode_results"]:
            if not episode["goal"]:
                failed_poses.append(
                    {
                        "initial_g1_xy": episode["initial_g1_xy"],
                        "initial_ball_xy": episode["initial_ball_xy"],
                        "initial_g1_yaw": episode["initial_g1_yaw"],
                    }
                )
    if report_paths and not failed_poses:
        raise ValueError("Hard-start reports contain no failed episodes")
    return failed_poses


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hard_start_poses = load_failed_poses(args.hard_start_report)
    env = G1GeometricResidualEnv(
        disabled_residual_phases=tuple(args.disable_residual_phase),
        hard_start_poses=hard_start_poses,
        hard_start_probability=args.hard_start_probability,
    )
    checkpoint_directory = args.output.parent / (
        args.output.stem + "_checkpoints"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=checkpoint_directory,
        name_prefix=args.output.stem,
    )
    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=args.rollout_steps,
            seed=args.seed,
            verbose=1,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            policy_kwargs={"log_std_init": -2.0},
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
        )
        model.save(args.output)
    finally:
        env.close()
    print(f"Hard-start poses: {len(hard_start_poses)}")
    print(f"Saved full geometric residual policy to {args.output}")


if __name__ == "__main__":
    main()
