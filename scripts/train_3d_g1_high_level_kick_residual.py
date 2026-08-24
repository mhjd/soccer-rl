import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.soccer_3d import G1HighLevelKickResidualEnv


DEFAULT_OUTPUT = Path("models/ppo_3d_g1_high_level_kick_residual.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a high-level command residual for G1 ball contact."
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = G1HighLevelKickResidualEnv()
    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=args.rollout_steps,
            seed=args.seed,
            verbose=1,
            learning_rate=args.learning_rate,
            policy_kwargs={"log_std_init": -2.0},
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()
    print(f"Saved high-level residual policy to {args.output}")


if __name__ == "__main__":
    main()
