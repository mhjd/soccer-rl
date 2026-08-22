import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.soccer_3d import G1SoccerEnv


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_fixed.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a high-level PPO policy for fixed-start G1 soccer.",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="High-level environment steps collected before each PPO update.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than 1")
    return args


def train(timesteps: int, seed: int, rollout_steps: int, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = G1SoccerEnv(render_mode=None)
    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=rollout_steps,
            seed=seed,
            verbose=1,
        )
        model.learn(total_timesteps=timesteps)
        model.save(output_path)
    finally:
        env.close()


def main():
    args = parse_args()
    print(
        "Training the fixed-start high-level G1 soccer policy for "
        f"{args.timesteps} timesteps with seed {args.seed}."
    )
    train(
        timesteps=args.timesteps,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        output_path=args.output,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
