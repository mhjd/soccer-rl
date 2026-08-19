import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.soccer_3d import CylinderSoccerEnv


DEFAULT_MODEL_PATH = Path("models/ppo_3d_cylinder_fixed.zip")
REWARD_STRATEGIES = (
    "combined",
    "contact_phased",
    "approach_warmup",
    "ball_goal_only",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on the fixed-goal 3D cylinder soccer task.",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reward-strategy",
        choices=REWARD_STRATEGIES,
        default="combined",
    )
    parser.add_argument(
        "--randomize-initial-positions",
        action="store_true",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="Environment steps collected before each PPO update phase.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than 1")

    return args


def train(
    timesteps,
    seed,
    rollout_steps,
    output_path,
    reward_strategy,
    randomize_initial_positions,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = CylinderSoccerEnv(
        render_mode=None,
        reward_strategy=reward_strategy,
        randomize_initial_positions=randomize_initial_positions,
    )

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
        f"Training fixed-goal PPO for {args.timesteps} timesteps "
        f"with seed {args.seed}, {args.reward_strategy} shaping, and "
        f"{'randomized' if args.randomize_initial_positions else 'fixed'} "
        "initial positions."
    )
    train(
        timesteps=args.timesteps,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        output_path=args.output,
        reward_strategy=args.reward_strategy,
        randomize_initial_positions=args.randomize_initial_positions,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
