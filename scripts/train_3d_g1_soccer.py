import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.soccer_3d import G1AdaptiveStartCurriculum, G1SoccerEnv


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
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue training from an existing high-level PPO model.",
    )
    parser.add_argument("--adaptive-curriculum", action="store_true")
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than 1")
    if args.resume is not None and not args.resume.exists():
        parser.error(f"model not found: {args.resume}")
    return args


def train(
    timesteps: int,
    seed: int,
    rollout_steps: int,
    output_path: Path,
    resume_path: Path | None,
    adaptive_curriculum: bool,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = G1SoccerEnv(render_mode=None)
    env = (
        G1AdaptiveStartCurriculum(base_env)
        if adaptive_curriculum
        else base_env
    )
    try:
        if resume_path is None:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                n_steps=rollout_steps,
                seed=seed,
                verbose=1,
            )
            reset_num_timesteps = True
        else:
            model = PPO.load(resume_path, env=env)
            if model.n_steps != rollout_steps:
                raise ValueError(
                    "A resumed PPO model must keep its original rollout "
                    f"length ({model.n_steps})"
                )
            model.set_random_seed(seed)
            reset_num_timesteps = False

        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(output_path)
        if adaptive_curriculum:
            print(
                "Curriculum finished at difficulty "
                f"{env.difficulty:.3f} after {env.completed_episodes} "
                f"episodes ({env.success_rate:.1%} total success)."
            )
    finally:
        env.close()


def main():
    args = parse_args()
    start_mode = "adaptive curriculum" if args.adaptive_curriculum else "fixed"
    source = "from scratch" if args.resume is None else f"from {args.resume}"
    print(
        f"Training the {start_mode} high-level G1 soccer policy "
        f"{source} for {args.timesteps} timesteps with seed {args.seed}."
    )
    train(
        timesteps=args.timesteps,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        output_path=args.output,
        resume_path=args.resume,
        adaptive_curriculum=args.adaptive_curriculum,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
