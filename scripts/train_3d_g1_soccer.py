import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.soccer_3d import G1AdaptiveStartCurriculum, G1SoccerEnv
from src.soccer_3d.g1_soccer_env import (
    MAX_EPISODE_STEPS,
    OBSERVATION_MODES,
    REWARD_MODES,
    TASK_OBSERVATION_SIZE,
)


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_fixed.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a high-level PPO policy for fixed-start G1 soccer.",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float)
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
    parser.add_argument(
        "--transfer-from",
        type=Path,
        help=(
            "Initialize a wider observation policy from an existing model "
            "while initially ignoring the appended features."
        ),
    )
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="task",
    )
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="goal",
    )
    parser.add_argument("--adaptive-curriculum", action="store_true")
    parser.add_argument("--recovery-curriculum", action="store_true")
    parser.add_argument(
        "--initial-curriculum-difficulty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=MAX_EPISODE_STEPS,
    )
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.target_kl is not None and args.target_kl <= 0.0:
        parser.error("--target-kl must be positive")
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than 1")
    if args.resume is not None and not args.resume.exists():
        parser.error(f"model not found: {args.resume}")
    if args.transfer_from is not None and not args.transfer_from.exists():
        parser.error(f"model not found: {args.transfer_from}")
    if args.resume is not None and args.transfer_from is not None:
        parser.error("--resume and --transfer-from cannot be combined")
    if args.transfer_from is not None and args.observation_mode == "task":
        parser.error("--transfer-from requires an expanded observation mode")
    if args.adaptive_curriculum and args.recovery_curriculum:
        parser.error("Choose only one curriculum mode")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if not 0.0 <= args.initial_curriculum_difficulty <= 1.0:
        parser.error("--initial-curriculum-difficulty must be in [0, 1]")
    return args


def create_transferred_model(
    source_path: Path,
    env,
    rollout_steps: int,
    seed: int,
    learning_rate: float,
    target_kl: float | None,
) -> PPO:
    source_model = PPO.load(source_path)
    if source_model.observation_space.shape != (TASK_OBSERVATION_SIZE,):
        raise ValueError(
            "The transfer source must use the task observation shape "
            f"({TASK_OBSERVATION_SIZE},)"
        )
    if source_model.n_steps != rollout_steps:
        raise ValueError(
            "A transferred PPO model must keep its source rollout length "
            f"({source_model.n_steps})"
        )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=rollout_steps,
        seed=seed,
        verbose=1,
        learning_rate=learning_rate,
        target_kl=target_kl,
    )
    source_parameters = source_model.policy.state_dict()
    target_parameters = model.policy.state_dict()
    transferred_parameters = {}
    expanded_layers = []

    for name, target_value in target_parameters.items():
        source_value = source_parameters[name]
        if source_value.shape == target_value.shape:
            transferred_parameters[name] = source_value.clone()
            continue
        if (
            source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] < target_value.shape[1]
        ):
            expanded_value = target_value.clone()
            expanded_value.zero_()
            expanded_value[:, : source_value.shape[1]] = source_value
            transferred_parameters[name] = expanded_value
            expanded_layers.append(name)
            continue
        raise ValueError(
            f"Cannot transfer policy parameter {name}: "
            f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
        )

    model.policy.load_state_dict(transferred_parameters)
    model.num_timesteps = source_model.num_timesteps
    print(
        "Transferred the source policy and zero-initialized appended inputs "
        f"for: {', '.join(expanded_layers)}"
    )
    return model


def train(
    timesteps: int,
    seed: int,
    rollout_steps: int,
    output_path: Path,
    resume_path: Path | None,
    transfer_path: Path | None,
    observation_mode: str,
    reward_mode: str,
    adaptive_curriculum: bool,
    recovery_curriculum: bool,
    initial_curriculum_difficulty: float,
    max_episode_steps: int,
    learning_rate: float,
    target_kl: float | None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = G1SoccerEnv(
        render_mode=None,
        max_episode_steps=max_episode_steps,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    env = (
        G1AdaptiveStartCurriculum(
            base_env,
            recovery_start_curriculum=recovery_curriculum,
            initial_difficulty=initial_curriculum_difficulty,
        )
        if adaptive_curriculum or recovery_curriculum
        else base_env
    )
    try:
        if transfer_path is not None:
            model = create_transferred_model(
                transfer_path,
                env,
                rollout_steps,
                seed,
                learning_rate,
                target_kl,
            )
            reset_num_timesteps = False
        elif resume_path is None:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                n_steps=rollout_steps,
                seed=seed,
                verbose=1,
                learning_rate=learning_rate,
                target_kl=target_kl,
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
        if adaptive_curriculum or recovery_curriculum:
            print(
                "Curriculum finished at difficulty "
                f"{env.difficulty:.3f} after {env.completed_episodes} "
                f"episodes ({env.success_rate:.1%} total success)."
            )
        if recovery_curriculum:
            print(
                f"Recovery starts: {env.recovery_episodes}, "
                f"success rate {env.recovery_success_rate:.1%}, "
                "final probability "
                f"{env.recovery_start_probability:.1%}."
            )
    finally:
        env.close()


def main():
    args = parse_args()
    if args.recovery_curriculum:
        start_mode = "recovery curriculum"
    elif args.adaptive_curriculum:
        start_mode = "adaptive position curriculum"
    else:
        start_mode = "fixed"
    if args.transfer_from is not None:
        source = f"transferred from {args.transfer_from}"
    elif args.resume is not None:
        source = f"from {args.resume}"
    else:
        source = "from scratch"
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
        transfer_path=args.transfer_from,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        adaptive_curriculum=args.adaptive_curriculum,
        recovery_curriculum=args.recovery_curriculum,
        initial_curriculum_difficulty=(
            args.initial_curriculum_difficulty
        ),
        max_episode_steps=args.max_episode_steps,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
