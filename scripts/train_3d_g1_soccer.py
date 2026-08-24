import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from src.soccer_3d import (
    G1AdaptiveBroadCurriculum,
    G1AdaptiveStartCurriculum,
    G1SoccerEnv,
)
from src.soccer_3d.g1_soccer_env import (
    MAX_EPISODE_STEPS,
    OBSERVATION_MODES,
    REWARD_MODES,
    TASK_OBSERVATION_SIZE,
)


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_fixed.zip")
ALGORITHMS = ("ppo", "sac")
SAC_BUFFER_SIZE = 200_000
SAC_LEARNING_STARTS = 5_000
SAC_BATCH_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a high-level policy for G1 soccer.",
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="ppo")
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
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=0,
        help="Save a checkpoint every N environment steps; 0 disables it.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue training from an existing high-level model.",
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
    parser.add_argument("--randomize-initial-positions", action="store_true")
    parser.add_argument("--adaptive-curriculum", action="store_true")
    parser.add_argument("--recovery-curriculum", action="store_true")
    parser.add_argument("--broad-curriculum", action="store_true")
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
    if args.checkpoint_frequency < 0:
        parser.error("--checkpoint-frequency cannot be negative")
    if args.resume is not None and not args.resume.exists():
        parser.error(f"model not found: {args.resume}")
    if args.transfer_from is not None and not args.transfer_from.exists():
        parser.error(f"model not found: {args.transfer_from}")
    if args.resume is not None and args.transfer_from is not None:
        parser.error("--resume and --transfer-from cannot be combined")
    if args.algorithm != "ppo" and args.transfer_from is not None:
        parser.error("--transfer-from currently supports PPO only")
    if args.algorithm != "ppo" and args.target_kl is not None:
        parser.error("--target-kl is a PPO-only option")
    if args.transfer_from is not None and args.observation_mode == "task":
        parser.error("--transfer-from requires an expanded observation mode")
    curriculum_modes = sum(
        (
            args.adaptive_curriculum,
            args.recovery_curriculum,
            args.broad_curriculum,
        )
    )
    if curriculum_modes > 1:
        parser.error("Choose only one curriculum mode")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if not 0.0 <= args.initial_curriculum_difficulty <= 1.0:
        parser.error("--initial-curriculum-difficulty must be in [0, 1]")
    if args.output is None:
        args.output = Path(
            f"models/{args.algorithm}_3d_g1_soccer_fixed.zip"
        )
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
    algorithm: str,
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
    broad_curriculum: bool,
    initial_curriculum_difficulty: float,
    max_episode_steps: int,
    randomize_initial_positions: bool,
    learning_rate: float,
    target_kl: float | None,
    checkpoint_frequency: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = G1SoccerEnv(
        render_mode=None,
        max_episode_steps=max_episode_steps,
        randomize_initial_positions=randomize_initial_positions,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    if broad_curriculum:
        env = G1AdaptiveBroadCurriculum(
            base_env,
            initial_difficulty=initial_curriculum_difficulty,
        )
    elif adaptive_curriculum or recovery_curriculum:
        env = G1AdaptiveStartCurriculum(
            base_env,
            recovery_start_curriculum=recovery_curriculum,
            initial_difficulty=initial_curriculum_difficulty,
        )
    else:
        env = base_env
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
        elif resume_path is None and algorithm == "ppo":
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
        elif resume_path is None:
            model = SAC(
                policy="MlpPolicy",
                env=env,
                seed=seed,
                verbose=1,
                learning_rate=learning_rate,
                buffer_size=SAC_BUFFER_SIZE,
                learning_starts=SAC_LEARNING_STARTS,
                batch_size=SAC_BATCH_SIZE,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
            )
            reset_num_timesteps = True
        else:
            model_class = PPO if algorithm == "ppo" else SAC
            model = model_class.load(resume_path, env=env)
            if algorithm == "ppo" and model.n_steps != rollout_steps:
                raise ValueError(
                    "A resumed PPO model must keep its original rollout "
                    f"length ({model.n_steps})"
                )
            if algorithm == "sac":
                replay_buffer_path = resume_path.with_suffix(
                    ".replay_buffer.pkl"
                )
                if replay_buffer_path.exists():
                    model.load_replay_buffer(replay_buffer_path)
                    print(f"Loaded replay buffer from {replay_buffer_path}")
                else:
                    print(
                        "No replay buffer found next to the resumed SAC "
                        "model; continued with an empty buffer."
                    )
            model.set_random_seed(seed)
            reset_num_timesteps = False

        checkpoint_callback = None
        if checkpoint_frequency:
            checkpoint_directory = output_path.parent / (
                output_path.stem + "_checkpoints"
            )
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_frequency,
                save_path=checkpoint_directory,
                name_prefix=output_path.stem,
            )
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=reset_num_timesteps,
            callback=checkpoint_callback,
        )
        model.save(output_path)
        if algorithm == "sac":
            replay_buffer_path = output_path.with_suffix(
                ".replay_buffer.pkl"
            )
            model.save_replay_buffer(replay_buffer_path)
            print(f"Saved replay buffer to {replay_buffer_path}")
        if adaptive_curriculum or recovery_curriculum or broad_curriculum:
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
    elif args.broad_curriculum:
        start_mode = "broad-position curriculum"
    elif args.adaptive_curriculum:
        start_mode = "adaptive position curriculum"
    elif args.randomize_initial_positions:
        start_mode = "randomized-position"
    else:
        start_mode = "fixed"
    if args.transfer_from is not None:
        source = f"transferred from {args.transfer_from}"
    elif args.resume is not None:
        source = f"from {args.resume}"
    else:
        source = "from scratch"
    print(
        f"Training the {start_mode} high-level G1 soccer policy with "
        f"{args.algorithm.upper()} "
        f"{source} for {args.timesteps} timesteps with seed {args.seed}."
    )
    train(
        algorithm=args.algorithm,
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
        broad_curriculum=args.broad_curriculum,
        initial_curriculum_difficulty=(
            args.initial_curriculum_difficulty
        ),
        max_episode_steps=args.max_episode_steps,
        randomize_initial_positions=args.randomize_initial_positions,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        checkpoint_frequency=args.checkpoint_frequency,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
