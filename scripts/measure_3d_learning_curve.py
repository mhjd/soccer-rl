import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from scripts.evaluate_3d_cylinder import evaluate
from src.soccer_3d import AdaptiveStartCurriculum, CylinderSoccerEnv


STRATEGIES = (
    "adaptive_curriculum",
    "contact_phased",
    "curriculum_contact_phased",
)
DEFAULT_OUTPUT_DIR = Path("models/learning_curves")


class PeriodicGoalEvaluation(BaseCallback):
    def __init__(
        self,
        output_path,
        eval_interval,
        eval_episodes,
        eval_seed,
        strategy,
        training_seed,
        curriculum=None,
    ):
        super().__init__()
        self.output_path = output_path
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.strategy = strategy
        self.training_seed = training_seed
        self.curriculum = curriculum
        self.next_evaluation = 0
        self.rows = []

    def _write_results(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=self.rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def _evaluate(self):
        results = evaluate(
            model=self.model,
            episodes=self.eval_episodes,
            seed=self.eval_seed,
            render=False,
            reward_strategy="goal_only",
            randomize_initial_positions=True,
        )
        difficulty = (
            self.curriculum.difficulty
            if self.curriculum is not None
            else 1.0
        )
        self.rows.append(
            {
                "strategy": self.strategy,
                "training_seed": self.training_seed,
                "timesteps": self.next_evaluation,
                "success_rate": results["success_rate"],
                "mean_episode_length": results["mean_episode_length"],
                "mean_steps_to_goal": results["mean_steps_to_goal"],
                "curriculum_difficulty": difficulty,
            }
        )
        self._write_results()
        print(
            f"Evaluation at {self.next_evaluation} steps: "
            f"{results['success_rate']:.1%} goals, "
            f"difficulty {difficulty:.3f}"
        )
        self.next_evaluation += self.eval_interval

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate()
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure 3D-cylinder goal success during PPO training.",
    )
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.eval_interval <= 0:
        parser.error("--eval-interval must be positive")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    return args


def main():
    args = parse_args()
    uses_curriculum = args.strategy != "contact_phased"
    uses_contact_phased_reward = "contact_phased" in args.strategy
    base_env = CylinderSoccerEnv(
        reward_strategy=(
            "contact_phased" if uses_contact_phased_reward else "goal_only"
        ),
        randomize_initial_positions=not uses_curriculum,
    )
    curriculum = (
        AdaptiveStartCurriculum(base_env)
        if uses_curriculum
        else None
    )
    training_env = curriculum if curriculum is not None else base_env
    output_stem = f"{args.strategy}_seed{args.seed}"
    output_path = args.output_dir / f"{output_stem}.csv"
    model_path = args.output_dir / output_stem
    callback = PeriodicGoalEvaluation(
        output_path=output_path,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        strategy=args.strategy,
        training_seed=args.seed,
        curriculum=curriculum,
    )
    model = PPO(
        policy="MlpPolicy",
        env=training_env,
        n_steps=2048,
        seed=args.seed,
        verbose=0,
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(model_path)
    finally:
        training_env.close()


if __name__ == "__main__":
    main()
