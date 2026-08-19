import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "soccer-rl-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.soccer_3d.cylinder_env import MAX_EPISODE_STEPS


DEFAULT_INPUT_DIR = Path("models/learning_curves")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot aggregated 3D-cylinder learning curves.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INPUT_DIR / "comparison.png",
    )
    return parser.parse_args()


def load_results(input_dir):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(input_dir.glob("*_seed*.csv")):
        with path.open(newline="") as input_file:
            for row in csv.DictReader(input_file):
                key = (row["strategy"], int(row["timesteps"]))
                success_rate = float(row["success_rate"])
                grouped[key]["success_rate"].append(success_rate)
                grouped[key]["curriculum_difficulty"].append(
                    float(row["curriculum_difficulty"])
                )
                if "mean_steps_to_goal" in row:
                    mean_steps_to_goal = float(row["mean_steps_to_goal"])
                elif success_rate > 0.0:
                    mean_episode_length = float(row["mean_episode_length"])
                    mean_steps_to_goal = (
                        mean_episode_length
                        - (1.0 - success_rate) * MAX_EPISODE_STEPS
                    ) / success_rate
                else:
                    mean_steps_to_goal = float("nan")
                grouped[key]["mean_steps_to_goal"].append(
                    mean_steps_to_goal
                )
    if not grouped:
        raise ValueError(f"No learning-curve CSV files found in {input_dir}")
    return grouped


def series_for(grouped, strategy, metric):
    points = sorted(
        (timesteps, values[metric])
        for (name, timesteps), values in grouped.items()
        if name == strategy
    )
    timesteps = np.array([point[0] for point in points])
    finite_values = [
        np.asarray(point[1])[np.isfinite(point[1])]
        for point in points
    ]
    means = np.array(
        [np.mean(values) if values.size else np.nan for values in finite_values]
    )
    stds = np.array(
        [np.std(values) if values.size else np.nan for values in finite_values]
    )
    return timesteps, means, stds


def speed_series_for(grouped, strategy):
    points = sorted(
        (timesteps, values)
        for (name, timesteps), values in grouped.items()
        if name == strategy
    )
    timesteps = np.array([point[0] for point in points])
    means = []
    stds = []
    for _, values in points:
        speeds = np.asarray(values["mean_steps_to_goal"])
        weights = np.asarray(values["success_rate"])
        valid = np.isfinite(speeds) & (weights > 0.0)
        if not np.any(valid):
            means.append(np.nan)
            stds.append(np.nan)
            continue
        mean = np.average(speeds[valid], weights=weights[valid])
        variance = np.average(
            (speeds[valid] - mean) ** 2,
            weights=weights[valid],
        )
        means.append(mean)
        stds.append(np.sqrt(variance))
    return timesteps, np.asarray(means), np.asarray(stds)


def main():
    args = parse_args()
    grouped = load_results(args.input_dir)
    figure, (success_axis, speed_axis, difficulty_axis) = plt.subplots(
        3,
        1,
        figsize=(9, 11),
        sharex=True,
    )
    labels = {
        "adaptive_curriculum": "Adaptive curriculum, sparse reward",
        "contact_phased": "Contact-phased shaping",
        "curriculum_contact_phased": "Curriculum + contact-phased shaping",
    }

    for strategy, label in labels.items():
        timesteps, means, stds = series_for(
            grouped,
            strategy,
            "success_rate",
        )
        success_axis.plot(timesteps, means, label=label)
        success_axis.fill_between(
            timesteps,
            np.clip(means - stds, 0.0, 1.0),
            np.clip(means + stds, 0.0, 1.0),
            alpha=0.18,
        )

        speed_steps, speed_means, speed_stds = speed_series_for(
            grouped,
            strategy,
        )
        speed_axis.plot(speed_steps, speed_means, label=label)
        speed_axis.fill_between(
            speed_steps,
            np.clip(speed_means - speed_stds, 0.0, MAX_EPISODE_STEPS),
            np.clip(speed_means + speed_stds, 0.0, MAX_EPISODE_STEPS),
            alpha=0.18,
        )

    for strategy, color in (
        ("adaptive_curriculum", "tab:blue"),
        ("curriculum_contact_phased", "tab:green"),
    ):
        curriculum_steps, difficulty_means, difficulty_stds = series_for(
            grouped,
            strategy,
            "curriculum_difficulty",
        )
        difficulty_axis.plot(
            curriculum_steps,
            difficulty_means,
            color=color,
            label=labels[strategy],
        )
        difficulty_axis.fill_between(
            curriculum_steps,
            np.clip(difficulty_means - difficulty_stds, 0.0, 1.0),
            np.clip(difficulty_means + difficulty_stds, 0.0, 1.0),
            color=color,
            alpha=0.18,
        )

    success_axis.set_ylabel("Goal success rate")
    success_axis.set_ylim(0.0, 1.0)
    success_axis.grid(alpha=0.25)
    success_axis.legend()
    speed_axis.set_ylabel("Mean steps to goal\n(successes only)")
    speed_axis.set_ylim(0.0, MAX_EPISODE_STEPS)
    speed_axis.grid(alpha=0.25)
    difficulty_axis.set_xlabel("Training interactions")
    difficulty_axis.set_ylabel("Curriculum difficulty")
    difficulty_axis.set_ylim(0.0, 1.0)
    difficulty_axis.grid(alpha=0.25)
    difficulty_axis.legend()
    figure.suptitle("Learning on fixed held-out start states")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
