import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.soccer_3d import G1KickResidualEnv
from src.soccer_3d.g1_kick_residual_env import (
    CONTACT_DISTANCE_RANGE,
    CONTACT_LATERAL_RANGE,
    WARMUP_DURATION_RANGE,
)
from src.soccer_3d.g1_locomotion import LEG_JOINT_COUNT


@dataclass(frozen=True)
class Challenge:
    warmup_duration: float
    contact_distance: float
    contact_lateral: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare zero and learned low-level residuals on a fixed "
            "boundary-value kick challenge suite."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/ppo_3d_g1_kick_residual_100k.zip"),
    )
    parser.add_argument("--gait-timings", type=int, default=25)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    if args.gait_timings < 2:
        parser.error("--gait-timings must be at least 2")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    return args


def challenge_suite(gait_timings):
    warmup_durations = np.linspace(
        WARMUP_DURATION_RANGE[0],
        WARMUP_DURATION_RANGE[1],
        gait_timings,
    )
    return [
        Challenge(float(warmup), float(distance), float(lateral))
        for warmup in warmup_durations
        for distance in CONTACT_DISTANCE_RANGE
        for lateral in CONTACT_LATERAL_RANGE
    ]


def run_episode(env, challenge, seed, model):
    observation, _ = env.reset(
        seed=seed,
        options={
            "warmup_duration": challenge.warmup_duration,
            "contact_distance": challenge.contact_distance,
            "contact_lateral": challenge.contact_lateral,
        },
    )
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if model is None:
            action = np.zeros(LEG_JOINT_COUNT, dtype=np.float32)
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
    return {
        "goal": bool(info["goal"]),
        "fell": bool(info["fell"]),
        "contact": bool(info["ball_contact_occurred"]),
        "steps": int(info["elapsed_steps"]),
    }


def summarize(label, results):
    goals = sum(result["goal"] for result in results)
    falls = sum(result["fell"] for result in results)
    contacts = sum(result["contact"] for result in results)
    goal_steps = [result["steps"] for result in results if result["goal"]]
    print(f"{label}:")
    print(f"  Goals: {goals}/{len(results)} ({goals / len(results):.1%})")
    print(f"  Contacts: {contacts}/{len(results)}")
    print(f"  Falls: {falls}/{len(results)}")
    print(
        "  Mean steps to goal: "
        + (f"{np.mean(goal_steps):.1f}" if goal_steps else "n/a")
    )


def summarize_corners(challenges, zero_results, learned_results):
    print("Results by distance and lateral extreme:")
    for distance in CONTACT_DISTANCE_RANGE:
        for lateral in CONTACT_LATERAL_RANGE:
            indices = [
                index
                for index, challenge in enumerate(challenges)
                if challenge.contact_distance == distance
                and challenge.contact_lateral == lateral
            ]
            zero_goals = sum(zero_results[index]["goal"] for index in indices)
            learned_goals = sum(
                learned_results[index]["goal"] for index in indices
            )
            zero_falls = sum(zero_results[index]["fell"] for index in indices)
            learned_falls = sum(
                learned_results[index]["fell"] for index in indices
            )
            fixed = sum(
                not zero_results[index]["goal"]
                and learned_results[index]["goal"]
                for index in indices
            )
            regressed = sum(
                zero_results[index]["goal"]
                and not learned_results[index]["goal"]
                for index in indices
            )
            print(
                f"  distance={distance:.2f}, lateral={lateral:+.2f}: "
                f"zero={zero_goals}/{len(indices)}, "
                f"learned={learned_goals}/{len(indices)}, "
                f"fixed={fixed}, regressed={regressed}, "
                f"falls={zero_falls}->{learned_falls}"
            )


def main():
    args = parse_args()
    model = PPO.load(args.model)
    challenges = challenge_suite(args.gait_timings)
    zero_env = G1KickResidualEnv(max_episode_steps=args.max_episode_steps)
    learned_env = G1KickResidualEnv(max_episode_steps=args.max_episode_steps)
    zero_results = []
    learned_results = []
    try:
        for index, challenge in enumerate(challenges):
            episode_seed = args.seed + index
            zero_results.append(
                run_episode(zero_env, challenge, episode_seed, model=None)
            )
            learned_results.append(
                run_episode(learned_env, challenge, episode_seed, model=model)
            )
    finally:
        zero_env.close()
        learned_env.close()

    summarize("Zero residual", zero_results)
    summarize("Learned low-level residual", learned_results)
    summarize_corners(challenges, zero_results, learned_results)
    fixed = sum(
        not zero["goal"] and learned["goal"]
        for zero, learned in zip(zero_results, learned_results)
    )
    regressed = sum(
        zero["goal"] and not learned["goal"]
        for zero, learned in zip(zero_results, learned_results)
    )
    print(f"Paired outcomes: fixed={fixed}, regressed={regressed}")
    print(
        "Challenge grid: "
        f"{args.gait_timings} gait timings x "
        f"{len(CONTACT_DISTANCE_RANGE)} distance extremes x "
        f"{len(CONTACT_LATERAL_RANGE)} lateral extremes "
        f"= {len(challenges)} exercises"
    )


if __name__ == "__main__":
    main()
