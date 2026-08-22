import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.soccer_3d import G1SoccerEnv


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_fixed.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a high-level PPO policy on fixed-start G1 soccer.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-initial-positions", action="store_true")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes in a MuJoCo window.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    return args


def evaluate(
    model: PPO,
    episodes: int,
    seed: int,
    render: bool,
    randomize_initial_positions: bool,
) -> dict[str, float | int]:
    env = G1SoccerEnv(
        render_mode="human" if render else None,
        randomize_initial_positions=randomize_initial_positions,
    )
    episode_rewards = []
    episode_lengths = []
    successful_episode_lengths = []
    goals = 0
    falls = 0

    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                episode_reward += reward
                episode_length += 1

            goal = bool(info["goal"])
            fell = bool(info["fell"])
            goals += int(goal)
            falls += int(fell)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if goal:
                successful_episode_lengths.append(episode_length)
    finally:
        env.close()

    return {
        "goals": goals,
        "falls": falls,
        "success_rate": goals / episodes,
        "fall_rate": falls / episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "mean_steps_to_goal": (
            float(np.mean(successful_episode_lengths))
            if successful_episode_lengths
            else float("nan")
        ),
    }


def main():
    args = parse_args()
    model = PPO.load(args.model)
    results = evaluate(
        model=model,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        randomize_initial_positions=args.randomize_initial_positions,
    )

    print(f"Goals: {results['goals']}/{args.episodes}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Falls: {results['falls']}/{args.episodes}")
    print(f"Fall rate: {results['fall_rate']:.1%}")
    print(f"Mean reward: {results['mean_reward']:.3f}")
    print(f"Mean episode length: {results['mean_episode_length']:.1f}")
    print(f"Episode length std: {results['episode_length_std']:.1f}")
    print(f"Mean steps to goal: {results['mean_steps_to_goal']:.1f}")


if __name__ == "__main__":
    main()
