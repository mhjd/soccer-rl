import argparse
from pathlib import Path

import numpy as np

from src.soccer_3d import CylinderSoccerEnv


REWARD_STRATEGIES = (
    "combined",
    "contact_phased",
    "approach_warmup",
    "ball_goal_only",
    "goal_only",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy in the 3D cylinder soccer environment.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional Stable-Baselines3 PPO model path.",
    )
    parser.add_argument("--episodes", type=int, default=10)
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
        "--render",
        action="store_true",
        help="Render episodes in a MuJoCo window.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def load_model(model_path):
    if model_path is None:
        return None

    from stable_baselines3 import PPO

    return PPO.load(model_path)


def evaluate(
    model,
    episodes,
    seed,
    render,
    reward_strategy,
    randomize_initial_positions,
):
    render_mode = "human" if render else None
    env = CylinderSoccerEnv(
        render_mode=render_mode,
        reward_strategy=reward_strategy,
        randomize_initial_positions=randomize_initial_positions,
    )
    episode_rewards = []
    episode_shaping_returns = []
    episode_lengths = []
    successful_episode_lengths = []
    goals = 0

    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_shaping_return = 0.0
            episode_length = 0

            while not (terminated or truncated):
                if model is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                episode_reward += reward
                episode_shaping_return += info["shaping_reward"]
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_shaping_returns.append(episode_shaping_return)
            episode_lengths.append(episode_length)
            if info["goal"]:
                successful_episode_lengths.append(episode_length)
            goals += int(info["goal"])
    finally:
        env.close()

    return {
        "goals": goals,
        "success_rate": goals / episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_shaping_return": float(np.mean(episode_shaping_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_steps_to_goal": (
            float(np.mean(successful_episode_lengths))
            if successful_episode_lengths
            else float("nan")
        ),
        "reward_std": float(np.std(episode_rewards)),
        "episode_length_std": float(np.std(episode_lengths)),
    }


def main():
    args = parse_args()
    model = load_model(args.model)

    if model is None:
        print(
            "No model supplied: using a zero-action smoke controller. "
            "These results validate the evaluation loop, not policy quality."
        )

    results = evaluate(
        model=model,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        reward_strategy=args.reward_strategy,
        randomize_initial_positions=args.randomize_initial_positions,
    )

    print(f"Goals: {results['goals']}/{args.episodes}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Mean reward: {results['mean_reward']:.3f}")
    print(
        "Mean shaping return: "
        f"{results['mean_shaping_return']:.3f}"
    )
    print(f"Reward std: {results['reward_std']:.3f}")
    print(f"Mean episode length: {results['mean_episode_length']:.1f}")
    print(f"Mean steps to goal: {results['mean_steps_to_goal']:.1f}")
    print(f"Episode length std: {results['episode_length_std']:.1f}")


if __name__ == "__main__":
    main()
