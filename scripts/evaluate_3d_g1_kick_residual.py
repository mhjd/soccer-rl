import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.soccer_3d import G1KickResidualEnv
from src.soccer_3d.g1_locomotion import LEG_JOINT_COUNT


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a learned or zero-action G1 low-level kick residual."
        )
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.model is not None and not args.model.exists():
        parser.error(f"model not found: {args.model}")
    return args


def evaluate(model, episodes, seed, max_episode_steps, render):
    env = G1KickResidualEnv(
        max_episode_steps=max_episode_steps,
        render_mode="human" if render else None,
    )
    goals = 0
    contacts = 0
    falls = 0
    goal_steps = []
    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                if model is None:
                    action = np.zeros(LEG_JOINT_COUNT, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                observation, _, terminated, truncated, info = env.step(
                    action
                )

            goal = bool(info["goal"])
            goals += int(goal)
            contacts += int(info["ball_contact_occurred"])
            falls += int(info["fell"])
            if goal:
                goal_steps.append(info["elapsed_steps"])
    finally:
        env.close()

    return {
        "goals": goals,
        "contacts": contacts,
        "falls": falls,
        "mean_goal_steps": (
            float(np.mean(goal_steps)) if goal_steps else float("nan")
        ),
    }


def main():
    args = parse_args()
    model = PPO.load(args.model) if args.model is not None else None
    results = evaluate(
        model=model,
        episodes=args.episodes,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        render=args.render,
    )
    label = "Learned residual" if model is not None else "Zero residual"
    print(f"Controller: {label}")
    print(
        f"Goals: {results['goals']}/{args.episodes} "
        f"({results['goals'] / args.episodes:.1%})"
    )
    print(f"Ball contacts: {results['contacts']}/{args.episodes}")
    print(f"Falls: {results['falls']}/{args.episodes}")
    print(f"Mean steps to goal: {results['mean_goal_steps']:.1f}")


if __name__ == "__main__":
    main()
