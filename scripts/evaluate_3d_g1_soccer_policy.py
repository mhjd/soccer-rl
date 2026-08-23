import argparse
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.soccer_3d import G1SoccerEnv
from src.soccer_3d.g1_soccer_env import MAX_EPISODE_STEPS, OBSERVATION_MODES


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_fixed.zip")
STALL_WINDOW_STEPS = 30
STALL_MAXIMUM_DISPLACEMENT = 0.2
STALL_MAXIMUM_MEAN_SPEED = 0.08


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a high-level PPO policy on fixed-start G1 soccer.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-initial-positions", action="store_true")
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="task",
    )
    parser.add_argument(
        "--recovery-start-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--recovery-state-difficulty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=MAX_EPISODE_STEPS,
    )
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
    if not 0.0 <= args.recovery_start_probability <= 1.0:
        parser.error("--recovery-start-probability must be in [0, 1]")
    if not 0.0 <= args.recovery_state_difficulty <= 1.0:
        parser.error("--recovery-state-difficulty must be in [0, 1]")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    return args


def evaluate(
    model: PPO,
    episodes: int,
    seed: int,
    render: bool,
    randomize_initial_positions: bool,
    recovery_start_probability: float,
    recovery_state_difficulty: float,
    max_episode_steps: int,
    observation_mode: str,
) -> dict[str, float | int]:
    env = G1SoccerEnv(
        render_mode="human" if render else None,
        randomize_initial_positions=randomize_initial_positions,
        recovery_start_probability=recovery_start_probability,
        max_episode_steps=max_episode_steps,
        observation_mode=observation_mode,
    )
    episode_rewards = []
    episode_lengths = []
    successful_episode_lengths = []
    goals = 0
    falls = 0
    recovery_episodes = 0
    recovery_goals = 0
    stalled_failures = 0

    try:
        for episode_index in range(episodes):
            observation, reset_info = env.reset(
                seed=seed + episode_index,
                options={
                    "recovery_state_difficulty": (
                        recovery_state_difficulty
                    )
                },
            )
            recovery_start = bool(reset_info["recovery_start"])
            recovery_episodes += int(recovery_start)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            recent_motion = deque(maxlen=STALL_WINDOW_STEPS)

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                episode_reward += reward
                episode_length += 1
                pelvis_id = env.controller.pelvis_id
                recent_motion.append(
                    (
                        env.data.xpos[pelvis_id, :2].copy(),
                        float(np.linalg.norm(observation[4:6])),
                    )
                )

            goal = bool(info["goal"])
            fell = bool(info["fell"])
            goals += int(goal)
            falls += int(fell)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if goal:
                successful_episode_lengths.append(episode_length)
                recovery_goals += int(recovery_start)
            elif len(recent_motion) == STALL_WINDOW_STEPS:
                displacement = np.linalg.norm(
                    recent_motion[-1][0] - recent_motion[0][0]
                )
                mean_speed = np.mean(
                    [sample[1] for sample in recent_motion]
                )
                stalled_failures += int(
                    displacement < STALL_MAXIMUM_DISPLACEMENT
                    and mean_speed < STALL_MAXIMUM_MEAN_SPEED
                )
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
        "recovery_episodes": recovery_episodes,
        "recovery_goals": recovery_goals,
        "stalled_failures": stalled_failures,
        "recovery_success_rate": (
            recovery_goals / recovery_episodes
            if recovery_episodes
            else 0.0
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
        recovery_start_probability=args.recovery_start_probability,
        recovery_state_difficulty=args.recovery_state_difficulty,
        max_episode_steps=args.max_episode_steps,
        observation_mode=args.observation_mode,
    )

    print(f"Goals: {results['goals']}/{args.episodes}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Falls: {results['falls']}/{args.episodes}")
    print(f"Fall rate: {results['fall_rate']:.1%}")
    print(f"Mean reward: {results['mean_reward']:.3f}")
    print(f"Mean episode length: {results['mean_episode_length']:.1f}")
    print(f"Episode length std: {results['episode_length_std']:.1f}")
    print(f"Mean steps to goal: {results['mean_steps_to_goal']:.1f}")
    print(f"Stalled failures: {results['stalled_failures']}")
    if results["recovery_episodes"]:
        print(
            "Recovery-start goals: "
            f"{results['recovery_goals']}/{results['recovery_episodes']}"
        )
        print(
            "Recovery-start success rate: "
            f"{results['recovery_success_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
