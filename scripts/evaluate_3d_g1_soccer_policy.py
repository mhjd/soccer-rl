import argparse
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from scripts.evaluate_3d_g1_geometric_controller import (
    PilotTools,
    initial_episode_info,
)
from src.soccer_3d import G1SoccerEnv
from src.soccer_3d.g1_broad_pose import sample_broad_pose
from src.soccer_3d.g1_soccer_env import MAX_EPISODE_STEPS, OBSERVATION_MODES


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_fixed.zip")
ALGORITHM_CLASSES = {"ppo": PPO, "sac": SAC}
CONTROLLERS = (*ALGORITHM_CLASSES, "geometric")
STALL_WINDOW_STEPS = 30
STALL_MAXIMUM_DISPLACEMENT = 0.2
STALL_MAXIMUM_MEAN_SPEED = 0.08


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a high-level policy on G1 soccer.",
    )
    parser.add_argument(
        "--algorithm",
        choices=CONTROLLERS,
        default="ppo",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print running metrics every N episodes; 0 disables progress.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-initial-positions", action="store_true")
    parser.add_argument("--broad-initial-positions", action="store_true")
    parser.add_argument("--aim-y-offset", type=float, default=0.25)
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
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.algorithm == "geometric":
        if args.model is not None:
            parser.error("--model is not used by the geometric controller")
    else:
        if args.model is None:
            args.model = DEFAULT_MODEL_PATH
        if not args.model.exists():
            parser.error(f"model not found: {args.model}")
    if args.randomize_initial_positions and args.broad_initial_positions:
        parser.error(
            "Choose either randomized behind-ball starts or broad starts"
        )
    if not 0.0 <= args.recovery_start_probability <= 1.0:
        parser.error("--recovery-start-probability must be in [0, 1]")
    if not 0.0 <= args.recovery_state_difficulty <= 1.0:
        parser.error("--recovery-state-difficulty must be in [0, 1]")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    return args


def evaluate(
    model: BaseAlgorithm | None,
    controller: str,
    episodes: int,
    seed: int,
    render: bool,
    randomize_initial_positions: bool,
    broad_initial_positions: bool,
    aim_y_offset: float,
    recovery_start_probability: float,
    recovery_state_difficulty: float,
    max_episode_steps: int,
    observation_mode: str,
    progress_every: int = 0,
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
    pose_rng = np.random.default_rng(seed)

    try:
        for episode_index in range(episodes):
            reset_options = {
                "recovery_state_difficulty": recovery_state_difficulty,
            }
            if broad_initial_positions:
                g1_xy, ball_xy, g1_yaw = sample_broad_pose(
                    pose_rng,
                    aim_y_offset,
                )
                reset_options.update(
                    {
                        "initial_g1_xy": g1_xy,
                        "initial_ball_xy": ball_xy,
                        "initial_g1_yaw": g1_yaw,
                    }
                )
            observation, reset_info = env.reset(
                seed=seed + episode_index,
                options=reset_options,
            )
            recovery_start = bool(reset_info["recovery_start"])
            recovery_episodes += int(recovery_start)
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            recent_motion = deque(maxlen=STALL_WINDOW_STEPS)

            if controller == "geometric":
                pilot = PilotTools(
                    env,
                    observation,
                    initial_episode_info(),
                    verbose=False,
                )
                _, _, info = pilot.solve()
                terminated = bool(info["goal"] or info["fell"])
                truncated = not terminated
                episode_reward = float(info["goal"])
                episode_length = int(info["elapsed_steps"])
            else:
                while not (terminated or truncated):
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                    observation, reward, terminated, truncated, info = (
                        env.step(action)
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

            completed_episodes = episode_index + 1
            if (
                progress_every
                and completed_episodes % progress_every == 0
                and completed_episodes < episodes
            ):
                print(
                    f"Progress: {completed_episodes}/{episodes} | "
                    f"success {goals / completed_episodes:.1%} | "
                    f"falls {falls / completed_episodes:.1%} | "
                    "mean length "
                    f"{np.mean(episode_lengths):.1f}",
                    flush=True,
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
    model = (
        None
        if args.algorithm == "geometric"
        else ALGORITHM_CLASSES[args.algorithm].load(args.model)
    )
    results = evaluate(
        model=model,
        controller=args.algorithm,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        randomize_initial_positions=args.randomize_initial_positions,
        broad_initial_positions=args.broad_initial_positions,
        aim_y_offset=args.aim_y_offset,
        recovery_start_probability=args.recovery_start_probability,
        recovery_state_difficulty=args.recovery_state_difficulty,
        max_episode_steps=args.max_episode_steps,
        observation_mode=args.observation_mode,
        progress_every=args.progress_every,
    )

    print(f"Controller: {args.algorithm.upper()}")
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
