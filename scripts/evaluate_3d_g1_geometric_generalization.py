import argparse

import numpy as np

from scripts.evaluate_3d_g1_geometric_controller import (
    PilotTools,
    initial_episode_info,
)
from src.soccer_3d import G1SoccerEnv
from src.soccer_3d.g1_soccer_env import (
    APPROACH_DISTANCE,
    APPROACH_LATERAL_OFFSET,
)


BALL_XY_LOW = np.array([0.1, -1.2])
BALL_XY_HIGH = np.array([2.2, 1.2])
G1_XY_LOW = np.array([-0.55, -1.5])
G1_XY_HIGH = np.array([2.5, 1.5])
APPROACH_XY_LOW = np.array([-0.65, -1.55])
APPROACH_XY_HIGH = np.array([2.55, 1.55])
MINIMUM_G1_BALL_DISTANCE = 0.7
GOAL_XY = np.array([3.2, 0.0])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the geometric controller on broad independent G1, "
            "ball, and yaw randomization."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aim-y-offset", type=float, default=0.25)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def approach_point(ball_xy, aim_y_offset):
    aim_xy = GOAL_XY + np.array([0.0, aim_y_offset])
    shot_direction = aim_xy - ball_xy
    shot_direction /= np.linalg.norm(shot_direction)
    shot_lateral = np.array([-shot_direction[1], shot_direction[0]])
    return (
        ball_xy
        - APPROACH_DISTANCE * shot_direction
        + APPROACH_LATERAL_OFFSET * shot_lateral
    )


def sample_broad_pose(rng, aim_y_offset):
    for _ in range(1000):
        ball_xy = rng.uniform(BALL_XY_LOW, BALL_XY_HIGH)
        target_xy = approach_point(ball_xy, aim_y_offset)
        if np.any(target_xy < APPROACH_XY_LOW) or np.any(
            target_xy > APPROACH_XY_HIGH
        ):
            continue

        g1_xy = rng.uniform(G1_XY_LOW, G1_XY_HIGH)
        if np.linalg.norm(g1_xy - ball_xy) < MINIMUM_G1_BALL_DISTANCE:
            continue

        g1_yaw = rng.uniform(-np.pi, np.pi)
        return g1_xy, ball_xy, g1_yaw
    raise RuntimeError("Could not sample a broad valid initial pose")


def position_category(g1_xy, ball_xy, aim_y_offset):
    aim_xy = GOAL_XY + np.array([0.0, aim_y_offset])
    shot_direction = aim_xy - ball_xy
    shot_direction /= np.linalg.norm(shot_direction)
    along_shot_axis = float(np.dot(g1_xy - ball_xy, shot_direction))
    if along_shot_axis < -0.25:
        return "behind_ball"
    if along_shot_axis > 0.25:
        return "ahead_of_ball"
    return "beside_ball"


def failure_reason(reached, aligned, info):
    if info["fell"]:
        return "fall"
    if not reached:
        return "approach_not_reached"
    if not aligned:
        return "shot_not_aligned"
    if not info["ball_contact_occurred"]:
        return "no_ball_contact"
    return "contact_without_goal"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = G1SoccerEnv(
        render_mode=None,
        randomize_initial_positions=False,
        recovery_start_probability=0.0,
        observation_mode="soccer_state",
        max_episode_steps=300,
    )
    goals = 0
    falls = 0
    contacts = 0
    goals_within_200_steps = 0
    goal_steps = []
    failures = {}
    category_results = {
        category: [0, 0]
        for category in ("behind_ball", "beside_ball", "ahead_of_ball")
    }

    try:
        for episode_index in range(args.episodes):
            g1_xy, ball_xy, g1_yaw = sample_broad_pose(
                rng,
                args.aim_y_offset,
            )
            observation, _ = env.reset(
                seed=args.seed + episode_index,
                options={
                    "initial_g1_xy": g1_xy,
                    "initial_ball_xy": ball_xy,
                    "initial_g1_yaw": g1_yaw,
                },
            )
            pilot = PilotTools(
                env,
                observation,
                initial_episode_info(),
                aim_y_offset=args.aim_y_offset,
                verbose=False,
            )
            reached, aligned, info = pilot.solve()
            goal = bool(info["goal"])
            category = position_category(
                g1_xy,
                ball_xy,
                args.aim_y_offset,
            )
            category_results[category][0] += int(goal)
            category_results[category][1] += 1
            goals += int(goal)
            falls += int(info["fell"])
            contacts += int(info["ball_contact_occurred"])
            if goal:
                goal_steps.append(info["elapsed_steps"])
                goals_within_200_steps += int(info["elapsed_steps"] <= 200)
            else:
                reason = failure_reason(reached, aligned, info)
                failures[reason] = failures.get(reason, 0) + 1

            if args.verbose and not goal:
                print(
                    f"episode={episode_index} "
                    f"g1={np.round(g1_xy, 2)} "
                    f"ball={np.round(ball_xy, 2)} "
                    f"yaw={g1_yaw:+.2f} "
                    f"category={category} "
                    f"failure={reason} "
                    f"steps={info['elapsed_steps']}",
                    flush=True,
                )
    finally:
        env.close()

    print(f"Broad-distribution goals: {goals}/{args.episodes} ({goals / args.episodes:.1%})")
    print(f"Goals within 200 steps: {goals_within_200_steps}/{args.episodes}")
    print(f"Ball contacts: {contacts}/{args.episodes}")
    print(f"Falls: {falls}/{args.episodes}")
    print(
        "Mean steps to goal: "
        + (f"{np.mean(goal_steps):.1f}" if goal_steps else "n/a")
    )
    print("Results by initial position relative to the ball:")
    for category, (category_goals, category_total) in category_results.items():
        rate = category_goals / category_total if category_total else 0.0
        print(
            f"  {category}: {category_goals}/{category_total} "
            f"({rate:.1%})"
        )
    print("Failure reasons:")
    if failures:
        for reason, count in sorted(failures.items()):
            print(f"  {reason}: {count}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
