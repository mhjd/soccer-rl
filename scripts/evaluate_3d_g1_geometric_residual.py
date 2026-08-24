import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.soccer_3d import G1GeometricResidualEnv
from src.soccer_3d.g1_broad_pose import position_category
from src.soccer_3d.g1_geometric_state_machine import GEOMETRIC_PHASES


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the stepwise geometric controller with a learned or "
            "zero residual."
        )
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--disable-residual-phase",
        action="append",
        choices=GEOMETRIC_PHASES,
        default=[],
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.model is not None and not args.model.exists():
        parser.error(f"model not found: {args.model}")
    return args


def main():
    args = parse_args()
    model = PPO.load(args.model) if args.model is not None else None
    env = G1GeometricResidualEnv(
        render_mode="human" if args.render else None,
        disabled_residual_phases=tuple(args.disable_residual_phase),
    )
    goals = 0
    falls = 0
    contacts = 0
    goal_steps = []
    results = []
    category_results = {
        category: [0, 0]
        for category in ("behind_ball", "beside_ball", "ahead_of_ball")
    }
    try:
        for episode_index in range(args.episodes):
            observation, reset_info = env.reset(
                seed=args.seed if episode_index == 0 else None
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                if model is None:
                    action = np.zeros(3, dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                observation, _, terminated, truncated, info = env.step(
                    action
                )

            goal = bool(info["goal"])
            fell = bool(info["fell"])
            contact = bool(info["ball_contact_occurred"])
            elapsed_steps = int(info["elapsed_steps"])
            g1_xy = np.asarray(reset_info["initial_g1_xy"])
            ball_xy = np.asarray(reset_info["initial_ball_xy"])
            category = position_category(
                g1_xy,
                ball_xy,
                env.aim_y_offset,
            )
            goals += int(goal)
            falls += int(fell)
            contacts += int(contact)
            category_results[category][0] += int(goal)
            category_results[category][1] += 1
            if goal:
                goal_steps.append(elapsed_steps)
            results.append(
                {
                    "episode": episode_index,
                    "initial_g1_xy": g1_xy.tolist(),
                    "initial_ball_xy": ball_xy.tolist(),
                    "initial_g1_yaw": float(
                        reset_info["initial_g1_yaw"]
                    ),
                    "category": category,
                    "goal": goal,
                    "fell": fell,
                    "contact": contact,
                    "elapsed_steps": elapsed_steps,
                    "final_phase": info["next_geometric_phase"],
                    "controller_exhausted": bool(
                        info["controller_exhausted"]
                    ),
                }
            )
    finally:
        env.close()

    label = "Learned residual" if model is not None else "Zero residual"
    print(f"Controller: {label}")
    print(f"Goals: {goals}/{args.episodes} ({goals / args.episodes:.1%})")
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

    if args.report_json is not None:
        report = {
            "model": str(args.model) if args.model is not None else None,
            "seed": args.seed,
            "episodes": args.episodes,
            "goals": goals,
            "falls": falls,
            "contacts": contacts,
            "mean_goal_steps": (
                float(np.mean(goal_steps)) if goal_steps else None
            ),
            "disabled_residual_phases": args.disable_residual_phase,
            "episode_results": results,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2))
        print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
