import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_3d_g1_geometric_controller import (
    GeometricTrace,
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
GOAL_HALF_WIDTH = 0.9
BALL_RADIUS = 0.11


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
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
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


def phase_metrics(trace, phase, attempt=None):
    samples = trace.samples_for(phase, attempt=attempt)
    if not samples:
        return None

    pelvis_positions = np.stack([sample.pelvis_xy for sample in samples])
    ball_positions = np.stack([sample.ball_xy for sample in samples])
    approach_distances = np.array(
        [
            np.linalg.norm(sample.approach_xy - sample.pelvis_xy)
            for sample in samples
        ]
    )
    heading_errors = np.abs(
        np.array([sample.heading_error for sample in samples])
    )
    shot_directions = np.stack(
        [
            np.array(
                [
                    np.cos(sample.yaw + sample.heading_error),
                    np.sin(sample.yaw + sample.heading_error),
                ]
            )
            for sample in samples
        ]
    )
    pelvis_ahead_distances = np.sum(
        (pelvis_positions - ball_positions) * shot_directions,
        axis=1,
    )
    passed_ball_indices = np.flatnonzero(pelvis_ahead_distances > 0.25)
    contact_indices = np.flatnonzero(
        np.array([sample.contact for sample in samples])
    )
    pelvis_steps = np.diff(pelvis_positions, axis=0)
    return {
        "steps": int(samples[-1].step - samples[0].step),
        "start_approach_distance": float(approach_distances[0]),
        "end_approach_distance": float(approach_distances[-1]),
        "minimum_approach_distance": float(np.min(approach_distances)),
        "start_heading_error": float(heading_errors[0]),
        "end_heading_error": float(heading_errors[-1]),
        "minimum_heading_error": float(np.min(heading_errors)),
        "pelvis_path_length": float(
            np.sum(np.linalg.norm(pelvis_steps, axis=1))
        ),
        "pelvis_displacement": float(
            np.linalg.norm(pelvis_positions[-1] - pelvis_positions[0])
        ),
        "ball_displacement": float(
            np.linalg.norm(ball_positions[-1] - ball_positions[0])
        ),
        "start_ball_xy": ball_positions[0].tolist(),
        "end_ball_xy": ball_positions[-1].tolist(),
        "maximum_pelvis_ahead_of_ball": float(
            np.max(pelvis_ahead_distances)
        ),
        "first_pelvis_passed_ball_step": (
            int(samples[passed_ball_indices[0]].step)
            if passed_ball_indices.size
            else None
        ),
        "first_contact_step": (
            int(samples[contact_indices[0]].step)
            if contact_indices.size
            else None
        ),
        "contact": bool(any(sample.contact for sample in samples)),
    }


def diagnostic_signature(reached, aligned, info, phases):
    if info["goal"]:
        return "goal"
    if info["fell"]:
        fallen_phase = next(
            (
                phase
                for phase, metrics in phases.items()
                if metrics is not None
                and any(
                    sample.fell for sample in phases[phase]["_samples"]
                )
            ),
            "unknown",
        )
        return f"fall_during_{fallen_phase}"

    detour = phases["detour"]
    approach = phases["approach"]
    if not reached:
        if detour is not None and approach is None:
            if detour["contact"] or detour["ball_displacement"] > 0.15:
                return "detour_ball_contact"
            if detour["pelvis_displacement"] < 0.15:
                return "detour_no_progress"
            return "detour_timeout_after_progress"
        if approach["contact"] or approach["ball_displacement"] > 0.15:
            return "approach_early_ball_contact"
        if approach["minimum_approach_distance"] < 0.16:
            return "approach_oscillation_near_target"
        progress = (
            approach["start_approach_distance"]
            - approach["end_approach_distance"]
        )
        if progress < 0.15:
            return "approach_no_progress"
        return "approach_timeout_after_progress"

    alignment = phases["alignment"]
    if not aligned:
        if alignment["contact"] or alignment["ball_displacement"] > 0.15:
            return "alignment_ball_contact"
        if alignment["minimum_heading_error"] < 0.12:
            return "alignment_oscillation_near_target"
        heading_progress = (
            alignment["start_heading_error"]
            - alignment["end_heading_error"]
        )
        if heading_progress < -0.1:
            return "alignment_wrong_direction"
        if heading_progress < 0.1:
            return "alignment_no_progress"
        return "alignment_insufficient_turn"

    drive = phases["drive_through"]
    lost_ball = drive["maximum_pelvis_ahead_of_ball"] > 0.25
    if not info["ball_contact_occurred"]:
        if lost_ball:
            return "drive_lost_ball_without_foot_contact"
        if drive["minimum_approach_distance"] < 0.16:
            return "drive_near_ball_without_contact"
        return "drive_no_ball_contact"
    end_ball_xy = np.asarray(drive["end_ball_xy"])
    if end_ball_xy[0] - BALL_RADIUS < GOAL_XY[0]:
        shot_wide = (
            abs(end_ball_xy[1] - GOAL_XY[1]) + BALL_RADIUS
            > GOAL_HALF_WIDTH
        )
        if lost_ball and shot_wide:
            return "drive_lost_ball_short_and_wide"
        if lost_ball:
            return "drive_lost_ball_stopped_short"
        return "drive_ball_stopped_short"
    if abs(end_ball_xy[1] - GOAL_XY[1]) + BALL_RADIUS > GOAL_HALF_WIDTH:
        return "drive_shot_wide"
    return "contact_without_detected_goal"


def episode_diagnostics(
    episode_index,
    g1_xy,
    ball_xy,
    g1_yaw,
    category,
    reached,
    aligned,
    info,
    trace,
):
    attempt_ids = sorted({sample.attempt for sample in trace.samples})
    attempt_diagnostics = []
    for attempt in attempt_ids:
        attempt_phases = {}
        for phase in (
            "detour",
            "approach",
            "alignment",
            "drive_through",
        ):
            metrics = phase_metrics(trace, phase, attempt=attempt)
            if metrics is not None:
                metrics["_samples"] = trace.samples_for(
                    phase,
                    attempt=attempt,
                )
            attempt_phases[phase] = metrics
        attempt_diagnostics.append(
            {"attempt": attempt, "phases": attempt_phases}
        )

    phases = attempt_diagnostics[-1]["phases"]

    signature = diagnostic_signature(
        reached,
        aligned,
        info,
        phases,
    )
    serializable_phases = {
        phase: (
            {
                key: value
                for key, value in metrics.items()
                if key != "_samples"
            }
            if metrics is not None
            else None
        )
        for phase, metrics in phases.items()
    }
    serializable_attempts = []
    for attempt in attempt_diagnostics:
        serializable_attempts.append(
            {
                "attempt": attempt["attempt"],
                "phases": {
                    phase: (
                        {
                            key: value
                            for key, value in metrics.items()
                            if key != "_samples"
                        }
                        if metrics is not None
                        else None
                    )
                    for phase, metrics in attempt["phases"].items()
                },
            }
        )
    return {
        "episode": episode_index,
        "initial_g1_xy": g1_xy.tolist(),
        "initial_ball_xy": ball_xy.tolist(),
        "initial_g1_yaw": float(g1_yaw),
        "position_category": category,
        "goal": bool(info["goal"]),
        "fell": bool(info["fell"]),
        "contact": bool(info["ball_contact_occurred"]),
        "elapsed_steps": int(info["elapsed_steps"]),
        "reached_approach": bool(reached),
        "aligned_shot": bool(aligned),
        "attempts": int(info.get("geometric_attempts", 1)),
        "drive_outcome": info.get("drive_outcome", "unknown"),
        "signature": signature,
        "phases": serializable_phases,
        "attempt_diagnostics": serializable_attempts,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = G1SoccerEnv(
        render_mode=None,
        randomize_initial_positions=False,
        recovery_start_probability=0.0,
        observation_mode="soccer_state",
        max_episode_steps=args.max_episode_steps,
    )
    goals = 0
    falls = 0
    contacts = 0
    goals_within_200_steps = 0
    goals_within_300_steps = 0
    goal_steps = []
    failures = {}
    signatures = Counter()
    diagnostic_episodes = []
    attempt_counts = []
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
            trace = GeometricTrace()
            pilot = PilotTools(
                env,
                observation,
                initial_episode_info(),
                aim_y_offset=args.aim_y_offset,
                verbose=False,
                trace=trace,
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
            diagnostic = episode_diagnostics(
                episode_index,
                g1_xy,
                ball_xy,
                g1_yaw,
                category,
                reached,
                aligned,
                info,
                trace,
            )
            diagnostic_episodes.append(diagnostic)
            attempt_counts.append(diagnostic["attempts"])
            signatures[diagnostic["signature"]] += 1
            goals += int(goal)
            falls += int(info["fell"])
            contacts += int(info["ball_contact_occurred"])
            if goal:
                goal_steps.append(info["elapsed_steps"])
                goals_within_200_steps += int(info["elapsed_steps"] <= 200)
                goals_within_300_steps += int(info["elapsed_steps"] <= 300)
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
                    f"signature={diagnostic['signature']} "
                    f"steps={info['elapsed_steps']}",
                    flush=True,
                )
    finally:
        env.close()

    print(f"Broad-distribution goals: {goals}/{args.episodes} ({goals / args.episodes:.1%})")
    print(f"Goals within 200 steps: {goals_within_200_steps}/{args.episodes}")
    print(f"Goals within 300 steps: {goals_within_300_steps}/{args.episodes}")
    print(f"Ball contacts: {contacts}/{args.episodes}")
    print(f"Falls: {falls}/{args.episodes}")
    print(
        "Episodes requiring a retry: "
        f"{sum(attempts > 1 for attempts in attempt_counts)}/"
        f"{args.episodes}"
    )
    print(f"Maximum attempts: {max(attempt_counts)}")
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
    print("Diagnostic signatures:")
    for signature, count in signatures.most_common():
        print(f"  {signature}: {count}")

    if args.report_json is not None:
        report = {
            "seed": args.seed,
            "episodes": args.episodes,
            "aim_y_offset": args.aim_y_offset,
            "max_episode_steps": args.max_episode_steps,
            "success_rate": goals / args.episodes,
            "signatures": dict(signatures),
            "episode_diagnostics": diagnostic_episodes,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Diagnostic report: {args.report_json}")


if __name__ == "__main__":
    main()
