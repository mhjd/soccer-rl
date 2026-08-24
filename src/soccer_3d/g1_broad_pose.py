import numpy as np

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
