import numpy as np

from src.soccer_3d.g1_soccer_env import (
    APPROACH_DISTANCE,
    APPROACH_LATERAL_OFFSET,
    MINIMUM_INITIAL_SEPARATION,
    RANDOM_BALL_XY_HIGH,
    RANDOM_BALL_XY_LOW,
    RANDOM_BEHIND_DISTANCE,
    RANDOM_LATERAL_OFFSET,
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


def sample_behind_ball_pose(rng):
    """Sample the full behind-ball distribution used by the learned policy."""
    ball_xy = rng.uniform(RANDOM_BALL_XY_LOW, RANDOM_BALL_XY_HIGH)
    behind_distance = rng.uniform(*RANDOM_BEHIND_DISTANCE)
    lateral_offset = rng.uniform(*RANDOM_LATERAL_OFFSET)
    g1_xy = np.array(
        [ball_xy[0] - behind_distance, ball_xy[1] + lateral_offset]
    )
    return g1_xy, ball_xy, 0.0


def interpolate_pose(behind_pose, broad_pose, difficulty):
    """Interpolate one easy pose toward one broad pose."""
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be in [0, 1]")
    behind_g1_xy, behind_ball_xy, behind_yaw = behind_pose
    broad_g1_xy, broad_ball_xy, broad_yaw = broad_pose
    g1_xy = np.asarray(behind_g1_xy) + difficulty * (
        np.asarray(broad_g1_xy) - np.asarray(behind_g1_xy)
    )
    ball_xy = np.asarray(behind_ball_xy) + difficulty * (
        np.asarray(broad_ball_xy) - np.asarray(behind_ball_xy)
    )
    g1_yaw = behind_yaw + difficulty * (broad_yaw - behind_yaw)
    return g1_xy, ball_xy, float(g1_yaw)


def sample_interpolated_pose(rng, difficulty, aim_y_offset):
    """Continuously expand behind-ball starts into the broad distribution."""
    for _ in range(1000):
        pose = interpolate_pose(
            sample_behind_ball_pose(rng),
            sample_broad_pose(rng, aim_y_offset),
            difficulty,
        )
        g1_xy, ball_xy, _ = pose
        if np.linalg.norm(g1_xy - ball_xy) >= MINIMUM_INITIAL_SEPARATION:
            return pose
    raise RuntimeError("Could not interpolate a valid initial pose")


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
