from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from src.soccer_3d.g1_locomotion import COMMAND_HIGH, COMMAND_LOW
from src.soccer_3d.g1_soccer_env import (
    APPROACH_DISTANCE,
    APPROACH_LATERAL_OFFSET,
    G1SoccerEnv,
)


BALL_PATH_CLEARANCE = 0.45
BALL_DETOUR_RADIUS = 0.8
DETOUR_SIDE_EPSILON = 0.05
GEOMETRIC_PHASES = (
    "detour",
    "approach",
    "alignment",
    "pose_refinement",
    "drive_through",
)


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class GeometricSituation:
    pelvis_xy: np.ndarray
    yaw: float
    ball_xy: np.ndarray
    approach_xy: np.ndarray
    goal_xy: np.ndarray
    heading_error: float


def read_geometric_situation(
    env: G1SoccerEnv,
    aim_y_offset: float,
) -> GeometricSituation:
    pelvis_id = env.controller.pelvis_id
    pelvis_xy = env.data.xpos[pelvis_id, :2].copy()
    rotation = env.data.xmat[pelvis_id].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    ball_xy = env.data.xpos[env._ball_body_id, :2].copy()
    goal_xy = env.data.site_xpos[env._goal_line_site_id, :2].copy()
    goal_xy[1] += aim_y_offset
    shot_direction = goal_xy - ball_xy
    shot_direction /= np.linalg.norm(shot_direction)
    shot_lateral = np.array([-shot_direction[1], shot_direction[0]])
    approach_xy = (
        ball_xy
        - APPROACH_DISTANCE * shot_direction
        + APPROACH_LATERAL_OFFSET * shot_lateral
    )
    shot_heading = float(
        np.arctan2(
            goal_xy[1] - ball_xy[1],
            goal_xy[0] - ball_xy[0],
        )
    )
    return GeometricSituation(
        pelvis_xy=pelvis_xy,
        yaw=yaw,
        ball_xy=ball_xy,
        approach_xy=approach_xy,
        goal_xy=goal_xy,
        heading_error=wrap_angle(shot_heading - yaw),
    )


def physical_to_normalized(command: np.ndarray) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32)
    scale = np.where(command < 0.0, -COMMAND_LOW, COMMAND_HIGH)
    return np.divide(
        command,
        scale,
        out=np.zeros_like(command),
        where=scale != 0.0,
    ).astype(np.float32)


class GeometricCommandStateMachine:
    """Yield one command at a time while retaining geometric-control phase."""

    def __init__(self, env: G1SoccerEnv, aim_y_offset: float = 0.25):
        self.env = env
        self.aim_y_offset = float(aim_y_offset)
        self.phase = "uninitialized"
        self.reached_approach = False
        self.aligned_shot = False
        self.exhausted = False
        self._commands = self._command_sequence()

    def read_situation(self) -> GeometricSituation:
        return read_geometric_situation(self.env, self.aim_y_offset)

    def next_command(self) -> np.ndarray | None:
        if self.exhausted:
            return None
        try:
            return np.asarray(next(self._commands), dtype=np.float32)
        except StopIteration:
            self.exhausted = True
            return None

    def command_toward(
        self,
        situation: GeometricSituation,
        target_xy: np.ndarray,
        face_shot: bool = True,
    ) -> np.ndarray:
        error_world = target_xy - situation.pelvis_xy
        distance = float(np.linalg.norm(error_world))
        pelvis_id = self.env.controller.pelvis_id
        rotation = self.env.data.xmat[pelvis_id].reshape(3, 3)
        error_robot = rotation[:2, :2].T @ error_world
        desired_speed = min(0.6, max(0.25, distance))
        translation = desired_speed * error_robot / distance
        translation = np.clip(
            translation,
            COMMAND_LOW[:2],
            COMMAND_HIGH[:2],
        )
        yaw_rate = (
            np.clip(1.5 * situation.heading_error, -0.2, 0.2)
            if face_shot
            else 0.0
        )
        return np.array(
            [translation[0], translation[1], yaw_rate],
            dtype=np.float32,
        )

    def _move_to_point(
        self,
        target_provider: Callable[[GeometricSituation], np.ndarray],
        *,
        position_tolerance: float = 0.08,
        max_steps: int = 100,
        face_shot: bool = True,
    ) -> Iterator[np.ndarray]:
        for _ in range(max_steps):
            situation = self.read_situation()
            target_xy = np.asarray(target_provider(situation), dtype=float)
            distance = float(
                np.linalg.norm(target_xy - situation.pelvis_xy)
            )
            if distance <= position_tolerance:
                zero_command = np.zeros(3, dtype=np.float32)
                yield zero_command
                yield zero_command
                return True
            yield self.command_toward(
                situation,
                target_xy,
                face_shot=face_shot,
            )
        return False

    def _move_to_approach(self) -> Iterator[np.ndarray]:
        situation = self.read_situation()
        direct_path = situation.approach_xy - situation.pelvis_xy
        path_length_squared = float(np.dot(direct_path, direct_path))
        if path_length_squared <= np.finfo(np.float64).eps:
            ball_blocks_path = False
        else:
            projection = float(
                np.clip(
                    np.dot(
                        situation.ball_xy - situation.pelvis_xy,
                        direct_path,
                    )
                    / path_length_squared,
                    0.0,
                    1.0,
                )
            )
            closest_path_point = (
                situation.pelvis_xy + projection * direct_path
            )
            path_clearance = float(
                np.linalg.norm(closest_path_point - situation.ball_xy)
            )
            ball_blocks_path = (
                0.0 < projection < 1.0
                and path_clearance < BALL_PATH_CLEARANCE
            )

        if ball_blocks_path:
            shot_direction = situation.goal_xy - situation.ball_xy
            shot_direction /= np.linalg.norm(shot_direction)
            shot_lateral = np.array(
                [-shot_direction[1], shot_direction[0]]
            )
            lateral_coordinate = float(
                np.dot(
                    situation.pelvis_xy - situation.ball_xy,
                    shot_lateral,
                )
            )
            if abs(lateral_coordinate) > DETOUR_SIDE_EPSILON:
                detour_side = np.sign(lateral_coordinate)
            else:
                candidates = (
                    situation.ball_xy
                    - BALL_DETOUR_RADIUS * shot_lateral,
                    situation.ball_xy
                    + BALL_DETOUR_RADIUS * shot_lateral,
                )
                detour_side = (
                    -1.0
                    if abs(candidates[0][1]) <= abs(candidates[1][1])
                    else 1.0
                )

            def detour_target(current: GeometricSituation) -> np.ndarray:
                direction = current.goal_xy - current.ball_xy
                direction /= np.linalg.norm(direction)
                lateral = np.array([-direction[1], direction[0]])
                return (
                    current.ball_xy
                    + detour_side * BALL_DETOUR_RADIUS * lateral
                )

            self.phase = "detour"
            if not (yield from self._move_to_point(
                detour_target,
                max_steps=100,
            )):
                return False

        self.phase = "approach"
        return (
            yield from self._move_to_point(
                lambda current: current.approach_xy,
                max_steps=120,
            )
        )

    def _align_with_shot(
        self,
        tolerance: float = 0.12,
        max_cycles: int = 60,
    ) -> Iterator[np.ndarray]:
        for _ in range(max_cycles):
            situation = self.read_situation()
            if abs(situation.heading_error) <= tolerance:
                return True
            yaw_rate = np.sign(situation.heading_error) * 0.2
            forward = np.array([0.35, 0.0, yaw_rate], dtype=np.float32)
            backward = np.array([-0.35, 0.0, yaw_rate], dtype=np.float32)
            yield forward
            yield forward
            yield backward
            yield backward
        return abs(self.read_situation().heading_error) <= tolerance

    def _refine_shot_pose(
        self,
        position_tolerance: float = 0.08,
        heading_tolerance: float = 0.12,
        max_cycles: int = 3,
    ) -> Iterator[np.ndarray]:
        for _ in range(max_cycles):
            situation = self.read_situation()
            position_error = float(
                np.linalg.norm(
                    situation.approach_xy - situation.pelvis_xy
                )
            )
            if (
                position_error <= position_tolerance
                and abs(situation.heading_error) <= heading_tolerance
            ):
                return True

            if position_error > position_tolerance:
                if not (yield from self._move_to_point(
                    lambda current: current.approach_xy,
                    position_tolerance=position_tolerance,
                    max_steps=80,
                )):
                    return False

            if not (yield from self._align_with_shot(
                tolerance=heading_tolerance,
            )):
                return False

        situation = self.read_situation()
        return bool(
            np.linalg.norm(situation.approach_xy - situation.pelvis_xy)
            <= position_tolerance
            and abs(situation.heading_error) <= heading_tolerance
        )

    def _drive_through_ball(self) -> Iterator[np.ndarray]:
        def beyond_ball(situation: GeometricSituation) -> np.ndarray:
            direction = situation.goal_xy - situation.ball_xy
            direction /= np.linalg.norm(direction)
            return situation.ball_xy + 1.5 * direction

        return (
            yield from self._move_to_point(
                beyond_ball,
                position_tolerance=0.05,
                max_steps=100,
                face_shot=True,
            )
        )

    def _command_sequence(self) -> Iterator[np.ndarray]:
        self.reached_approach = yield from self._move_to_approach()
        if not self.reached_approach:
            return

        self.phase = "alignment"
        self.aligned_shot = yield from self._align_with_shot()
        if not self.aligned_shot:
            return

        self.phase = "pose_refinement"
        self.aligned_shot = yield from self._refine_shot_pose()
        if not self.aligned_shot:
            return

        self.phase = "drive_through"
        yield from self._drive_through_ball()
