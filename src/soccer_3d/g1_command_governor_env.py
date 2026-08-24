from __future__ import annotations

import mujoco
import numpy as np

from src.soccer_3d.g1_soccer_env import (
    COMMAND_HIGH,
    COMMAND_LOW,
    MINIMUM_WALKING_TRANSLATION,
    STAND_TRANSLATION_THRESHOLD,
    G1SoccerEnv,
    normalized_action_to_command,
)

COMMAND_GOVERNOR_MODES = ("wall",)
DEFAULT_WALL_MARGIN = 0.35
WALL_TANGENT_MARGIN = 0.30
WALL_GEOMETRIES = (
    ("arena_back_wall", 0, 1.0),
    ("arena_left_wall", 1, -1.0),
    ("arena_right_wall", 1, 1.0),
    ("arena_goal_wall_left", 0, -1.0),
    ("arena_goal_wall_right", 0, -1.0),
)


class G1CommandGovernorEnv(G1SoccerEnv):
    """Optionally make learned high-level commands easier to execute."""

    def __init__(
        self,
        *args,
        wall_margin: float = DEFAULT_WALL_MARGIN,
        **kwargs,
    ):
        if wall_margin < 0.0:
            raise ValueError("wall_margin cannot be negative")

        super().__init__(*args, **kwargs)
        self.wall_margin = float(wall_margin)
        self._wall_constraints = self._load_wall_constraints()

    def _load_wall_constraints(self):
        constraints = []
        for geom_name, normal_axis, normal_sign in WALL_GEOMETRIES:
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            center = self.model.geom_pos[geom_id, :2].copy()
            half_size = self.model.geom_size[geom_id, :2].copy()
            normal = np.zeros(2, dtype=np.float64)
            normal[normal_axis] = normal_sign
            face = center.copy()
            face[normal_axis] += normal_sign * half_size[normal_axis]
            tangent_axis = 1 - normal_axis
            constraints.append(
                (
                    normal,
                    face,
                    tangent_axis,
                    center[tangent_axis] - half_size[tangent_axis],
                    center[tangent_axis] + half_size[tangent_axis],
                )
            )
        return constraints

    def _remove_wall_inward_velocity(
        self,
        command: np.ndarray,
    ) -> np.ndarray:
        pelvis_id = self.controller.pelvis_id
        pelvis_xy = self.data.xpos[pelvis_id, :2]
        pelvis_rotation = self.data.xmat[pelvis_id].reshape(3, 3)
        pelvis_yaw = np.arctan2(
            pelvis_rotation[1, 0],
            pelvis_rotation[0, 0],
        )
        robot_to_world_xy = np.array(
            [
                [np.cos(pelvis_yaw), -np.sin(pelvis_yaw)],
                [np.sin(pelvis_yaw), np.cos(pelvis_yaw)],
            ],
            dtype=np.float64,
        )
        world_velocity = robot_to_world_xy @ command[:2]

        for normal, face, tangent_axis, tangent_low, tangent_high in (
            self._wall_constraints
        ):
            distance_inside = float(np.dot(pelvis_xy - face, normal))
            tangent_position = pelvis_xy[tangent_axis]
            near_wall_segment = (
                tangent_low - WALL_TANGENT_MARGIN
                <= tangent_position
                <= tangent_high + WALL_TANGENT_MARGIN
            )
            inward_speed = float(np.dot(world_velocity, normal))
            if (
                distance_inside <= self.wall_margin
                and near_wall_segment
                and inward_speed < 0.0
            ):
                world_velocity -= inward_speed * normal

        projected = command.copy()
        projected[:2] = robot_to_world_xy.T @ world_velocity
        translation_speed = float(np.linalg.norm(projected[:2]))
        if translation_speed <= STAND_TRANSLATION_THRESHOLD:
            projected[:2] = 0.0
        else:
            activation_level = float(
                np.max(
                    np.abs(projected[:2]) / MINIMUM_WALKING_TRANSLATION
                )
            )
            if activation_level < 1.0:
                projected[:2] /= activation_level
            translation_limits = np.where(
                projected[:2] < 0.0,
                -COMMAND_LOW[:2],
                COMMAND_HIGH[:2],
            )
            range_scale = float(
                np.min(
                    translation_limits
                    / np.maximum(np.abs(projected[:2]), 1e-12)
                )
            )
            if range_scale < 1.0:
                projected[:2] *= range_scale
        return np.clip(projected, COMMAND_LOW, COMMAND_HIGH)

    def _action_to_command(self, action: np.ndarray) -> np.ndarray:
        command = normalized_action_to_command(action)
        return self._remove_wall_inward_velocity(command).astype(np.float32)
