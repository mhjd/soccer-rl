import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np
from stable_baselines3 import PPO

from src.soccer_3d import G1SoccerEnv
from src.soccer_3d.g1_soccer_env import (
    APPROACH_DISTANCE,
    APPROACH_LATERAL_OFFSET,
    CONTROL_TIMESTEP,
)
from src.soccer_3d.g1_locomotion import COMMAND_HIGH, COMMAND_LOW


def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class Situation:
    pelvis_xy: np.ndarray
    yaw: float
    ball_xy: np.ndarray
    approach_xy: np.ndarray
    goal_xy: np.ndarray
    shot_heading: float
    ball_relative: np.ndarray
    approach_relative: np.ndarray
    heading_error: float


@dataclass
class TraceSample:
    attempt: int
    phase: str
    step: int
    pelvis_xy: np.ndarray
    ball_xy: np.ndarray
    approach_xy: np.ndarray
    yaw: float
    heading_error: float
    command: np.ndarray
    contact: bool
    goal: bool
    fell: bool


class GeometricTrace:
    """Observe controller phases without affecting their decisions."""

    def __init__(self):
        self.attempt = 0
        self.phase = "uninitialized"
        self.samples = []

    def start_attempt(self, attempt):
        self.attempt = attempt

    def start_phase(self, phase, pilot):
        self.phase = phase
        self.record(
            pilot.read_situation(),
            np.zeros(3, dtype=np.float32),
            pilot.info,
        )

    def record(self, situation, command, info):
        self.samples.append(
            TraceSample(
                attempt=self.attempt,
                phase=self.phase,
                step=int(info.get("elapsed_steps", 0)),
                pelvis_xy=situation.pelvis_xy.copy(),
                ball_xy=situation.ball_xy.copy(),
                approach_xy=situation.approach_xy.copy(),
                yaw=situation.yaw,
                heading_error=situation.heading_error,
                command=np.asarray(command, dtype=np.float32).copy(),
                contact=bool(info.get("ball_contact_occurred", False)),
                goal=bool(info.get("goal", False)),
                fell=bool(info.get("fell", False)),
            )
        )

    def samples_for(self, phase, attempt=None):
        return [
            sample
            for sample in self.samples
            if sample.phase == phase
            and (attempt is None or sample.attempt == attempt)
        ]


DEFAULT_MODEL_PATH = Path("models/ppo_3d_g1_soccer_executable_commands.zip")
BALL_PATH_CLEARANCE = 0.45
BALL_DETOUR_RADIUS = 0.8
DETOUR_SIDE_EPSILON = 0.05


class PairedViewer:
    def __init__(self, policy_env, geometric_env, playback_speed):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        self.policy_env = policy_env
        self.geometric_env = geometric_env
        self.playback_speed = playback_speed
        self.window = glfw.create_window(
            1600,
            700,
            "PPO policy (left) | geometric controller (right)",
            None,
            None,
        )
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the comparison window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.policy_scene = mujoco.MjvScene(
            policy_env.model,
            maxgeom=10000,
        )
        self.geometric_scene = mujoco.MjvScene(
            geometric_env.model,
            maxgeom=10000,
        )
        self.policy_context = mujoco.MjrContext(
            policy_env.model,
            mujoco.mjtFontScale.mjFONTSCALE_150,
        )
        self.geometric_context = mujoco.MjrContext(
            geometric_env.model,
            mujoco.mjtFontScale.mjFONTSCALE_150,
        )
        self.policy_option = mujoco.MjvOption()
        self.geometric_option = mujoco.MjvOption()
        self.policy_camera = self._fixed_camera(policy_env)
        self.geometric_camera = self._fixed_camera(geometric_env)

    @staticmethod
    def _fixed_camera(env):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = env._overview_camera_id
        return camera

    @property
    def is_running(self):
        return not glfw.window_should_close(self.window)

    @staticmethod
    def _status(info, done):
        if info.get("goal", False):
            return "GOAL"
        if info.get("fell", False):
            return "FELL"
        if done:
            return "FAILED"
        return "running"

    def render(
        self,
        *,
        seed,
        policy_info,
        policy_done,
        geometric_info,
        geometric_done,
    ):
        if not self.is_running:
            return False

        frame_start = time.perf_counter()
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        left_width = width // 2
        left_viewport = mujoco.MjrRect(0, 0, left_width, height)
        right_viewport = mujoco.MjrRect(
            left_width,
            0,
            width - left_width,
            height,
        )

        mujoco.mjv_updateScene(
            self.policy_env.model,
            self.policy_env.data,
            self.policy_option,
            None,
            self.policy_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.policy_scene,
        )
        mujoco.mjr_render(
            left_viewport,
            self.policy_scene,
            self.policy_context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            left_viewport,
            "PPO POLICY",
            (
                f"seed {seed}\n"
                f"step {policy_info.get('elapsed_steps', 0)}\n"
                f"{self._status(policy_info, policy_done)}"
            ),
            self.policy_context,
        )

        mujoco.mjv_updateScene(
            self.geometric_env.model,
            self.geometric_env.data,
            self.geometric_option,
            None,
            self.geometric_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.geometric_scene,
        )
        mujoco.mjr_render(
            right_viewport,
            self.geometric_scene,
            self.geometric_context,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            right_viewport,
            "GEOMETRIC CONTROLLER",
            (
                f"seed {seed}\n"
                f"step {geometric_info.get('elapsed_steps', 0)}\n"
                f"{self._status(geometric_info, geometric_done)}"
            ),
            self.geometric_context,
        )
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        frame_duration = CONTROL_TIMESTEP / self.playback_speed
        remaining = frame_duration - (time.perf_counter() - frame_start)
        if remaining > 0.0:
            glfw.wait_events_timeout(remaining)
        return self.is_running

    def hold(self, duration):
        deadline = time.perf_counter() + duration
        while self.is_running and time.perf_counter() < deadline:
            glfw.wait_events_timeout(
                min(0.05, deadline - time.perf_counter())
            )

    def close(self):
        self.policy_context.free()
        self.geometric_context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


class PilotTools:
    def __init__(
        self,
        env,
        observation,
        info,
        aim_y_offset=0.0,
        verbose=False,
        recorded_actions=None,
        trace=None,
    ):
        self.env = env
        self.observation = observation
        self.info = info
        self.aim_y_offset = aim_y_offset
        self.verbose = verbose
        self.recorded_actions = recorded_actions
        self.trace = trace
        self.episode_ended = False

    def read_situation(self):
        pelvis_id = self.env.controller.pelvis_id
        pelvis_xy = self.env.data.xpos[pelvis_id, :2].copy()
        rotation = self.env.data.xmat[pelvis_id].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        ball_xy = self.env.data.xpos[self.env._ball_body_id, :2].copy()
        goal_xy = self.env.data.site_xpos[
            self.env._goal_line_site_id, :2
        ].copy()
        goal_xy[1] += self.aim_y_offset
        shot_direction = goal_xy - ball_xy
        shot_direction /= np.linalg.norm(shot_direction)
        shot_lateral = np.array([-shot_direction[1], shot_direction[0]])
        approach_xy = (
            ball_xy
            - APPROACH_DISTANCE * shot_direction
            + APPROACH_LATERAL_OFFSET * shot_lateral
        )
        approach_relative = rotation[:2, :2].T @ (
            approach_xy - pelvis_xy
        )
        shot_heading = float(
            np.arctan2(
                goal_xy[1] - ball_xy[1],
                goal_xy[0] - ball_xy[0],
            )
        )
        return Situation(
            pelvis_xy=pelvis_xy,
            yaw=yaw,
            ball_xy=ball_xy,
            approach_xy=approach_xy,
            goal_xy=goal_xy,
            shot_heading=shot_heading,
            ball_relative=self.observation[:2].copy(),
            approach_relative=approach_relative,
            heading_error=wrap_angle(shot_heading - yaw),
        )

    @staticmethod
    def physical_to_normalized(command):
        scale = np.where(command < 0.0, -COMMAND_LOW, COMMAND_HIGH)
        return np.divide(
            command,
            scale,
            out=np.zeros_like(command),
            where=scale != 0.0,
        ).astype(np.float32)

    def apply_command(self, command, steps=1):
        action = self.physical_to_normalized(
            np.asarray(command, dtype=np.float32)
        )
        for _ in range(steps):
            if self.recorded_actions is not None:
                self.recorded_actions.append(action.copy())
            self.observation, _, terminated, truncated, self.info = (
                self.env.step(action)
            )
            self.episode_ended = terminated or truncated
            if self.trace is not None:
                self.trace.record(
                    self.read_situation(),
                    command,
                    self.info,
                )
            if self.episode_ended:
                break
        return self.episode_ended

    def command_toward(self, situation, target_xy, face_shot=True):
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
        return np.array([translation[0], translation[1], yaw_rate])

    def move_to_point(
        self,
        target_provider,
        *,
        position_tolerance=0.08,
        max_steps=100,
        face_shot=True,
    ):
        for step in range(max_steps):
            situation = self.read_situation()
            target_xy = np.asarray(target_provider(situation), dtype=float)
            error_world = target_xy - situation.pelvis_xy
            distance = float(np.linalg.norm(error_world))
            if distance <= position_tolerance:
                self.apply_command(np.zeros(3, dtype=np.float32), steps=2)
                return True

            ended = self.apply_command(
                self.command_toward(
                    situation,
                    target_xy,
                    face_shot=face_shot,
                )
            )
            if ended:
                return False
            if self.verbose and step % 10 == 9:
                self.print_summary(prefix="tracking")
        return False

    def move_to_approach(self):
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
                    situation.ball_xy - BALL_DETOUR_RADIUS * shot_lateral,
                    situation.ball_xy + BALL_DETOUR_RADIUS * shot_lateral,
                )
                detour_side = (
                    -1.0
                    if abs(candidates[0][1]) <= abs(candidates[1][1])
                    else 1.0
                )

            def detour_target(current_situation):
                current_shot_direction = (
                    current_situation.goal_xy - current_situation.ball_xy
                )
                current_shot_direction /= np.linalg.norm(
                    current_shot_direction
                )
                current_shot_lateral = np.array(
                    [
                        -current_shot_direction[1],
                        current_shot_direction[0],
                    ]
                )
                return (
                    current_situation.ball_xy
                    + detour_side
                    * BALL_DETOUR_RADIUS
                    * current_shot_lateral
                )

            if self.trace is not None:
                self.trace.start_phase("detour", self)
            if not self.move_to_point(detour_target, max_steps=100):
                return False

        if self.trace is not None:
            self.trace.start_phase("approach", self)
        return self.move_to_point(
            lambda situation: situation.approach_xy,
            max_steps=120,
        )

    def align_with_shot(self, tolerance=0.12, max_cycles=60):
        for _ in range(max_cycles):
            situation = self.read_situation()
            if abs(situation.heading_error) <= tolerance:
                return True
            yaw_rate = np.sign(situation.heading_error) * 0.2
            if self.apply_command(np.array([0.35, 0.0, yaw_rate]), steps=2):
                return False
            if self.apply_command(np.array([-0.35, 0.0, yaw_rate]), steps=2):
                return False
        return abs(self.read_situation().heading_error) <= tolerance

    def drive_through_ball(self, max_steps=100):
        def beyond_ball(situation):
            direction = situation.goal_xy - situation.ball_xy
            direction /= np.linalg.norm(direction)
            return situation.ball_xy + 1.5 * direction

        return self.move_to_point(
            beyond_ball,
            position_tolerance=0.05,
            max_steps=max_steps,
            face_shot=True,
        )

    def solve(self):
        attempt = 1
        if self.trace is not None:
            self.trace.start_attempt(attempt)
        if self.verbose:
            print("primitive=move_to_approach", flush=True)
        reached = self.move_to_approach()
        if self.verbose:
            self.print_summary()
        if not reached or self.info.get("goal") or self.info.get("fell"):
            return reached, False, self.info
        if self.trace is not None:
            self.trace.start_phase("alignment", self)
        if self.verbose:
            print("primitive=align_with_shot", flush=True)
        aligned = self.align_with_shot()
        if self.verbose:
            self.print_summary()
        if not aligned or self.info.get("goal") or self.info.get("fell"):
            return reached, aligned, self.info
        if self.trace is not None:
            self.trace.start_phase("drive_through", self)
        if self.verbose:
            print("primitive=drive_through_ball", flush=True)
        self.drive_through_ball()
        if self.verbose:
            self.print_summary()
        self.info = dict(self.info)
        self.info["geometric_attempts"] = attempt
        self.info["drive_outcome"] = "completed"
        return reached, aligned, self.info

    def print_summary(self, prefix="state"):
        situation = self.read_situation()
        print(
            f"{prefix}: step={self.info.get('elapsed_steps', 0)} "
            f"pelvis={np.round(situation.pelvis_xy, 2)} "
            f"yaw={situation.yaw:+.2f} "
            f"ball={np.round(situation.ball_xy, 2)} "
            f"approach_error={np.round(situation.approach_relative, 2)} "
            f"heading_error={situation.heading_error:+.2f} "
            f"contact={self.info.get('ball_contact_occurred', False)} "
            f"goal={self.info.get('goal', False)} "
            f"fell={self.info.get('fell', False)}",
            flush=True,
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retry failed PPO recovery starts with a deterministic "
            "geometric controller."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--failed-episodes",
        type=int,
        help="Stop after retrying this many failed PPO episodes.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aim-y-offset", type=float, default=0.25)
    parser.add_argument(
        "--render-failures",
        action="store_true",
        help=(
            "Replay every failed PPO start with the PPO policy on the left "
            "and the geometric controller on the right."
        ),
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Comparison playback speed relative to real time.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.failed_episodes is not None and args.failed_episodes <= 0:
        parser.error("--failed-episodes must be positive")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    return args


def reset_recovery(env, seed):
    return env.reset(
        seed=seed,
        options={"recovery_state_difficulty": 1.0},
    )


def run_policy_episode(env, model, seed):
    observation, _ = reset_recovery(env, seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
    return info


def run_geometric_episode(env, seed, aim_y_offset, verbose):
    observation, _ = reset_recovery(env, seed)
    info = {
        "elapsed_steps": 0,
        "command": np.zeros(3),
        "ball_contact_occurred": False,
        "goal": False,
        "fell": False,
    }
    recorded_actions = []
    pilot = PilotTools(
        env,
        observation,
        info,
        aim_y_offset=aim_y_offset,
        verbose=verbose,
        recorded_actions=recorded_actions,
    )
    reached, aligned, info = pilot.solve()
    return reached, aligned, info, recorded_actions


def initial_episode_info():
    return {
        "elapsed_steps": 0,
        "ball_contact_occurred": False,
        "goal": False,
        "fell": False,
    }


def render_comparison(viewer, model, seed, geometric_actions):
    policy_observation, _ = reset_recovery(viewer.policy_env, seed)
    reset_recovery(viewer.geometric_env, seed)
    policy_info = initial_episode_info()
    geometric_info = initial_episode_info()
    policy_done = False
    geometric_done = False
    geometric_action_index = 0

    while not (policy_done and geometric_done):
        if not policy_done:
            policy_action, _ = model.predict(
                policy_observation,
                deterministic=True,
            )
            (
                policy_observation,
                _,
                policy_terminated,
                policy_truncated,
                policy_info,
            ) = viewer.policy_env.step(policy_action)
            policy_done = policy_terminated or policy_truncated

        if not geometric_done:
            if geometric_action_index >= len(geometric_actions):
                geometric_done = True
            else:
                geometric_action = geometric_actions[
                    geometric_action_index
                ]
                geometric_action_index += 1
                (
                    _,
                    _,
                    geometric_terminated,
                    geometric_truncated,
                    geometric_info,
                ) = viewer.geometric_env.step(geometric_action)
                geometric_done = (
                    geometric_terminated or geometric_truncated
                )

        if not viewer.render(
            seed=seed,
            policy_info=policy_info,
            policy_done=policy_done,
            geometric_info=geometric_info,
            geometric_done=geometric_done,
        ):
            return False

    viewer.hold(1.0)
    return viewer.is_running


def main():
    args = parse_args()
    model = PPO.load(args.model)
    env = G1SoccerEnv(
        render_mode=None,
        randomize_initial_positions=True,
        recovery_start_probability=1.0,
        observation_mode="soccer_state",
        max_episode_steps=200,
    )
    policy_display_env = None
    geometric_display_env = None
    viewer = None
    if args.render_failures:
        policy_display_env = G1SoccerEnv(
            render_mode=None,
            randomize_initial_positions=True,
            recovery_start_probability=1.0,
            observation_mode="soccer_state",
            max_episode_steps=200,
        )
        geometric_display_env = G1SoccerEnv(
            render_mode=None,
            randomize_initial_positions=True,
            recovery_start_probability=1.0,
            observation_mode="soccer_state",
            max_episode_steps=200,
        )
        viewer = PairedViewer(
            policy_display_env,
            geometric_display_env,
            args.playback_speed,
        )
    policy_goals = 0
    policy_falls = 0
    failed_seeds = []
    geometric_goals = 0
    geometric_falls = 0
    geometric_contacts = 0
    approach_reached = 0
    shot_aligned = 0
    goal_steps = []
    attempted_episodes = 0

    try:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            attempted_episodes += 1
            policy_info = run_policy_episode(env, model, seed)
            policy_goals += int(policy_info["goal"])
            policy_falls += int(policy_info["fell"])
            if policy_info["goal"]:
                continue

            failed_seeds.append(seed)
            (
                reached,
                aligned,
                geometric_info,
                geometric_actions,
            ) = run_geometric_episode(
                env,
                seed,
                args.aim_y_offset,
                args.verbose,
            )
            approach_reached += int(reached)
            shot_aligned += int(aligned)
            geometric_goals += int(geometric_info["goal"])
            geometric_falls += int(geometric_info["fell"])
            geometric_contacts += int(
                geometric_info["ball_contact_occurred"]
            )
            if geometric_info["goal"]:
                goal_steps.append(geometric_info["elapsed_steps"])
            if args.verbose:
                print(
                    f"seed={seed} policy_goal=False "
                    f"geometric_goal={geometric_info['goal']} "
                    f"steps={geometric_info['elapsed_steps']}",
                    flush=True,
                )
            if viewer is not None and not render_comparison(
                viewer,
                model,
                seed,
                geometric_actions,
            ):
                break
            if (
                args.failed_episodes is not None
                and len(failed_seeds) >= args.failed_episodes
            ):
                break
    finally:
        env.close()
        if viewer is not None:
            viewer.close()
        if policy_display_env is not None:
            policy_display_env.close()
        if geometric_display_env is not None:
            geometric_display_env.close()

    failed_count = len(failed_seeds)
    print(f"PPO goals: {policy_goals}/{attempted_episodes}")
    print(f"PPO falls: {policy_falls}/{attempted_episodes}")
    print(f"PPO failed seeds retried: {failed_count}")
    print(f"Approach points reached: {approach_reached}/{failed_count}")
    print(f"Shot headings aligned: {shot_aligned}/{failed_count}")
    print(f"Ball contacts: {geometric_contacts}/{failed_count}")
    print(f"Deterministic goals: {geometric_goals}/{failed_count}")
    print(f"Deterministic falls: {geometric_falls}/{failed_count}")
    rescue_rate = geometric_goals / failed_count if failed_count else 0.0
    print(f"Deterministic rescue rate: {rescue_rate:.1%}")
    print(
        "Mean deterministic steps to rescued goal: "
        f"{np.mean(goal_steps):.1f}" if goal_steps else "n/a"
    )
    print("First failed PPO seeds:", failed_seeds[:20])


if __name__ == "__main__":
    main()
