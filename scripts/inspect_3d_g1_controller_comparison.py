import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np
from stable_baselines3 import PPO, SAC

from src.soccer_3d import G1SoccerEnv
from src.soccer_3d.g1_broad_pose import sample_broad_pose
from src.soccer_3d.g1_geometric_state_machine import (
    GeometricCommandStateMachine,
    physical_to_normalized,
)
from src.soccer_3d.g1_soccer_env import CONTROL_TIMESTEP

DEFAULT_PPO_MODEL = Path("models/ppo_3d_g1_soccer_benchmark_seed0.zip")
DEFAULT_SAC_MODEL = Path(
    "models/sac_3d_g1_soccer_broad_curriculum_seed0.zip"
)
BROAD_SAMPLE_AIM_Y_OFFSET = 0.25


@dataclass
class ControllerRun:
    label: str
    env: G1SoccerEnv
    model: PPO | SAC | None
    observation: np.ndarray
    info: dict
    geometric: GeometricCommandStateMachine | None = None
    done: bool = False

    @property
    def status(self) -> str:
        if self.info.get("goal", False):
            return "GOAL"
        if self.info.get("fell", False):
            return "FELL"
        if self.done:
            return "FAILED"
        return "RUNNING"

    def step(self):
        if self.done:
            return
        if self.geometric is None:
            action, _ = self.model.predict(
                self.observation,
                deterministic=True,
            )
        else:
            command = self.geometric.next_command()
            if command is None:
                self.done = True
                return
            action = physical_to_normalized(command)

        self.observation, _, terminated, truncated, self.info = (
            self.env.step(action)
        )
        self.done = terminated or truncated


class TripleViewer:
    def __init__(self, runs, playback_speed: float):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.runs = runs
        self.playback_speed = playback_speed
        self.window = glfw.create_window(
            1800,
            700,
            "PPO | SAC broad curriculum | Geometric",
            None,
            None,
        )
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the comparison window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.scenes = []
        self.contexts = []
        self.options = []
        self.cameras = []
        for run in runs:
            self.scenes.append(
                mujoco.MjvScene(run.env.model, maxgeom=10000)
            )
            self.contexts.append(
                mujoco.MjrContext(
                    run.env.model,
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                )
            )
            self.options.append(mujoco.MjvOption())
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = run.env._overview_camera_id
            self.cameras.append(camera)

    @property
    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def render(self, episode: int, episodes: int, wait: bool = True):
        frame_start = time.perf_counter()
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        column_width = width // len(self.runs)

        for index, run in enumerate(self.runs):
            x = index * column_width
            viewport_width = (
                width - x if index == len(self.runs) - 1 else column_width
            )
            viewport = mujoco.MjrRect(x, 0, viewport_width, height)
            mujoco.mjv_updateScene(
                run.env.model,
                run.env.data,
                self.options[index],
                None,
                self.cameras[index],
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scenes[index],
            )
            mujoco.mjr_render(
                viewport,
                self.scenes[index],
                self.contexts[index],
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                run.label,
                (
                    f"episode {episode}/{episodes}\n"
                    f"step {run.info.get('elapsed_steps', 0)}\n"
                    f"{run.status}"
                ),
                self.contexts[index],
            )

        glfw.swap_buffers(self.window)
        glfw.poll_events()
        if wait:
            frame_duration = CONTROL_TIMESTEP / self.playback_speed
            remaining = frame_duration - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                glfw.wait_events_timeout(remaining)

    def hold(self, duration: float):
        deadline = time.perf_counter() + duration
        while self.is_running and time.perf_counter() < deadline:
            glfw.wait_events_timeout(
                min(0.05, deadline - time.perf_counter())
            )

    def close(self):
        for context in self.contexts:
            context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render PPO, SAC, and geometric control on synchronized broad "
            "G1 soccer episodes."
        )
    )
    parser.add_argument("--ppo-model", type=Path, default=DEFAULT_PPO_MODEL)
    parser.add_argument("--sac-model", type=Path, default=DEFAULT_SAC_MODEL)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--episode-hold", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not args.ppo_model.exists():
        parser.error(f"PPO model not found: {args.ppo_model}")
    if not args.sac_model.exists():
        parser.error(f"SAC model not found: {args.sac_model}")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if args.episode_hold < 0.0:
        parser.error("--episode-hold cannot be negative")
    return args


def reset_runs(runs, pose, seed):
    g1_xy, ball_xy, g1_yaw = pose
    options = {
        "initial_g1_xy": g1_xy,
        "initial_ball_xy": ball_xy,
        "initial_g1_yaw": g1_yaw,
    }
    for run in runs:
        run.observation, run.info = run.env.reset(
            seed=seed,
            options=options,
        )
        run.done = False
        run.geometric = (
            GeometricCommandStateMachine(run.env, aim_y_offset=0.0)
            if run.model is None
            else None
        )


def run_comparison(args):
    models = (
        ("PPO", PPO.load(args.ppo_model)),
        ("SAC CURRICULUM", SAC.load(args.sac_model)),
        ("GEOMETRIC", None),
    )
    runs = []
    for label, model in models:
        env = G1SoccerEnv(
            max_episode_steps=args.max_episode_steps,
            observation_mode="soccer_state",
            reward_mode="goal",
        )
        runs.append(
            ControllerRun(
                label=label,
                env=env,
                model=model,
                observation=np.empty(0, dtype=np.float32),
                info={},
            )
        )

    viewer = None
    pose_rng = np.random.default_rng(args.seed)
    try:
        if not args.headless:
            viewer = TripleViewer(runs, args.playback_speed)
        for episode_index in range(args.episodes):
            pose = sample_broad_pose(
                pose_rng,
                BROAD_SAMPLE_AIM_Y_OFFSET,
            )
            reset_runs(
                runs,
                pose,
                args.seed + episode_index,
            )
            if viewer is not None:
                viewer.render(episode_index + 1, args.episodes)

            while not all(run.done for run in runs):
                for run in runs:
                    run.step()
                if viewer is not None:
                    if not viewer.is_running:
                        return
                    viewer.render(episode_index + 1, args.episodes)

            summary = ", ".join(
                f"{run.label}={run.status}@{run.info.get('elapsed_steps', 0)}"
                for run in runs
            )
            print(f"Episode {episode_index + 1}: {summary}", flush=True)
            if viewer is not None:
                viewer.render(
                    episode_index + 1,
                    args.episodes,
                    wait=False,
                )
                viewer.hold(args.episode_hold)
    finally:
        if viewer is not None:
            viewer.close()
        for run in runs:
            run.env.close()


def main():
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
