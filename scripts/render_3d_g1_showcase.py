import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import SAC

from src.soccer_3d import G1CommandGovernorEnv
from src.soccer_3d.g1_locomotion import PHYSICS_STEPS_PER_CONTROL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_SCENE_PATH = (
    PROJECT_ROOT
    / "src"
    / "soccer_3d"
    / "assets"
    / "g1"
    / "showcase_scene.xml"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "sac_3d_g1_soccer_broad_curriculum_seed0.zip"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "media" / "g1-soccer-showcase.mp4"
DEFAULT_POSTER_PATH = (
    PROJECT_ROOT / "media" / "g1-soccer-showcase-poster.jpg"
)
INTRO_DURATION = 4.0
LIVE_OUTRO_DURATION = 1.4
FROZEN_OUTRO_DURATION = 0.8
CAMERA_RESPONSE = 7.0


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def interpolate(start, end, amount: float):
    return start + (end - start) * smoothstep(amount)


class VideoWriter:
    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vf",
                ("eq=contrast=1.045:saturation=1.08:brightness=0.008,"
                "vignette=PI/7"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8")
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.strip()}")


def save_poster(frame: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = frame.shape
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        input=np.ascontiguousarray(frame).tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not write poster: {result.stderr.decode().strip()}"
        )


class CinematicRenderer:
    def __init__(
        self,
        env: G1CommandGovernorEnv,
        output_path: Path,
        poster_path: Path,
        width: int,
        height: int,
        fps: int,
    ):
        self.env = env
        self.fps = fps
        self.renderer = mujoco.Renderer(
            env.model,
            height=height,
            width=width,
        )
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[3] = 0
        self.writer = VideoWriter(output_path, width, height, fps)
        self.poster_path = poster_path
        self.poster_frame = None
        self.frame_count = 0
        self.camera_state = None
        self.camera_velocity = np.zeros(6, dtype=np.float64)

    def _scene_points(self):
        pelvis = self.env.data.xpos[self.env.controller.pelvis_id].copy()
        ball = self.env.data.xpos[self.env._ball_body_id].copy()
        goal = self.env.data.site_xpos[self.env._goal_line_site_id].copy()
        return pelvis, ball, goal

    def _follow_camera_target(
        self,
        lookat: np.ndarray,
        distance: float,
        azimuth: float,
        elevation: float,
    ):
        target = np.concatenate(
            [lookat, np.array([distance, azimuth, elevation])]
        )
        if self.camera_state is None:
            self.camera_state = target.copy()
        else:
            timestep = 1.0 / self.fps
            omega_squared = CAMERA_RESPONSE * CAMERA_RESPONSE
            damping = 1.0 + 2.0 * timestep * CAMERA_RESPONSE
            target_weight = timestep * timestep * omega_squared
            inverse_determinant = 1.0 / (damping + target_weight)
            previous_state = self.camera_state
            self.camera_state = (
                damping * previous_state
                + timestep * self.camera_velocity
                + target_weight * target
            ) * inverse_determinant
            self.camera_velocity = (
                self.camera_velocity
                + timestep * omega_squared * (target - previous_state)
            ) * inverse_determinant

        self.camera.lookat[:] = self.camera_state[:3]
        self.camera.distance = float(self.camera_state[3])
        self.camera.azimuth = float(self.camera_state[4])
        self.camera.elevation = float(self.camera_state[5])

    def _set_intro_camera(self, elapsed: float):
        pelvis, ball, goal = self._scene_points()
        pullback = smoothstep((elapsed - 0.55) / (INTRO_DURATION - 0.55))
        close_target = ball + np.array([0.0, 0.0, 0.025])
        wide_target = 0.35 * pelvis + 0.40 * ball + 0.25 * goal
        wide_target[2] = 0.55
        lookat = interpolate(
            close_target,
            wide_target,
            pullback,
        )
        self._follow_camera_target(
            lookat,
            float(interpolate(0.48, 4.45, pullback)),
            float(interpolate(142.0, 128.0, pullback)),
            float(interpolate(-12.0, -16.0, pullback)),
        )

    def _set_action_camera(self, elapsed: float, expected_duration: float):
        pelvis, ball, goal = self._scene_points()
        progress = smoothstep(elapsed / max(expected_duration, 1e-6))
        target = 0.30 * pelvis + 0.45 * ball + 0.25 * goal
        target[2] = 0.58
        self._follow_camera_target(
            target,
            float(interpolate(4.45, 4.1, progress)),
            float(interpolate(128.0, 118.0, progress)),
            float(interpolate(-16.0, -14.0, progress)),
        )

    def _set_outro_camera(self, elapsed: float, duration: float):
        pelvis, ball, goal = self._scene_points()
        progress = smoothstep(elapsed / max(duration, 1e-6))
        target = 0.48 * pelvis + 0.34 * ball + 0.18 * goal
        target[2] = 0.62
        self._follow_camera_target(
            target,
            float(interpolate(4.1, 4.8, progress)),
            float(interpolate(118.0, 28.0, progress)),
            float(interpolate(-14.0, -22.0, progress)),
        )

    def _render(self):
        self.renderer.update_scene(
            self.env.data,
            camera=self.camera,
            scene_option=self.scene_option,
        )
        frame = self.renderer.render().copy()
        self.writer.write(frame)
        self.frame_count += 1
        return frame

    def render_intro(self):
        for frame_index in range(round(INTRO_DURATION * self.fps)):
            elapsed = frame_index / self.fps
            self._set_intro_camera(elapsed)
            frame = self._render()
            if frame_index == round(0.78 * INTRO_DURATION * self.fps):
                self.poster_frame = frame

    def render_action_frame(self, elapsed: float, expected_duration: float):
        self._set_action_camera(elapsed, expected_duration)
        self._render()

    def render_outro_frame(self, elapsed: float, duration: float):
        self._set_outro_camera(elapsed, duration)
        self._render()

    def close(self):
        self.renderer.close()
        self.writer.close()
        if self.poster_frame is None:
            raise RuntimeError("No poster frame was captured")
        save_poster(self.poster_frame, self.poster_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a cinematic G1 soccer portfolio video.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--poster", type=Path, default=DEFAULT_POSTER_PATH)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"model not found: {args.model}")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    return args


def render_showcase(args):
    model = SAC.load(args.model)
    env = G1CommandGovernorEnv(
        scene_path=SHOWCASE_SCENE_PATH,
        max_episode_steps=100,
        observation_mode="soccer_state",
        reward_mode="goal",
    )
    renderer = None
    try:
        observation, _ = env.reset(seed=args.seed)
        renderer = CinematicRenderer(
            env,
            args.output,
            args.poster,
            args.width,
            args.height,
            args.fps,
        )
        renderer.render_intro()

        action_start_time = float(env.data.time)
        next_frame_time = action_start_time
        expected_action_duration = 3.0

        def capture_action_frame():
            nonlocal next_frame_time
            while env.data.time + 1e-9 >= next_frame_time:
                elapsed = next_frame_time - action_start_time
                renderer.render_action_frame(
                    elapsed,
                    expected_action_duration,
                )
                next_frame_time += 1.0 / args.fps

        env.set_physics_step_callback(capture_action_frame)
        terminated = False
        truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)

        env.set_physics_step_callback(None)
        if not info.get("goal", False):
            raise RuntimeError(f"Showcase episode did not score: {info}")

        live_outro_start = float(env.data.time)
        next_outro_frame = live_outro_start
        zero_command = np.zeros(3, dtype=np.float32)
        live_steps = round(LIVE_OUTRO_DURATION / env.model.opt.timestep)
        for physics_step in range(live_steps):
            if physics_step % PHYSICS_STEPS_PER_CONTROL == 0:
                env.controller.policy_step(env.data, zero_command)
            env.data.ctrl[:] = env.controller.torques(env.data)
            mujoco.mj_step(env.model, env.data)
            while env.data.time + 1e-9 >= next_outro_frame:
                elapsed = next_outro_frame - live_outro_start
                renderer.render_outro_frame(
                    elapsed,
                    LIVE_OUTRO_DURATION + FROZEN_OUTRO_DURATION,
                )
                next_outro_frame += 1.0 / args.fps

        frozen_frames = round(FROZEN_OUTRO_DURATION * args.fps)
        for frame_index in range(frozen_frames):
            elapsed = LIVE_OUTRO_DURATION + frame_index / args.fps
            renderer.render_outro_frame(
                elapsed,
                LIVE_OUTRO_DURATION + FROZEN_OUTRO_DURATION,
            )

        duration = renderer.frame_count / args.fps
        print(f"Goal after {info['elapsed_steps']} high-level steps")
        print(f"Rendered {renderer.frame_count} frames ({duration:.2f} s)")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def main():
    render_showcase(parse_args())


if __name__ == "__main__":
    main()
