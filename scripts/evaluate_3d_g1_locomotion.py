import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np

from src.soccer_3d.g1_locomotion import (
    CONTROL_TIMESTEP,
    PHYSICS_STEPS_PER_CONTROL,
    G1LocomotionController,
    reset_g1_for_locomotion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = (
    PROJECT_ROOT
    / "src"
    / "soccer_3d"
    / "assets"
    / "g1"
    / "inspection_scene.xml"
)
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "models" / "g1_locomotion" / "policy.onnx"
)


@dataclass(frozen=True)
class RolloutResult:
    command: np.ndarray
    mean_local_velocity: np.ndarray
    final_displacement: np.ndarray
    yaw_change: float
    minimum_pelvis_height: float
    minimum_upright_alignment: float
    fell: bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the isolated G1 locomotion controller.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--suite",
        action="store_true",
        help=(
            "Run fixed stand, forward, backward, lateral, and yaw checks "
            "headlessly."
        ),
    )
    args = parser.parse_args()
    if args.duration <= 1.0:
        parser.error("--duration must be greater than 1 second")
    if args.render and args.suite:
        parser.error("--render and --suite cannot be used together")
    if not args.policy.exists():
        parser.error(
            f"missing policy {args.policy}; run "
            "`make download-g1-locomotion-policy` first"
        )
    return args


def run_rollout(
    policy_path: Path,
    command: np.ndarray,
    duration: float,
    render: bool = False,
) -> RolloutResult:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    controller = G1LocomotionController(model, policy_path)
    reset_g1_for_locomotion(model, data, controller)
    controller.reset(data, command)

    pelvis_id = controller.pelvis_id
    initial_xy = data.xpos[pelvis_id, :2].copy()
    local_velocities = []
    pelvis_heights = []
    upright_alignments = []
    headings = []
    sample_after = 1.0
    physics_steps = round(duration / model.opt.timestep)

    def simulate(viewer=None):
        for physics_step in range(physics_steps):
            step_started_at = time.monotonic()
            if physics_step % PHYSICS_STEPS_PER_CONTROL == 0:
                controller.policy_step(data, command)

            data.ctrl[:] = controller.torques(data)
            mujoco.mj_step(model, data)

            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError("G1 produced a non-finite position")
            if not np.all(np.isfinite(data.qvel)):
                raise RuntimeError("G1 produced a non-finite velocity")

            pelvis_rotation = data.xmat[pelvis_id].reshape(3, 3)
            headings.append(
                float(np.arctan2(pelvis_rotation[1, 0], pelvis_rotation[0, 0]))
            )
            if data.time >= sample_after:
                local_velocities.append(
                    pelvis_rotation.T @ data.qvel[:3]
                )
            pelvis_heights.append(float(data.xpos[pelvis_id, 2]))
            upright_alignments.append(float(pelvis_rotation[2, 2]))

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                remaining = model.opt.timestep - (
                    time.monotonic() - step_started_at
                )
                if remaining > 0.0:
                    time.sleep(remaining)

    if render:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            simulate(viewer)
    else:
        simulate()

    final_xy = data.xpos[pelvis_id, :2].copy()
    minimum_height = min(pelvis_heights)
    minimum_upright = min(upright_alignments)
    mean_local_velocity = np.mean(local_velocities, axis=0)
    yaw_change = float(np.unwrap(headings)[-1] - np.unwrap(headings)[0])
    fell = minimum_height < 0.45 or minimum_upright < 0.5
    return RolloutResult(
        command=command.copy(),
        mean_local_velocity=mean_local_velocity,
        final_displacement=final_xy - initial_xy,
        yaw_change=yaw_change,
        minimum_pelvis_height=minimum_height,
        minimum_upright_alignment=minimum_upright,
        fell=fell,
    )


def print_result(result):
    command = result.command
    velocity = result.mean_local_velocity
    displacement = result.final_displacement
    print(
        f"command: vx={command[0]:+.2f}, vy={command[1]:+.2f}, "
        f"yaw_rate={command[2]:+.2f}"
    )
    print(
        f"mean local velocity: x={velocity[0]:+.3f}, "
        f"y={velocity[1]:+.3f}, z={velocity[2]:+.3f} m/s"
    )
    print(
        f"world displacement: x={displacement[0]:+.3f}, "
        f"y={displacement[1]:+.3f} m"
    )
    print(f"yaw change: {result.yaw_change:+.3f} rad")
    print(
        f"minimum pelvis height: {result.minimum_pelvis_height:.3f} m"
    )
    print(
        "minimum upright alignment: "
        f"{result.minimum_upright_alignment:.3f}"
    )
    print(f"fell: {result.fell}")


def validate_suite(results: list[RolloutResult]):
    stand, forward, backward, lateral, turning = results
    failures = []

    if stand.fell or np.linalg.norm(stand.final_displacement) >= 0.2:
        failures.append("stand: robot did not remain stable and nearly still")
    if forward.fell or forward.mean_local_velocity[0] <= 0.25:
        failures.append("forward: positive X command was not followed")
    if backward.fell or backward.mean_local_velocity[0] >= -0.25:
        failures.append("backward: negative X command was not followed")
    if lateral.fell or lateral.mean_local_velocity[1] <= 0.1:
        failures.append("lateral: positive Y command was not followed")
    if turning.fell or turning.yaw_change <= 0.4:
        failures.append("turning: positive yaw command did not produce a turn")

    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    args = parse_args()
    if args.suite:
        commands = (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
            np.array([-0.5, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.3, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.2], dtype=np.float32),
        )
        results = []
        for index, command in enumerate(commands):
            if index:
                print()
            result = run_rollout(args.policy, command, args.duration)
            results.append(result)
            print_result(result)
        validate_suite(results)
        print("\nlocomotion suite: passed")
    else:
        command = np.array(
            [args.vx, args.vy, args.yaw_rate],
            dtype=np.float32,
        )
        result = run_rollout(
            args.policy,
            command,
            args.duration,
            render=args.render,
        )
        print_result(result)


if __name__ == "__main__":
    main()
