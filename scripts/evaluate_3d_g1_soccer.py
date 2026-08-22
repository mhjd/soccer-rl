import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np

from src.soccer_3d.g1_locomotion import (
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
    / "soccer_scene.xml"
)
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "models" / "g1_locomotion" / "policy.onnx"
)
SETTLE_DURATION = 1.0
WALK_DURATION = 4.0
STOP_DURATION = 1.0


@dataclass(frozen=True)
class SoccerIntegrationResult:
    ball_displacement: np.ndarray
    peak_ball_speed: float
    first_foot_contact_time: float | None
    goal_time: float | None
    minimum_pelvis_height: float
    minimum_upright_alignment: float
    fell: bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the G1 walking controller in a fixed soccer scene.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.vx <= 0.0:
        parser.error("--vx must be positive")
    if not args.policy.exists():
        parser.error(
            f"missing policy {args.policy}; run "
            "`make download-g1-locomotion-policy` first"
        )
    return args


def run_integration(
    policy_path: Path,
    forward_speed: float,
    render: bool = False,
) -> SoccerIntegrationResult:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    controller = G1LocomotionController(model, policy_path)
    reset_g1_for_locomotion(model, data, controller)

    zero_command = np.zeros(3, dtype=np.float32)
    walk_command = np.array(
        [forward_speed, 0.0, 0.0],
        dtype=np.float32,
    )
    controller.reset(data, zero_command)

    ball_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ball",
    )
    ball_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "ball_geom",
    )
    goal_line_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "goal_line",
    )
    foot_body_ids = {
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_ankle_roll_link",
        ),
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_ankle_roll_link",
        ),
    }
    if (
        ball_body_id < 0
        or ball_geom_id < 0
        or goal_line_site_id < 0
        or min(foot_body_ids) < 0
    ):
        raise RuntimeError("Soccer scene is missing required named elements")

    initial_ball_position = data.xpos[ball_body_id].copy()
    pelvis_heights = []
    upright_alignments = []
    peak_ball_speed = 0.0
    first_foot_contact_time = None
    goal_time = None
    total_duration = SETTLE_DURATION + WALK_DURATION + STOP_DURATION
    physics_steps = round(total_duration / model.opt.timestep)

    def command_at(simulation_time: float) -> np.ndarray:
        if simulation_time < SETTLE_DURATION:
            return zero_command
        if simulation_time < SETTLE_DURATION + WALK_DURATION:
            return walk_command
        return zero_command

    def simulate(viewer=None):
        nonlocal first_foot_contact_time, goal_time, peak_ball_speed

        for physics_step in range(physics_steps):
            step_started_at = time.monotonic()
            command = command_at(data.time)
            if physics_step % PHYSICS_STEPS_PER_CONTROL == 0:
                controller.policy_step(data, command)

            data.ctrl[:] = controller.torques(data)
            mujoco.mj_step(model, data)

            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError("G1 produced a non-finite position")
            if not np.all(np.isfinite(data.qvel)):
                raise RuntimeError("G1 produced a non-finite velocity")

            pelvis_rotation = data.xmat[controller.pelvis_id].reshape(3, 3)
            pelvis_heights.append(
                float(data.xpos[controller.pelvis_id, 2])
            )
            upright_alignments.append(float(pelvis_rotation[2, 2]))
            peak_ball_speed = max(
                peak_ball_speed,
                float(np.linalg.norm(data.cvel[ball_body_id, 3:])),
            )

            if goal_time is None:
                ball_position = data.xpos[ball_body_id]
                ball_radius = model.geom_size[ball_geom_id, 0]
                goal_position = data.site_xpos[goal_line_site_id]
                goal_half_width = model.site_size[goal_line_site_id, 1]
                ball_crossed_line = (
                    ball_position[0] - ball_radius > goal_position[0]
                )
                ball_inside_posts = (
                    abs(ball_position[1] - goal_position[1])
                    < goal_half_width - ball_radius
                )
                if ball_crossed_line and ball_inside_posts:
                    goal_time = float(data.time)

            if first_foot_contact_time is None:
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if contact.geom1 == ball_geom_id:
                        other_geom_id = contact.geom2
                    elif contact.geom2 == ball_geom_id:
                        other_geom_id = contact.geom1
                    else:
                        continue

                    if model.geom_bodyid[other_geom_id] in foot_body_ids:
                        first_foot_contact_time = float(data.time)
                        break

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
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "soccer_overview",
            )
            simulate(viewer)
    else:
        simulate()

    ball_displacement = data.xpos[ball_body_id] - initial_ball_position
    minimum_height = min(pelvis_heights)
    minimum_upright = min(upright_alignments)
    fell = minimum_height < 0.45 or minimum_upright < 0.5
    return SoccerIntegrationResult(
        ball_displacement=ball_displacement,
        peak_ball_speed=peak_ball_speed,
        first_foot_contact_time=first_foot_contact_time,
        goal_time=goal_time,
        minimum_pelvis_height=minimum_height,
        minimum_upright_alignment=minimum_upright,
        fell=fell,
    )


def print_result(result: SoccerIntegrationResult):
    displacement = result.ball_displacement
    contact = (
        "none"
        if result.first_foot_contact_time is None
        else f"{result.first_foot_contact_time:.3f} s"
    )
    print(
        "ball displacement: "
        f"x={displacement[0]:+.3f}, "
        f"y={displacement[1]:+.3f}, "
        f"z={displacement[2]:+.3f} m"
    )
    print(f"peak ball speed: {result.peak_ball_speed:.3f} m/s")
    print(f"first foot-ball contact: {contact}")
    goal = "not scored" if result.goal_time is None else f"{result.goal_time:.3f} s"
    print(f"goal: {goal}")
    print(
        f"minimum pelvis height: {result.minimum_pelvis_height:.3f} m"
    )
    print(
        "minimum upright alignment: "
        f"{result.minimum_upright_alignment:.3f}"
    )
    print(f"fell: {result.fell}")


def validate_result(result: SoccerIntegrationResult):
    failures = []
    if result.fell:
        failures.append("the G1 fell during the scripted soccer rollout")
    if result.first_foot_contact_time is None:
        failures.append("no foot-ball contact was detected")
    if result.goal_time is None:
        failures.append("the complete ball did not cross the goal line")
    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    args = parse_args()
    result = run_integration(
        args.policy,
        forward_speed=args.vx,
        render=args.render,
    )
    print_result(result)
    validate_result(result)
    print("g1 soccer integration: passed")


if __name__ == "__main__":
    main()
