import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "soccer_3d"
    / "assets"
    / "g1"
    / "inspection_scene.xml"
)
STAND_KEYFRAME = "stand"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and simulate the isolated Unitree G1 model.",
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo passive viewer during the smoke simulation.",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    return args


def load_standing_g1():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    stand_keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        STAND_KEYFRAME,
    )
    if stand_keyframe_id < 0:
        raise RuntimeError(f"Missing {STAND_KEYFRAME!r} keyframe")

    mujoco.mj_resetDataKeyframe(model, data, stand_keyframe_id)
    mujoco.mj_forward(model, data)
    return model, data


def step_until(model, data, duration, viewer=None):
    end_time = data.time + duration
    while data.time < end_time:
        step_started_at = time.monotonic()
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError("G1 simulation produced a non-finite position")
        if not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("G1 simulation produced a non-finite velocity")

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            remaining_step_time = (
                model.opt.timestep - (time.monotonic() - step_started_at)
            )
            if remaining_step_time > 0.0:
                time.sleep(remaining_step_time)


def report(model, data):
    pelvis_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )
    pelvis_rotation = data.xmat[pelvis_id].reshape(3, 3)
    pelvis_up_alignment = pelvis_rotation[2, 2]

    print(
        f"Loaded G1: nq={model.nq}, nv={model.nv}, "
        f"joints={model.njnt}, actuators={model.nu}"
    )
    print(f"Simulated time: {data.time:.3f} s")
    print(f"Pelvis height: {data.xpos[pelvis_id, 2]:.3f} m")
    print(f"Pelvis upright alignment: {pelvis_up_alignment:.3f}")
    print(f"Active contacts: {data.ncon}")


def main():
    args = parse_args()
    model, data = load_standing_g1()

    if args.render:
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
                "inspection",
            )
            step_until(model, data, args.duration, viewer=viewer)
    else:
        step_until(model, data, args.duration)

    report(model, data)


if __name__ == "__main__":
    main()
