from pathlib import Path

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from src.soccer_3d import G1KickResidualEnv
from src.soccer_3d.g1_locomotion import (
    EFFORT_LIMIT_HARDWARE,
    LEG_JOINT_COUNT,
    MAX_LEG_JOINT_RESIDUAL,
    PHYSICS_STEPS_PER_CONTROL,
    G1LocomotionController,
    reset_g1_for_locomotion,
)
from src.soccer_3d.g1_soccer_env import DEFAULT_POLICY_PATH, SCENE_PATH


def create_system(policy_path: Path):
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    controller = G1LocomotionController(model, policy_path)
    reset_g1_for_locomotion(model, data, controller)
    return model, data, controller


def advance_control_step(model, data, controller, command, residual=None):
    controller.policy_step(data, command, residual)
    for _ in range(PHYSICS_STEPS_PER_CONTROL):
        data.ctrl[:] = controller.torques(data)
        mujoco.mj_step(model, data)


def check_zero_residual_equivalence(policy_path: Path):
    base_model, base_data, base = create_system(policy_path)
    residual_model, residual_data, residual = create_system(policy_path)
    command = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    zero = np.zeros(LEG_JOINT_COUNT, dtype=np.float32)
    base.reset(base_data, command)
    residual.reset(residual_data, command)

    for _ in range(100):
        advance_control_step(base_model, base_data, base, command)
        advance_control_step(
            residual_model,
            residual_data,
            residual,
            command,
            zero,
        )

    if not np.array_equal(base_data.qpos, residual_data.qpos):
        raise RuntimeError("A zero residual changed G1 positions")
    if not np.array_equal(base_data.qvel, residual_data.qvel):
        raise RuntimeError("A zero residual changed G1 velocities")


def check_target_composition(policy_path: Path):
    model, data, controller = create_system(policy_path)
    command = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    residual = np.linspace(
        -MAX_LEG_JOINT_RESIDUAL,
        MAX_LEG_JOINT_RESIDUAL,
        LEG_JOINT_COUNT,
        dtype=np.float32,
    )
    controller.reset(data, command)
    controller.policy_step(data, command, residual)

    target_difference = (
        controller.target_joint_position_hardware
        - controller.base_target_joint_position_hardware
    )
    if not np.allclose(target_difference[:LEG_JOINT_COUNT], residual):
        raise RuntimeError("Residual values were not added to leg targets")
    if not np.array_equal(
        target_difference[LEG_JOINT_COUNT:],
        np.zeros(29 - LEG_JOINT_COUNT),
    ):
        raise RuntimeError("A leg residual changed upper-body targets")
    torques = controller.torques(data)
    if not np.all(np.isfinite(torques)):
        raise RuntimeError("A bounded residual produced non-finite torques")
    if np.any(np.abs(torques) > EFFORT_LIMIT_HARDWARE):
        raise RuntimeError("Residual torques exceeded actuator limits")


def check_bounded_rollout(policy_path: Path):
    model, data, controller = create_system(policy_path)
    command = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    residual = np.zeros(LEG_JOINT_COUNT, dtype=np.float32)
    residual[0] = MAX_LEG_JOINT_RESIDUAL
    controller.reset(data, command)

    minimum_height = np.inf
    minimum_upright = np.inf
    for _ in range(150):
        advance_control_step(
            model,
            data,
            controller,
            command,
            residual,
        )
        pelvis_rotation = data.xmat[controller.pelvis_id].reshape(3, 3)
        minimum_height = min(
            minimum_height,
            float(data.xpos[controller.pelvis_id, 2]),
        )
        minimum_upright = min(
            minimum_upright,
            float(pelvis_rotation[2, 2]),
        )

    if minimum_height < 0.45 or minimum_upright < 0.5:
        raise RuntimeError(
            "A maximum single-joint residual destabilized the test rollout"
        )


def check_residual_environment():
    env = G1KickResidualEnv(max_episode_steps=2)
    try:
        check_env(env, skip_render_check=True)
        observation_a, info_a = env.reset(seed=123)
        observation_b, info_b = env.reset(seed=123)
        if not np.array_equal(observation_a, observation_b):
            raise RuntimeError("Equal seeds produced different contact states")
        for name in (
            "warmup_duration",
            "contact_distance",
            "contact_lateral",
        ):
            if info_a[name] != info_b[name]:
                raise RuntimeError(f"Equal seeds changed {name}")

        initial_time = env.data.time
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(LEG_JOINT_COUNT, dtype=np.float32)
        )
        elapsed_time = env.data.time - initial_time
        if not np.isclose(elapsed_time, 0.02):
            raise RuntimeError(
                "One residual action advanced "
                f"{elapsed_time:.6f} s instead of 0.020000 s"
            )
        if not env.observation_space.contains(observation):
            raise RuntimeError("Residual step returned an invalid observation")
        if reward != 0.0 or terminated or truncated:
            raise RuntimeError("Neutral first residual step ended the episode")
        if info["goal"] or info["fell"]:
            raise RuntimeError("Neutral first step reported a terminal event")
    finally:
        env.close()


def main():
    policy_path = Path(DEFAULT_POLICY_PATH)
    check_zero_residual_equivalence(policy_path)
    check_target_composition(policy_path)
    check_bounded_rollout(policy_path)
    check_residual_environment()
    print("g1 low-level residual interface: passed")


if __name__ == "__main__":
    main()
