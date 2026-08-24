import numpy as np
from gymnasium.utils.env_checker import check_env

from src.soccer_3d import G1HighLevelKickResidualEnv
from src.soccer_3d.g1_high_level_kick_residual_env import (
    MAX_HIGH_LEVEL_COMMAND_RESIDUAL,
    apply_high_level_command_residual,
)
from src.soccer_3d.g1_locomotion import COMMAND_HIGH, COMMAND_LOW


def rollout(env, seed, action, steps=10):
    observation, _ = env.reset(seed=seed)
    trajectory = [observation.copy()]
    infos = []
    for _ in range(steps):
        observation, _, terminated, truncated, info = env.step(action)
        trajectory.append(observation.copy())
        infos.append(info)
        if terminated or truncated:
            break
    return np.stack(trajectory), infos


def main():
    env = G1HighLevelKickResidualEnv()
    reference_env = G1HighLevelKickResidualEnv()
    try:
        check_env(env, skip_render_check=True)

        zero = np.zeros(3, dtype=np.float32)
        trajectory, infos = rollout(env, seed=17, action=zero)
        reference_trajectory, reference_infos = rollout(
            reference_env,
            seed=17,
            action=zero,
        )
        np.testing.assert_allclose(
            trajectory,
            reference_trajectory,
            rtol=0.0,
            atol=0.0,
        )
        for info, reference_info in zip(infos, reference_infos):
            np.testing.assert_allclose(
                info["command"],
                info["base_command"],
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                info["command"],
                reference_info["command"],
                rtol=0.0,
                atol=0.0,
            )

        final_command, residual = apply_high_level_command_residual(
            np.array([0.95, 0.28, 0.18], dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
        np.testing.assert_allclose(
            residual,
            MAX_HIGH_LEVEL_COMMAND_RESIDUAL,
        )
        np.testing.assert_allclose(final_command, COMMAND_HIGH)

        final_command, _ = apply_high_level_command_residual(
            np.array([-0.45, -0.28, -0.18], dtype=np.float32),
            -np.ones(3, dtype=np.float32),
        )
        np.testing.assert_allclose(final_command, COMMAND_LOW)

        before_time = env.data.time
        env.reset(seed=23)
        before_time = env.data.time
        env.step(zero)
        np.testing.assert_allclose(
            env.data.time - before_time,
            env.control_timestep,
            rtol=0.0,
            atol=1e-12,
        )
    finally:
        env.close()
        reference_env.close()

    print("G1 high-level residual interface checks passed")


if __name__ == "__main__":
    main()
