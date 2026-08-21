from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


PHYSICS_TIMESTEP = 0.002
CONTROL_TIMESTEP = 0.02
PHYSICS_STEPS_PER_CONTROL = round(CONTROL_TIMESTEP / PHYSICS_TIMESTEP)
HISTORY_LENGTH = 5
ACTION_SCALE = 0.25

HARDWARE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# For each policy-space joint, gives the corresponding Unitree hardware index.
POLICY_TO_HARDWARE = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int32,
)

DEFAULT_JOINT_POSITION_POLICY = np.array(
    [
        -0.1,
        -0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3,
        0.3,
        0.3,
        0.3,
        -0.2,
        -0.2,
        0.25,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.97,
        0.97,
        0.15,
        -0.15,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

STIFFNESS_HARDWARE = np.array(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        200,
        200,
        200,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
    ],
    dtype=np.float32,
)

DAMPING_HARDWARE = np.array(
    [
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        5,
        5,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    ],
    dtype=np.float32,
)

EFFORT_LIMIT_HARDWARE = np.array(
    [
        88,
        88,
        88,
        139,
        50,
        50,
        88,
        88,
        88,
        139,
        50,
        50,
        88,
        50,
        50,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
        25,
        25,
        25,
        25,
        25,
        5,
        5,
    ],
    dtype=np.float64,
)

COMMAND_LOW = np.array([-0.5, -0.3, -0.2], dtype=np.float32)
COMMAND_HIGH = np.array([1.0, 0.3, 0.2], dtype=np.float32)

OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


def policy_to_hardware(values: np.ndarray) -> np.ndarray:
    hardware_values = np.empty(29, dtype=values.dtype)
    hardware_values[POLICY_TO_HARDWARE] = values
    return hardware_values


DEFAULT_JOINT_POSITION_HARDWARE = policy_to_hardware(
    DEFAULT_JOINT_POSITION_POLICY
)


class G1LocomotionController:
    def __init__(self, model: mujoco.MjModel, policy_path: Path):
        self.model = model
        self._configure_model_for_locomotion()
        self._resolve_model_indices()

        self.session = ort.InferenceSession(str(policy_path))
        self._validate_policy_contract()

        self.last_action = np.zeros(29, dtype=np.float32)
        self.target_joint_position_hardware = (
            DEFAULT_JOINT_POSITION_HARDWARE.copy()
        )
        self._history: dict[str, deque[np.ndarray]] = {}

    def _configure_model_for_locomotion(self):
        if self.model.nu != len(HARDWARE_JOINT_NAMES):
            raise ValueError(
                f"Expected 29 actuators, found {self.model.nu}"
            )

        self.model.opt.timestep = PHYSICS_TIMESTEP
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

        # The Menagerie model uses position actuators. The pretrained policy's
        # deployment contract instead applies an external PD controller and
        # sends its resulting torques through direct motor actuators.
        self.model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_gainprm[:, 0] = 1.0
        self.model.actuator_biasprm[:] = 0.0
        self.model.actuator_ctrllimited[:] = 1
        self.model.actuator_ctrlrange[:, 0] = -EFFORT_LIMIT_HARDWARE
        self.model.actuator_ctrlrange[:, 1] = EFFORT_LIMIT_HARDWARE
        self.model.actuator_forcelimited[:] = 1
        self.model.actuator_forcerange[:, 0] = -EFFORT_LIMIT_HARDWARE
        self.model.actuator_forcerange[:, 1] = EFFORT_LIMIT_HARDWARE

    def _resolve_model_indices(self):
        qpos_addresses = []
        qvel_addresses = []
        actuator_joint_ids = []

        for hardware_index, joint_name in enumerate(HARDWARE_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"Missing G1 joint {joint_name!r}")

            qpos_addresses.append(self.model.jnt_qposadr[joint_id])
            qvel_addresses.append(self.model.jnt_dofadr[joint_id])

            actuator_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                hardware_index,
            )
            if actuator_name != joint_name:
                raise ValueError(
                    "Actuator order does not match the expected hardware "
                    f"order at index {hardware_index}: {actuator_name!r}"
                )
            actuator_joint_ids.append(
                self.model.actuator_trnid[hardware_index, 0]
            )

        expected_joint_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in HARDWARE_JOINT_NAMES
        ]
        if actuator_joint_ids != expected_joint_ids:
            raise ValueError("Actuators are connected to unexpected joints")

        self.joint_qpos_addresses = np.asarray(qpos_addresses)
        self.joint_qvel_addresses = np.asarray(qvel_addresses)
        self.pelvis_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )

        for hardware_index, joint_id in enumerate(expected_joint_ids):
            dof_address = self.model.jnt_dofadr[joint_id]
            self.model.dof_armature[dof_address] = 0.01
            self.model.dof_damping[dof_address] = 0.05
            self.model.dof_frictionloss[dof_address] = (
                0.1
                if "wrist" in HARDWARE_JOINT_NAMES[hardware_index]
                else 0.2
            )

    def _validate_policy_contract(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs":
            raise ValueError("Expected one ONNX input named 'obs'")
        if inputs[0].shape != [1, 480]:
            raise ValueError(
                f"Expected observation shape [1, 480], got {inputs[0].shape}"
            )
        if len(outputs) != 1 or outputs[0].shape != [1, 29]:
            raise ValueError("Expected one ONNX output with shape [1, 29]")

    def reset(self, data: mujoco.MjData, command: np.ndarray):
        command = self._validated_command(command)
        self.last_action.fill(0.0)
        self.target_joint_position_hardware[:] = (
            DEFAULT_JOINT_POSITION_HARDWARE
        )

        terms = self._observation_terms(data, command)
        self._history = {}
        for name in OBSERVATION_TERMS:
            initial_value = terms[name]
            self._history[name] = deque(
                (initial_value.copy() for _ in range(HISTORY_LENGTH)),
                maxlen=HISTORY_LENGTH,
            )

    def policy_step(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
    ) -> np.ndarray:
        command = self._validated_command(command)
        terms = self._observation_terms(data, command)
        for name in OBSERVATION_TERMS:
            self._history[name].append(terms[name])

        observation = np.concatenate(
            [
                np.concatenate(tuple(self._history[name]))
                for name in OBSERVATION_TERMS
            ]
        ).astype(np.float32, copy=False)
        action = self.session.run(
            None,
            {"obs": observation[np.newaxis, :]},
        )[0][0].astype(np.float32, copy=False)

        self.last_action[:] = action
        target_policy = (
            DEFAULT_JOINT_POSITION_POLICY + ACTION_SCALE * action
        )
        self.target_joint_position_hardware[:] = policy_to_hardware(
            target_policy
        )
        return action.copy()

    def torques(self, data: mujoco.MjData) -> np.ndarray:
        joint_position = data.qpos[self.joint_qpos_addresses]
        joint_velocity = data.qvel[self.joint_qvel_addresses]
        torque = (
            STIFFNESS_HARDWARE
            * (self.target_joint_position_hardware - joint_position)
            - DAMPING_HARDWARE * joint_velocity
        )
        return np.clip(
            torque,
            -EFFORT_LIMIT_HARDWARE,
            EFFORT_LIMIT_HARDWARE,
        )

    def _observation_terms(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
    ) -> dict[str, np.ndarray]:
        rotation = data.xmat[self.pelvis_id].reshape(3, 3)
        projected_gravity = rotation.T @ np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )

        angular_velocity = data.sensor(
            "imu-pelvis-angular-velocity"
        ).data
        hardware_joint_position = data.qpos[self.joint_qpos_addresses]
        hardware_joint_velocity = data.qvel[self.joint_qvel_addresses]
        policy_joint_position = hardware_joint_position[POLICY_TO_HARDWARE]
        policy_joint_velocity = hardware_joint_velocity[POLICY_TO_HARDWARE]

        return {
            "base_ang_vel": np.asarray(
                angular_velocity * 0.2,
                dtype=np.float32,
            ),
            "projected_gravity": np.asarray(
                projected_gravity,
                dtype=np.float32,
            ),
            "velocity_commands": command.astype(np.float32, copy=True),
            "joint_pos_rel": np.asarray(
                policy_joint_position - DEFAULT_JOINT_POSITION_POLICY,
                dtype=np.float32,
            ),
            "joint_vel_rel": np.asarray(
                policy_joint_velocity * 0.05,
                dtype=np.float32,
            ),
            "last_action": self.last_action.copy(),
        }

    @staticmethod
    def _validated_command(command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,):
            raise ValueError("Velocity command must have shape (3,)")
        if np.any(command < COMMAND_LOW) or np.any(command > COMMAND_HIGH):
            raise ValueError(
                f"Velocity command {command} is outside the training range "
                f"[{COMMAND_LOW}, {COMMAND_HIGH}]"
            )
        return command


def reset_g1_for_locomotion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: G1LocomotionController,
):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = (0.0, 0.0, 0.793)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[controller.joint_qpos_addresses] = (
        DEFAULT_JOINT_POSITION_HARDWARE
    )
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
