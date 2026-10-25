"""Parity check: batch leg controller / swing trajectory == per-robot legacy path.

Drives both pipelines for a fixed sequence of (base pose, joint state, contact
forces, gait phase), then asserts they produce identical torques.

  conda run -n isaacgym python scripts/test_batch_controller_parity.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'linear_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np

from kinematics import quat2matrix
from robot_data import RobotData
from leg_controller import LegController
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator
from gait import Gait
from linear_mpc_configs import LinearMpcConfig
from robot_configs import A1Config
from batch_kinematics_controller import BatchKinematicsController


def main():
    urdf = os.path.join(os.path.dirname(__file__), '..', 'robot', 'a1', 'urdf', 'a1.urdf')
    rng = np.random.default_rng(123)

    B = 8
    num_steps = 30
    base_vel_des = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    yaw_rate_des = 0.0

    # ── per-robot legacy stack ──
    rd_list   = [RobotData(robot_model=urdf) for _ in range(B)]
    leg_ctrls = [LegController(A1Config.Kp_swing, A1Config.Kd_swing) for _ in range(B)]
    swing_trajs = [[SwingFootTrajectoryGenerator(leg) for leg in range(4)] for _ in range(B)]
    gaits = [Gait.TROTTING10 for _ in range(B)]   # all share state (Enum)

    # ── batch stack ──
    batch = BatchKinematicsController(num_robots=B, robot_config=A1Config,
                                       dt_control=LinearMpcConfig.dt_control,
                                       gravity=LinearMpcConfig.gravity)
    gait_shared = Gait.TROTTING10

    # Random per-robot states (held fixed across steps for simplicity).
    pos_b = rng.uniform(-0.2, 0.2, (B, 3)).astype(np.float32);  pos_b[:, 2] += 0.4
    quat_b = rng.standard_normal((B, 4)).astype(np.float32)
    quat_b /= np.linalg.norm(quat_b, axis=1, keepdims=True)
    R_b = np.stack([quat2matrix(quat_b[i].tolist()) for i in range(B)], axis=0).astype(np.float32)
    lin_b = rng.uniform(-0.3, 0.3, (B, 3)).astype(np.float32)
    ang_b = rng.uniform(-0.5, 0.5, (B, 3)).astype(np.float32)
    q_b   = np.stack([
        np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.5, 1.0), rng.uniform(-2.0, -1.0)] * 4,
                  dtype=np.float32) for _ in range(B)
    ], axis=0)
    qdot_b = rng.uniform(-1.0, 1.0, (B, 12)).astype(np.float32)

    iters_per_mpc = LinearMpcConfig.iteration_between_mpc

    max_diff = 0.0
    for step in range(num_steps):
        # advance gait clock
        for g in gaits:
            g.set_iteration(iters_per_mpc, step)
        gait_shared.set_iteration(iters_per_mpc, step)

        # synthetic contact forces (varies per step)
        F_b = (50.0 * rng.standard_normal((B, 12)).astype(np.float32)
               + np.tile([0, 0, 30.0], 4))

        # ── batch path ──
        batch.update_kinematics(q_b=q_b, qdot_b=qdot_b, pos_base_b=pos_b,
                                  R_base_b=R_b, lin_vel_base_b=lin_b,
                                  ang_vel_base_b=ang_b)
        batch_pos_des, batch_vel_des, swing_states_shared = batch.update_swing_targets(
            gait_shared, base_vel_des, yaw_rate_des)
        torques_batch = batch.compute_torques(
            contact_forces_world  = F_b,
            swing_states          = swing_states_shared,
            base_pos_swingfoot_des= batch_pos_des,
            base_vel_swingfoot_des= batch_vel_des,
        )                                                               # (B, 12)

        # ── legacy path ──
        torques_legacy = np.zeros((B, 12), dtype=np.float32)
        for i in range(B):
            rd_list[i].update(pos_base=pos_b[i], quat_base=quat_b[i],
                               lin_vel_base=lin_b[i], ang_vel_base=ang_b[i],
                               q=q_b[i], qdot=qdot_b[i])
            swing_states = gaits[i].get_swing_state()
            pos_targets = np.zeros((4, 3));  vel_targets = np.zeros((4, 3))
            for leg in range(4):
                if swing_states[leg] > 0:
                    swing_trajs[i][leg].set_foot_placement(
                        rd_list[i], gaits[i], base_vel_des, yaw_rate_des)
                    p_d, v_d = swing_trajs[i][leg].compute_traj_swingfoot(
                        rd_list[i], gaits[i])
                    pos_targets[leg, :] = p_d
                    vel_targets[leg, :] = v_d
            torques_legacy[i] = leg_ctrls[i].update(
                rd_list[i], F_b[i], swing_states, pos_targets, vel_targets)

        diff = np.abs(torques_batch - torques_legacy).max()
        max_diff = max(max_diff, diff)

    print(f"Max torque diff over {num_steps} steps × B={B}: {max_diff:.3e}")
    if max_diff > 1e-3:
        print("FAIL")
        sys.exit(1)
    print("OK — batch controller torques match legacy pipeline within 1e-3.")


if __name__ == '__main__':
    main()
