"""Verify the analytic batch FK + Jacobian against pinocchio (via RobotData).

Runs N random (base pose, joint config) samples through both and asserts
foot positions and foot Jacobian columns match within tolerance.

  conda run -n isaacgym python scripts/test_kinematics_a1_batch.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'linear_mpc'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np

from kinematics import quat2matrix
from kinematics_a1_batch import (
    fk_jacobian_batch, fk_quantities_batch, base_pos_thighs_batch,
    SIDE_SIGN, HIP_OFFSET, L_ABAD, L_THIGH, L_CALF,
)
from robot_data import RobotData


def random_unit_quat(rng):
    q = rng.standard_normal(4).astype(np.float32)
    q /= np.linalg.norm(q)
    return q


def main():
    urdf = os.path.join(os.path.dirname(__file__), '..', 'robot', 'a1', 'urdf', 'a1.urdf')
    rd = RobotData(robot_model=urdf, state_estimation=False)

    rng = np.random.default_rng(42)
    N = 20
    max_pos_err  = 0.0
    max_J_err    = 0.0
    max_thigh_err= 0.0
    max_velf_err = 0.0
    max_pos_world_err = 0.0

    for trial in range(N):
        # Random base pose + joint config (within reasonable bounds)
        pos_base  = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        quat      = random_unit_quat(rng)                          # [w, x, y, z]
        lin_vel   = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        ang_vel   = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        q_joints  = np.concatenate([
            np.array([
                rng.uniform(-0.5, 0.5),   # abad
                rng.uniform(-1.0, 1.0),   # hip
                rng.uniform(-2.5, -0.5),  # knee
            ], dtype=np.float32) for _ in range(4)
        ])
        qdot      = rng.uniform(-2.0, 2.0, 12).astype(np.float32)

        # ── pinocchio reference ──────────────────────────────────────────
        rd.update(pos_base=pos_base, quat_base=quat,
                   lin_vel_base=lin_vel, ang_vel_base=ang_vel,
                   q=q_joints, qdot=qdot)

        ref_pos_base_feet_world  = np.stack(rd.pos_base_feet,        axis=0)  # (4,3)
        ref_pos_base_feet_base   = np.stack(rd.base_pos_base_feet,   axis=0)  # (4,3)
        ref_pos_thighs_base      = np.stack(rd.base_pos_base_thighs, axis=0)  # (4,3)
        ref_base_vel_base_feet   = np.stack(rd.base_vel_base_feet,   axis=0)  # (4,3)
        ref_Jv_full              = [np.asarray(J, dtype=np.float32) for J in rd.Jv_feet]  # 4× (3,18)
        R_base                   = rd.R_base.astype(np.float32)

        # ── batch analytic ───────────────────────────────────────────────
        out = fk_quantities_batch(
            q_b               = q_joints[None, :],
            qdot_b            = qdot[None, :],
            R_base_b          = R_base[None, :, :],
            pos_base_b        = pos_base[None, :],
            lin_vel_base_in_b = lin_vel[None, :],
            ang_vel_base_in_b = ang_vel[None, :],
        )
        pos_base_feet_base   = out['pos_base_feet_base'][0]               # (4,3)
        pos_base_feet_world  = out['pos_base_feet_world'][0]              # (4,3)
        J_leg_base           = out['J_leg_base'][0]                       # (4,3,3)
        base_vel_base_feet   = out['base_vel_base_feet'][0]               # (4,3)
        pos_thighs_base      = out['pos_thighs_base'][0]                  # (4,3)

        # ── compare ──────────────────────────────────────────────────────
        # 1. foot position in base axes
        err = np.abs(pos_base_feet_base - ref_pos_base_feet_base).max()
        max_pos_err = max(max_pos_err, err)

        # 2. foot position in world axes (R_base @ base)
        err = np.abs(pos_base_feet_world - ref_pos_base_feet_world).max()
        max_pos_world_err = max(max_pos_world_err, err)

        # 3. thigh position in base axes
        err = np.abs(pos_thighs_base - ref_pos_thighs_base).max()
        max_thigh_err = max(max_thigh_err, err)

        # 4. foot velocity (rel-to-base in base axes)
        err = np.abs(base_vel_base_feet - ref_base_vel_base_feet).max()
        max_velf_err = max(max_velf_err, err)

        # 5. Jacobian — pinocchio gives (3, 18) in WORLD (LOCAL_WORLD_ALIGNED)
        # Our J_leg_base is (3, 3) in BASE axes. Conversion:
        #     J_world_full[:, 6+3i:6+3(i+1)] for leg i ≈ R_base @ J_leg_base[i]
        for leg in range(4):
            ref_cols   = ref_Jv_full[leg][:, 6 + 3 * leg : 6 + 3 * (leg + 1)]   # (3,3)
            ours_world = R_base @ J_leg_base[leg]                                # (3,3)
            err = np.abs(ours_world - ref_cols).max()
            max_J_err = max(max_J_err, err)

    print(f"\nVerified {N} random configs:")
    print(f"  max |pos_base_feet (base axes)  - pinocchio|  = {max_pos_err:.3e}")
    print(f"  max |pos_base_feet (world axes) - pinocchio|  = {max_pos_world_err:.3e}")
    print(f"  max |pos_thighs    (base axes)  - pinocchio|  = {max_thigh_err:.3e}")
    print(f"  max |base_vel_base_feet         - pinocchio|  = {max_velf_err:.3e}")
    print(f"  max |J_leg (rotated to world)   - pinocchio|  = {max_J_err:.3e}")
    tol = 1e-4
    bad = max(max_pos_err, max_pos_world_err, max_thigh_err,
              max_velf_err, max_J_err) > tol
    if bad:
        print(f"\n!!! FAIL: error exceeds tol={tol}")
        sys.exit(1)
    print(f"\nAll within {tol}.  Batch kinematics matches pinocchio.")

    # ── stress test: stack B=64 different configs into one batch call,
    # verify batched output matches per-row pinocchio ────────────────────
    print("\nStress test (B=64 batched call):")
    B = 64
    q_batch    = rng.uniform(-1.0, 1.0, (B, 12)).astype(np.float32)
    qdot_batch = rng.uniform(-2.0, 2.0, (B, 12)).astype(np.float32)
    quat_batch = rng.standard_normal((B, 4)).astype(np.float32)
    quat_batch /= np.linalg.norm(quat_batch, axis=1, keepdims=True)
    pos_b   = rng.uniform(-1.0, 1.0, (B, 3)).astype(np.float32)
    lv_b    = rng.uniform(-1.0, 1.0, (B, 3)).astype(np.float32)
    av_b    = rng.uniform(-1.0, 1.0, (B, 3)).astype(np.float32)
    R_batch = np.stack([quat2matrix(quat_batch[i].tolist()) for i in range(B)], axis=0).astype(np.float32)

    out_b = fk_quantities_batch(q_b=q_batch, qdot_b=qdot_batch,
                                 R_base_b=R_batch, pos_base_b=pos_b,
                                 lin_vel_base_in_b=lv_b, ang_vel_base_in_b=av_b)

    bad = 0
    for i in range(B):
        rd.update(pos_base=pos_b[i], quat_base=quat_batch[i],
                   lin_vel_base=lv_b[i], ang_vel_base=av_b[i],
                   q=q_batch[i], qdot=qdot_batch[i])
        ref = np.stack(rd.base_vel_base_feet, axis=0)
        if np.abs(out_b['base_vel_base_feet'][i] - ref).max() > 1e-4:
            bad += 1
    print(f"  B=64 batched call: {B - bad}/{B} rows match pinocchio at tol 1e-4.")
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
