"""Batched A1 kinematics + leg controller + swing foot trajectory.

Drop-in replacement for the per-robot pinocchio + LegController +
SwingFootTrajectoryGenerator loop in isaacgym_a1.py.  Operates on (B, ·)
arrays so the entire phase-3 path runs as numpy ops without per-robot
Python iteration.

Behavior is preserved bit-exact (within single-precision noise) against the
existing per-robot path:
    • FK / Jacobian — see test_kinematics_a1_batch.py (max diff ~1e-7)
    • Leg controller stance/swing torques — same formulas, just batched
    • Swing trajectory — analytic cubic Hermite (zero endpoint velocities)
      replicating pydrake.PiecewisePolynomial.CubicHermite

Assumes (and the script enforces) that all NUM_ROBOTS robots share a single
Gait instance, so swing_states / total_swing_time are scalar per leg.
"""
from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from kinematics_a1_batch import fk_quantities_batch
from gait import Gait


# ── Cubic Hermite (zero endpoint velocities) ───────────────────────────────

def _cubic_hermite_zero_vel(t, tA, tB, pA, pB):
    """Two-keypoint cubic with vA = vB = 0.  Vectorized over leading dims.

    Args (broadcastable):
        t, tA, tB : scalar or array
        pA, pB    : (..., 3) endpoint positions
    Returns (pos, vel), both (..., 3).
    """
    h = tB - tA
    h_safe = np.where(np.abs(h) < 1e-9, 1e-9, h)
    s = (t - tA) / h_safe
    s = np.clip(s, 0.0, 1.0)
    h00 = 1.0 - 3.0 * s * s + 2.0 * s ** 3
    h01 = 3.0 * s * s - 2.0 * s ** 3
    pos = h00[..., None] * pA + h01[..., None] * pB
    # dpos/dt = (1/h) * (6s² − 6s) * (pB − pA)
    coef = (6.0 * s * s - 6.0 * s) / h_safe
    vel = coef[..., None] * (pA - pB)   # = (-coef)*(pB-pA), but matches sign convention
    return pos, vel


def _swing_traj_eval(cur_t, total_T, pos_init, pos_mid, pos_final):
    """Two-segment cubic-Hermite (init → mid at T/2 → final at T), zero vel at every break.

    Args:
        cur_t   : scalar or (B,)        current time within the swing
        total_T : scalar                 total swing time
        pos_init, pos_mid, pos_final : (..., 3) per-(robot, leg) keypoints

    Returns (pos, vel), each (..., 3).
    """
    half = 0.5 * total_T
    in_first = cur_t <= half
    pA = np.where(in_first[..., None], pos_init, pos_mid)
    pB = np.where(in_first[..., None], pos_mid,  pos_final)
    tA = np.where(in_first, 0.0,  half)
    tB = np.where(in_first, half, total_T)
    pos, vel = _cubic_hermite_zero_vel(cur_t, tA, tB, pA, pB)
    return pos, vel


# ── Batched controller ─────────────────────────────────────────────────────

class BatchKinematicsController:
    """Batches FK + Jacobian + leg controller + swing-foot trajectory across B robots.

    Public surface mirrors what isaacgym_a1.py needed from the per-robot path:
        - update(joint_q, joint_qd, pos_base, R_base, lin_vel, ang_vel, gait, ...)
        - compute_torques(...) → (B, 12) torque array
    """

    def __init__(self,
                 num_robots: int,
                 robot_config,
                 dt_control: float,
                 gravity: float):
        self.B = int(num_robots)
        self.dt_control = float(dt_control)
        self.gravity = float(gravity)
        self.swing_height = float(robot_config.swing_height)
        # Kp / Kd are diagonal; keep as (3,3) for general matmul consistency.
        self.Kp_swing = np.asarray(robot_config.Kp_swing, dtype=np.float32)
        self.Kd_swing = np.asarray(robot_config.Kd_swing, dtype=np.float32)

        # Per-leg shared state (driven by single shared Gait instance):
        self.is_first_swing      = np.ones(4, dtype=bool)
        self.remaining_swing_time = np.zeros(4, dtype=np.float32)
        # Per-robot-per-leg foot endpoints (world frame):
        self.footpos_init  = np.zeros((self.B, 4, 3), dtype=np.float32)
        self.footpos_final = np.zeros((self.B, 4, 3), dtype=np.float32)

        # Cached batched kinematics (refreshed each step in update_kinematics)
        self._kin = None
        self._lin_vel_base = None
        self._ang_vel_base = None
        self._pos_base     = None
        self._R_base       = None

    # ──────────────────────────────────────────────────────────────────────

    def update_kinematics(self,
                           q_b: np.ndarray,
                           qdot_b: np.ndarray,
                           pos_base_b: np.ndarray,
                           R_base_b: np.ndarray,
                           lin_vel_base_b: np.ndarray,
                           ang_vel_base_b: np.ndarray) -> dict:
        """Run the batched analytic FK and stash results.  Returns the dict."""
        kin = fk_quantities_batch(
            q_b               = q_b,
            qdot_b            = qdot_b,
            R_base_b          = R_base_b,
            pos_base_b        = pos_base_b,
            lin_vel_base_in_b = lin_vel_base_b,
            ang_vel_base_in_b = ang_vel_base_b,
        )
        self._kin = kin
        self._lin_vel_base = lin_vel_base_b
        self._ang_vel_base = ang_vel_base_b
        self._pos_base     = pos_base_b
        self._R_base       = R_base_b
        return kin

    # ──────────────────────────────────────────────────────────────────────

    def update_swing_targets(self,
                              gait: Gait,
                              base_vel_base_des: np.ndarray,
                              yaw_turn_rate_des) -> tuple:
        """Update per-leg swing endpoints and return (pos_des, vel_des) in BASE axes.

        Args:
            gait              : the shared Gait instance
            base_vel_base_des : (3,) shared, (1,3), or (B,3) per-robot
            yaw_turn_rate_des : scalar, (1,), (B,), or (B,1) per-robot

        Returns:
            base_pos_swingfoot_des : (B, 4, 3)  swing foot pos target in base axes
            base_vel_swingfoot_des : (B, 4, 3)  swing foot vel target in base axes
            swing_states           : (4,)  shared swing schedule for this step
        """
        B = self.B
        swing_states = np.asarray(gait.get_swing_state(), dtype=np.float32)   # (4,)
        total_swing_time = float(gait.swing_time)
        total_stance_time = float(gait.stance_time)

        # Normalize cmd shapes once, then everything below is per-robot.
        from tqpnet_mpc import _normalize_cmd_shapes
        base_vel_base_des, yaw_turn_rate_des = _normalize_cmd_shapes(
            base_vel_base_des, yaw_turn_rate_des, B)                       # (B,3), (B,)

        v_des_world = np.einsum('bij,bj->bi', self._R_base,
                                 base_vel_base_des).astype(np.float32)     # (B, 3)

        # foot placement per leg (Raibert + capture point) — identical formula
        # to SwingFootTrajectoryGenerator.set_foot_placement, batched.
        kin = self._kin
        thighs_base = kin['pos_thighs_base']                            # (B, 4, 3)
        pos_feet_world = kin['pos_feet_world']                          # (B, 4, 3)

        # Per-robot RotZ at half-stance offset for yaw rate.  Each robot rotates
        # its own thigh-direction vector by its own yaw-rate × Δt.
        yaw_turn = yaw_turn_rate_des * (0.5 * total_stance_time)        # (B,)
        cy, sy = np.cos(yaw_turn), np.sin(yaw_turn)
        zero = np.zeros(B, dtype=np.float32)
        one  = np.ones(B,  dtype=np.float32)
        RotZ = np.stack([
            np.stack([cy, -sy, zero], axis=-1),
            np.stack([sy,  cy, zero], axis=-1),
            np.stack([zero, zero, one], axis=-1),
        ], axis=-2).astype(np.float32)                                   # (B, 3, 3)
        thighs_corrected = np.einsum('bij,blj->bli', RotZ, thighs_base)  # (B, 4, 3)

        # Only update per-leg state for legs currently swinging (matches the
        # script's `if swing_states[leg_idx] > 0:` gate around set_foot_placement
        # in the legacy path).  Stance legs keep their state untouched.
        is_swinging = (swing_states > 0.0)                               # (4,)

        for leg in range(4):
            if not is_swinging[leg]:
                continue
            if self.is_first_swing[leg]:
                self.remaining_swing_time[leg] = total_swing_time
            else:
                self.remaining_swing_time[leg] -= self.dt_control

        rs_t  = self.remaining_swing_time[None, :, None].astype(np.float32)   # (1, 4, 1)
        cmd_b = base_vel_base_des[:, None, :]                                  # (B, 1, 3)

        # final pos in WORLD: pos_base + R_base @ (thighs_corr + cmd*rs_t)
        #                          + 0.5*Tstance*v_base_world + 0.03*(v_world - v_des_world)
        local = thighs_corrected + cmd_b * rs_t                          # (B, 4, 3)
        rotated_local = np.einsum('bij,blj->bli', self._R_base, local)   # (B, 4, 3)
        v_base_world = self._lin_vel_base                                 # (B, 3)
        v_des_b      = v_des_world                                        # (B, 3)
        capture_term = (0.5 * total_stance_time) * v_base_world + 0.03 * (v_base_world - v_des_b)   # (B, 3)
        world_footpos_final = (self._pos_base[:, None, :] + rotated_local
                                + capture_term[:, None, :])                    # (B, 4, 3)

        # capture-point yaw correction (per-robot wz)
        wz = yaw_turn_rate_des                                             # (B,)
        z_over_g = (0.5 * self._pos_base[:, 2] / self.gravity)             # (B,)
        wz_term = (z_over_g * wz)[:, None]                                  # (B, 1)
        world_footpos_final[..., 0] += wz_term * v_base_world[:, 1:2]
        world_footpos_final[..., 1] += wz_term * (-v_base_world[:, 0:1])
        world_footpos_final[..., 2] = -0.0255

        # per-leg first-swing init capture + final pos write — gated on swing
        for leg in range(4):
            if not is_swinging[leg]:
                continue
            if self.is_first_swing[leg]:
                self.footpos_init[:, leg, :] = pos_feet_world[:, leg, :]
                self.is_first_swing[leg] = False
            self.footpos_final[:, leg, :] = world_footpos_final[:, leg, :]
            if swing_states[leg] >= 1.0:
                self.is_first_swing[leg] = True

        # mid-point: average of init/final but z forced to swing_height
        footpos_mid = 0.5 * (self.footpos_init + self.footpos_final)
        footpos_mid[..., 2] = self.swing_height

        # cur_swing_time per leg (broadcast across robots since gait is shared)
        cur_swing_t = np.broadcast_to(
            (total_swing_time - self.remaining_swing_time)[None, :],
            (B, 4)).astype(np.float32)                                   # (B, 4)

        # cubic Hermite traj evaluation
        pos_world, vel_world = _swing_traj_eval(
            cur_swing_t, total_swing_time,
            self.footpos_init, footpos_mid, self.footpos_final)          # both (B, 4, 3)

        # rotate world → base axes (matches compute_traj_swingfoot)
        delta_pos = pos_world - self._pos_base[:, None, :]
        delta_vel = vel_world - self._lin_vel_base[:, None, :]
        Rt = self._R_base.transpose(0, 2, 1)                             # base ← world
        base_pos_des = np.einsum('bij,blj->bli', Rt, delta_pos)
        base_vel_des = np.einsum('bij,blj->bli', Rt, delta_vel)

        return base_pos_des, base_vel_des, swing_states

    # ──────────────────────────────────────────────────────────────────────

    def compute_torques(self,
                         contact_forces_world: np.ndarray,
                         swing_states: np.ndarray,
                         base_pos_swingfoot_des: np.ndarray,
                         base_vel_swingfoot_des: np.ndarray) -> np.ndarray:
        """Batched leg controller — replaces LegController.update over all robots/legs.

        For stance leg (swing_states[leg] == 0):
            tau_3 = J_leg_baseᵀ · (R_baseᵀ · (-F_world))
        For swing leg (swing_states[leg] > 0):
            err_world = Kp · R_base · (p_des − p_now) + Kd · R_base · (v_des − v_now)
            tau_3 = J_leg_baseᵀ · (R_baseᵀ · err_world)
                  = J_leg_baseᵀ · (Kp_diag · (p_des − p_now) + Kd_diag · (v_des − v_now))
            (since Kp / Kd are diagonal isotropic, they commute with R_base.)

        Args:
            contact_forces_world   : (B, 12)   first-horizon GRF in world frame
            swing_states           : (4,)      per-leg shared swing schedule
            base_pos_swingfoot_des : (B, 4, 3) target swing foot pos (base axes)
            base_vel_swingfoot_des : (B, 4, 3) target swing foot vel (base axes)

        Returns:
            torques : (B, 12)  joint torques in [FL_q0..FL_q2, FR_..., RL_..., RR_...]
        """
        B = self.B
        kin = self._kin
        J_leg_base    = kin['J_leg_base']            # (B, 4, 3, 3)
        pos_now_base  = kin['pos_base_feet_base']    # (B, 4, 3)
        vel_now_base  = kin['base_vel_base_feet']    # (B, 4, 3)

        # Construct per-leg "wrench in BASE axes" w_b: (B, 4, 3).
        # Stance:  w_b = R_baseᵀ · (-F_world)
        # Swing :  w_b = Kp_diag·(p_des - p_now) + Kd_diag·(v_des - v_now)
        F_world = contact_forces_world.reshape(B, 4, 3)
        # R_baseᵀ broadcast
        Rt = self._R_base.transpose(0, 2, 1)          # (B, 3, 3)
        F_base = np.einsum('bij,blj->bli', Rt, F_world)            # (B, 4, 3)
        stance_w = -F_base

        # Diagonal Kp/Kd assumption (verified for A1Config: diag([200,200,200])):
        kp = np.diag(self.Kp_swing).astype(np.float32)             # (3,)
        kd = np.diag(self.Kd_swing).astype(np.float32)             # (3,)
        swing_w = (kp[None, None, :] * (base_pos_swingfoot_des - pos_now_base)
                    + kd[None, None, :] * (base_vel_swingfoot_des - vel_now_base))

        # Per-leg select stance vs swing.  swing_states[leg] > 0 → swing.
        is_swing = (swing_states > 0.0).astype(np.float32)[None, :, None]   # (1, 4, 1)
        wrench_b = is_swing * swing_w + (1.0 - is_swing) * stance_w          # (B, 4, 3)

        # tau_3 = J_leg_baseᵀ · wrench_b
        torques = np.einsum('blji,blj->bli', J_leg_base, wrench_b)            # (B, 4, 3)
        return torques.reshape(B, 12).astype(np.float32)
