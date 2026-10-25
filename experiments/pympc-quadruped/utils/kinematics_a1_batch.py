"""Batched analytic forward kinematics + foot Jacobian for A1.

A1's leg is a 3-DOF chain (abad → hip-pitch → knee-pitch) that can be solved
in closed form per-leg per-robot, then evaluated for all (B, 4) legs in one
NumPy/Torch vectorized expression.  This replaces the per-robot pinocchio
calls in `RobotData.update`, eliminating the dominant Python-loop overhead
when running thousands of robots in parallel in Isaac Gym.

URDF leg parameters (a1.urdf, lines ~89–560):

                               hip_offset_x   hip_offset_y    thigh_offset_y    L_thigh    L_calf
    FL  (sideSign +1)            +0.183           +0.047          +0.08505         0.2       0.2
    FR  (sideSign -1)            +0.183           -0.047          -0.08505         0.2       0.2
    RL  (sideSign +1)            -0.183           +0.047          +0.08505         0.2       0.2
    RR  (sideSign -1)            -0.183           -0.047          -0.08505         0.2       0.2

Joint axes:
    abad   (q0):  rotation around +x  (Rx(q0))
    hip    (q1):  rotation around +y  (Ry(q1))
    knee   (q2):  rotation around +y  (Ry(q2))

Frame chain (positions are origins of each frame in its parent's frame):
    base ── [hip_x, hip_y_off1, 0] ── abad joint ── [Rx(q0)] ──>
    abad ── [0, thigh_y_off, 0]    ── thigh joint ── [Ry(q1)] ──>
    thigh ── [0, 0, -L_thigh]      ── calf joint  ── [Ry(q2)] ──>
    calf  ── [0, 0, -L_calf]       ── foot

Foot position relative to the **hip joint origin**, expressed in **base axes**
(matches rl-mpc-locomotion convention; differs by a constant hip offset from
foot-relative-to-base-origin):

    p_hip_x = -L_thigh sin(q1) - L_calf sin(q1+q2)
    p_hip_y =  s · L_abad cos(q0) + r · sin(q0)
    p_hip_z =  s · L_abad sin(q0) - r · cos(q0)

where  s = sideSign  and  r = L_thigh cos(q1) + L_calf cos(q1+q2).

The Jacobian dp_hip / d[q0,q1,q2] is the analytical derivative (verified
against pinocchio in test_kinematics_a1_batch.py).
"""
from __future__ import annotations
import numpy as np


# ── A1-specific constants ───────────────────────────────────────────────────
# leg index convention (matches MPC + URDF): 0=FL, 1=FR, 2=RL, 3=RR
SIDE_SIGN  = np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float32)   # (4,)
HIP_OFFSET = np.array([                                              # (4, 3)
    [+0.183, +0.047, 0.0],   # FL
    [+0.183, -0.047, 0.0],   # FR
    [-0.183, +0.047, 0.0],   # RL
    [-0.183, -0.047, 0.0],   # RR
], dtype=np.float32)
L_ABAD  = 0.08505
L_THIGH = 0.20
L_CALF  = 0.20

# Thigh-joint origin in **base-axes**, parameterized by q_abad.  Derivation:
#   thigh_origin_base = hip_offset + Rx(q_abad) @ [0, sideSign·L_abad, 0]
#                     = hip_offset + [0,  s·L_abad·cos(q0),  s·L_abad·sin(q0)]


def fk_jacobian_batch(q_b: np.ndarray):
    """Analytic batch FK + foot Jacobian for A1.

    Args:
        q_b : (B, 12) joint angles, ordered
                  [FL_q0, FL_q1, FL_q2, FR_..., RL_..., RR_...]
              q0 = abad, q1 = hip-pitch, q2 = knee-pitch.

    Returns:
        pos_base_feet_base_axes : (B, 4, 3)
            Foot position relative to base origin, in base axes.
        J_leg_base              : (B, 4, 3, 3)
            Foot velocity (in base axes) ∂p/∂q for the leg's own 3 joints.
    """
    B = q_b.shape[0]
    q = q_b.reshape(B, 4, 3).astype(np.float32)
    q0, q1, q2 = q[..., 0], q[..., 1], q[..., 2]
    c0, s0 = np.cos(q0), np.sin(q0)
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)
    c12 = c1 * c2 - s1 * s2     # cos(q1 + q2)
    s12 = s1 * c2 + c1 * s2     # sin(q1 + q2)
    sgn = SIDE_SIGN                                                  # (4,)
    sgn_b = sgn[None, :]                                             # (1, 4) for broadcast

    Lt, Lc, La = L_THIGH, L_CALF, L_ABAD
    r = Lt * c1 + Lc * c12       # "vertical reach" in abad frame, (B, 4)

    # ── Position relative to hip joint, in base axes ────────────────────────
    p_hip_x = -Lt * s1 - Lc * s12
    p_hip_y =  sgn_b * La * c0 + r * s0
    p_hip_z =  sgn_b * La * s0 - r * c0

    pos_base_feet = np.stack([p_hip_x, p_hip_y, p_hip_z], axis=-1)   # (B, 4, 3)
    pos_base_feet = pos_base_feet + HIP_OFFSET[None, :, :]            # add base→hip offset

    # ── Jacobian (3×3) per (robot, leg), in base axes ──────────────────────
    # Rows: world axes (x, y, z); cols: q0=abad, q1=hip, q2=knee.
    J = np.empty((B, 4, 3, 3), dtype=np.float32)

    # ∂p_x: independent of q0
    J[..., 0, 0] = 0.0
    J[..., 0, 1] = -Lt * c1 - Lc * c12
    J[..., 0, 2] = -Lc * c12

    # ∂p_y / ∂q0 = -sgn·La·s0 + r·c0
    # ∂p_y / ∂q1 = (∂r/∂q1) · s0 = (-Lt·s1 - Lc·s12) · s0
    # ∂p_y / ∂q2 = (-Lc·s12) · s0
    J[..., 1, 0] = -sgn_b * La * s0 + r * c0
    J[..., 1, 1] = (-Lt * s1 - Lc * s12) * s0
    J[..., 1, 2] = -Lc * s12 * s0

    # ∂p_z / ∂q0 =  sgn·La·c0 + r·s0
    # ∂p_z / ∂q1 = -(∂r/∂q1) · c0 = (Lt·s1 + Lc·s12) · c0
    # ∂p_z / ∂q2 =  Lc·s12 · c0
    J[..., 2, 0] = sgn_b * La * c0 + r * s0
    J[..., 2, 1] = (Lt * s1 + Lc * s12) * c0
    J[..., 2, 2] = Lc * s12 * c0

    return pos_base_feet, J


def base_pos_thighs_batch(q_abad_b: np.ndarray) -> np.ndarray:
    """Thigh-joint origin in base axes, given (B, 4) abad angles.

    thigh_origin_base = hip_offset + Rx(q_abad) @ [0, sgn·L_abad, 0]
                      = hip_offset + [0, sgn·L_abad·cos(q0), sgn·L_abad·sin(q0)]
    """
    c0 = np.cos(q_abad_b)                                            # (B, 4)
    s0 = np.sin(q_abad_b)
    sgn = SIDE_SIGN[None, :]                                          # (1, 4)
    out = np.zeros((q_abad_b.shape[0], 4, 3), dtype=np.float32)
    out[..., 1] = sgn * L_ABAD * c0
    out[..., 2] = sgn * L_ABAD * s0
    out += HIP_OFFSET[None, :, :]
    return out


def fk_quantities_batch(q_b: np.ndarray,
                         qdot_b: np.ndarray,
                         R_base_b: np.ndarray,
                         pos_base_b: np.ndarray,
                         lin_vel_base_in_b: np.ndarray,
                         ang_vel_base_in_b: np.ndarray):
    """Compute everything the deployment loop needs from joint state + base pose.

    Args:
        q_b                : (B, 12)   joint positions
        qdot_b             : (B, 12)   joint velocities
        R_base_b           : (B, 3, 3) base rotation (world ← base)
        pos_base_b         : (B, 3)    base position in world
        lin_vel_base_in_b  : (B, 3)    raw lin_vel_base passed to RobotData
        ang_vel_base_in_b  : (B, 3)    raw ang_vel_base passed to RobotData

    Note on velocity convention (must match RobotData/pinocchio drop-in):
        pinocchio's FreeFlyer joint expects body-frame floating-base velocity,
        but RobotData feeds Isaac Gym's world-frame velocities directly. With
        that misuse, pinocchio's `base_vel_base_feet` reduces to
            (v_in − R_base.T @ v_in)  +  ω_in × p_base  +  J_leg_base @ qdot_leg
        We replicate that exactly so this function is a behavior-preserving
        replacement for the per-robot pinocchio call.  (For R_base ≈ I the
        first term vanishes and the formula equals the physically correct
        ω_body × p_base + J·qdot.)

    Returns dict with keys:
        pos_base_feet_base    : (B, 4, 3)  foot in base axes, base-origin
        pos_base_feet_world   : (B, 4, 3)  foot in world axes, base-origin
                                 (= R_base @ pos_base_feet_base ; matches
                                 robot_data.pos_base_feet semantics)
        pos_feet_world        : (B, 4, 3)  foot in world frame
        J_leg_base            : (B, 4, 3, 3)  foot Jacobian in base axes
        base_vel_base_feet    : (B, 4, 3)  matches RobotData.base_vel_base_feet
        pos_thighs_base       : (B, 4, 3)  thigh joint origin in base axes
                                 (matches robot_data.base_pos_base_thighs)
    """
    pos_base_feet_base, J_leg_base = fk_jacobian_batch(q_b)              # (B,4,3), (B,4,3,3)
    qdot = qdot_b.reshape(qdot_b.shape[0], 4, 3).astype(np.float32)      # (B, 4, 3)

    v_from_joints = np.einsum('blij,blj->bli', J_leg_base, qdot)          # (B, 4, 3)

    # ω_in × p_base, broadcast over 4 legs.
    v_from_omega = np.cross(ang_vel_base_in_b[:, None, :], pos_base_feet_base)  # (B, 4, 3)

    # extra = v_in − R_base.T @ v_in   (vanishes when R_base = I)
    Rt_v = np.einsum('bji,bj->bi', R_base_b, lin_vel_base_in_b)          # R_base.T @ v
    extra = (lin_vel_base_in_b - Rt_v)[:, None, :]                        # (B, 1, 3)

    base_vel_base_feet = v_from_joints + v_from_omega + extra

    # World-axes foot pos (base origin) = R_base @ p_base
    pos_base_feet_world = np.einsum('bij,blj->bli', R_base_b, pos_base_feet_base)
    pos_feet_world      = pos_base_feet_world + pos_base_b[:, None, :]

    pos_thighs_base = base_pos_thighs_batch(
        q_b.reshape(q_b.shape[0], 4, 3)[..., 0])                        # (B, 4, 3)

    return dict(
        pos_base_feet_base = pos_base_feet_base,
        pos_base_feet_world= pos_base_feet_world,
        pos_feet_world     = pos_feet_world,
        J_leg_base         = J_leg_base,
        base_vel_base_feet = base_vel_base_feet,
        pos_thighs_base    = pos_thighs_base,
    )
