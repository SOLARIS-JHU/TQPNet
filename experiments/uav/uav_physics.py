"""UAV quadrotor physics and MPC condensed-form utilities (parametric N_H).

Adapted from uav_lemniscate_osqp.py with N_H as a parameter.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

M_NOM    = 0.408
L_ARM    = 0.136
BETA_NOM = 0.2
IXX_NOM  = 0.0047
IYY_NOM  = 0.0047
IZZ_NOM  = 0.0089
G        = 9.81
K_THRUST = 9.81 * 22.5 / 1000
DT       = 1.0 / 14.0
N_S      = 13
N_U      = 4
U_HOV    = M_NOM * G / (4.0 * K_THRUST)

LEMN_A  = 1.0
LEMN_Z0 = 1.0
LEMN_W  = 0.03

DU_MIN  = -U_HOV
DU_MAX  = 11.1 - U_HOV


def lemn_point(t: float) -> np.ndarray:
    w = LEMN_W
    return np.array([LEMN_A * np.sin(w * t),
                     LEMN_A * np.sin(w * t) * np.cos(w * t),
                     LEMN_Z0])


def lemn_velocity(t: float) -> np.ndarray:
    w = LEMN_W
    return np.array([
        LEMN_A * w * np.cos(w * t) / DT,
        LEMN_A * w * (np.cos(w * t)**2 - np.sin(w * t)**2) / DT,
        0.0,
    ])


def make_ref_seq(t_current: float, N_H: int = 50) -> np.ndarray:
    refs = []
    for k in range(N_H + 1):
        tk = t_current + k
        s  = np.zeros(N_S)
        s[3:6]  = lemn_point(tk)
        s[9:12] = lemn_velocity(tk)
        refs.append(s)
    return np.array(refs)


def build_linear_model():
    """Discrete-time 13-state quadrotor model (delta-u formulation)."""
    Ac = np.zeros((12, 12))
    Ac[3, 9] = Ac[4, 10] = Ac[5, 11] = 1.0
    Ac[0, 6] = Ac[1,  7] = Ac[2,  8] = 1.0
    Ac[9,  0] = -G;            Ac[9,  9]  = -BETA_NOM / M_NOM
    Ac[10, 1] =  G;            Ac[10, 10] = -BETA_NOM / M_NOM
    Ac[11, 11] = -BETA_NOM / M_NOM
    Bc = np.zeros((12, 4))
    Bc[6, 1] =  K_THRUST * L_ARM / IXX_NOM
    Bc[6, 3] = -K_THRUST * L_ARM / IXX_NOM
    Bc[7, 2] =  K_THRUST * L_ARM / IYY_NOM
    Bc[7, 0] = -K_THRUST * L_ARM / IYY_NOM
    Bc[8, 0] = -K_THRUST * L_ARM / IZZ_NOM
    Bc[8, 1] =  K_THRUST * L_ARM / IZZ_NOM
    Bc[8, 2] = -K_THRUST * L_ARM / IZZ_NOM
    Bc[8, 3] =  K_THRUST * L_ARM / IZZ_NOM
    Bc[11, :] = K_THRUST / M_NOM
    Ac13 = np.zeros((13, 13)); Ac13[:12, :12] = Ac; Ac13[12, 5] = 1.0
    Bc13 = np.zeros((13, 4));  Bc13[:12, :]  = Bc
    return np.eye(13) + DT * Ac13, DT * Bc13


Q_S = np.diag([0., 0., 0., 10., 10., 10., 1., 1., 1., 10., 10., 10., 50.])
R_S = 0.1 * np.eye(N_U)
P_S = 5.0 * Q_S


def condense_mpc_matrices(A, B, x_ref_seq, N_H: int = 50):
    """Return (Q_qp, M_qp, r_qp, A_u, c_u, A_st, M_st, off)."""
    n, m_u = A.shape[0], B.shape[1]
    Abar = np.zeros((N_H * n, n))
    Bbar = np.zeros((N_H * n, N_H * m_u))
    Ap = np.eye(n)
    for i in range(N_H):
        Ap = A @ Ap
        Abar[i * n:(i + 1) * n, :] = Ap
        for j in range(i + 1):
            Bbar[i * n:(i + 1) * n, j * m_u:(j + 1) * m_u] = \
                np.linalg.matrix_power(A, i - j) @ B

    Q_blk             = block_diag(*([Q_S] * N_H))
    Q_blk[-n:, -n:]  = P_S
    R_blk             = block_diag(*([R_S] * N_H))
    Q_qp = Bbar.T @ Q_blk @ Bbar + R_blk + 1e-9 * np.eye(N_H * m_u)
    M_qp = Bbar.T @ Q_blk @ Abar

    # z-sinking correction
    d_aff = np.zeros(n); d_aff[12] = -DT * LEMN_Z0
    D_vec = np.zeros(N_H * n)
    acc, Apow = np.zeros(n), np.eye(n)
    for i in range(N_H):
        acc += Apow @ d_aff
        D_vec[i * n:(i + 1) * n] = acc
        Apow = A @ Apow

    X_ref = x_ref_seq[1:].flatten()
    r_qp  = -Bbar.T @ Q_blk @ (X_ref - D_vec)

    I_Nu = np.eye(N_H * m_u)
    A_u  = np.vstack([I_Nu, -I_Nu])
    c_u  = np.hstack([np.full(N_H * m_u, DU_MIN), np.full(N_H * m_u, -DU_MAX)])

    ang    = np.pi / 6
    idx_th = [i * n for i in range(N_H)]
    idx_ph = [i * n + 1 for i in range(N_H)]
    Bth, Ath = Bbar[idx_th, :], Abar[idx_th, :]
    Bph, Aph = Bbar[idx_ph, :], Abar[idx_ph, :]
    A_st = np.vstack([Bth, -Bth, Bph, -Bph])
    M_st = np.vstack([-Ath, Ath, -Aph, Aph])
    off  = np.hstack([-np.full(N_H, ang)] * 4)

    return Q_qp, M_qp, r_qp, A_u, c_u, A_st, M_st, off


def qp_from_x0(M_qp, r_qp, M_st, c_u, off, x0: np.ndarray, A_u):
    q_qp   = M_qp @ x0 + r_qp
    c_st   = off - M_st @ x0
    return q_qp, np.hstack([c_u, c_st])
