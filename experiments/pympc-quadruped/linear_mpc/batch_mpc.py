"""Batch Model Predictive Controller using qpth (differentiable batch QP solver).

Drop-in replacement for `ModelPredictiveController` (mpc.py) when controlling
N quadruped robots simultaneously. All N MPC QPs are stacked into a single
qpth.QPFunction call, solved on GPU in parallel.

QP formulation (per robot):
    min  1/2 u^T H u + g^T u
    s.t. C_lb <= C u <= C_ub        (friction cone + fz bounds)

is converted to qpth's form:
    min  1/2 z^T Q z + p^T z
    s.t. G z <= h
where G = [C; -C] and h = [C_ub; -C_lb], with rows whose bounds are ±inf dropped.

Per leg per horizon step we end up with 6 inequality rows
(4 friction-cone half-spaces + fz upper + fz lower).

Usage
-----
    batch_mpc = BatchModelPredictiveController(LinearMpcConfig, A1Config,
                                                num_robots=4, device='cuda:0')
    for robot_idx in range(num_robots):
        batch_mpc.update_robot_state(robot_idx, robot_data_list[robot_idx])

    contact_forces_batch = batch_mpc.update_mpc_if_needed_batch(
        iter_counter, vel_base_des, yaw_turn_rate_des, gait_tables)
    # contact_forces_batch shape: (num_robots, 12)

The first-horizon GRF for each robot is returned (numpy array, shape (12,) per
robot).  Internally GPU computation is batched.
"""
from __future__ import annotations
import os
import sys
import time
from typing import List, TYPE_CHECKING, Union

sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np
import torch
from qpth.qp import QPFunction

from kinematics import quat2ZYXangle, vec2so3
from linear_mpc_configs import LinearMpcConfig
from robot_configs import RobotConfig

# RobotData is only used in type hints + runtime as duck-typed object.
# Import lazily so this module can be smoke-imported without pinocchio.
if TYPE_CHECKING:
    from robot_data import RobotData


class BatchModelPredictiveController:
    """Same MPC formulation as `ModelPredictiveController`, batched over N robots."""

    def __init__(self,
                 mpc_config: LinearMpcConfig,
                 robot_config: RobotConfig,
                 num_robots: int,
                 device: Union[str, torch.device] = 'cuda:0'):
        # state: x = [θ, p, ω, v, g], dim 13
        self.num_state = 13
        # input: u = [f1..f4], dim 12
        self.num_input = 12
        self.B = num_robots
        self.device = torch.device(device)

        self._load_parameters(mpc_config, robot_config)

        # Per-robot stateful trajectory bookkeeping (mirrors single MPC).
        self.is_initialized = np.zeros(num_robots, dtype=bool)
        self.is_first_run = np.ones(num_robots, dtype=bool)
        self.current_state = np.zeros((num_robots, 13), dtype=np.float32)
        self.yaw = np.zeros(num_robots, dtype=np.float32)
        # Cached per-robot rotation matrices (B,3,3); maintained in update_robot_state*
        # so the accumulator can run as a single batched einsum instead of a Python loop.
        self.R_base = np.zeros((num_robots, 3, 3), dtype=np.float32)
        self.pos_base_feet = np.zeros((num_robots, 4, 3), dtype=np.float32)
        self.roll_init = np.zeros(num_robots, dtype=np.float32)
        self.pitch_init = np.zeros(num_robots, dtype=np.float32)
        self.xpos_base_desired = np.zeros(num_robots, dtype=np.float32)
        self.ypos_base_desired = np.zeros(num_robots, dtype=np.float32)
        self.yaw_desired = np.zeros(num_robots, dtype=np.float32)
        self._robot_data_list: list = [None] * num_robots
        self.__contact_forces = np.zeros((num_robots, 12 * self.horizon), dtype=np.float32)

        # Pre-build inputs that don't depend on robot state
        self._build_static_qp_terms()

    # ──────────────────────────────────────────────────────────────────────

    def _load_parameters(self, mpc_config: LinearMpcConfig, robot_config: RobotConfig):
        self.dt_control = mpc_config.dt_control
        self.iterations_between_mpc = mpc_config.iteration_between_mpc
        self.dt = 0.05
        self.horizon = mpc_config.horizon
        self.mu = mpc_config.friction_coef
        self.fz_max = robot_config.fz_max
        self.gravity = mpc_config.gravity

        self.base_inertia_base = robot_config.base_inertia_base
        self.mass = robot_config.mass_base
        self.com_height_des = robot_config.base_height_des

        # MPC weights (shared across robots)
        Qi = mpc_config.Q
        Ri = mpc_config.R
        self.Qbar_np = np.kron(np.identity(self.horizon), Qi)
        self.Rbar_np = np.kron(np.identity(self.horizon), Ri)

    def _build_static_qp_terms(self):
        H = self.horizon
        # Q_blk, R_blk on GPU (shared)
        self.Qbar = torch.tensor(self.Qbar_np, dtype=torch.float32, device=self.device)
        self.Rbar = torch.tensor(self.Rbar_np, dtype=torch.float32, device=self.device)
        # Per-leg friction cone block in qpth Gz <= h form (6 rows).
        # rows: [-1, 0, -μ; +1, 0, -μ; 0, -1, -μ; 0, +1, -μ;  0, 0, +1; 0, 0, -1]
        mu = self.mu
        G_leg = torch.tensor([
            [-1.0,  0.0, -mu],   # -fx - μfz <= 0
            [+1.0,  0.0, -mu],   # +fx - μfz <= 0
            [ 0.0, -1.0, -mu],   # -fy - μfz <= 0
            [ 0.0, +1.0, -mu],   # +fy - μfz <= 0
            [ 0.0,  0.0, +1.0],  # +fz <= gait·fz_max
            [ 0.0,  0.0, -1.0],  # -fz <= 0
        ], dtype=torch.float32, device=self.device)                        # (6, 3)
        # Stack across 4 legs per horizon step → (6·4, 12) block per step
        zeros3 = torch.zeros(6, 3, device=self.device)
        leg_block_rows = []
        for leg_in_step in range(4):
            row = [zeros3, zeros3, zeros3, zeros3]
            row[leg_in_step] = G_leg
            leg_block_rows.append(torch.cat(row, dim=1))                    # (6, 12)
        G_step = torch.cat(leg_block_rows, dim=0)                          # (24, 12)
        # Block-diag across H horizon steps → (24·H, 12·H)
        G_full = torch.zeros(24 * H, 12 * H, device=self.device)
        for h in range(H):
            G_full[24 * h:24 * (h + 1), 12 * h:12 * (h + 1)] = G_step
        self.G_template = G_full                                           # (24·H, 12·H)

        # Tile to batch
        self.G_batch = self.G_template.unsqueeze(0).expand(self.B, -1, -1).contiguous()

    # ──────────────────────────────────────────────────────────────────────
    # Per-robot state intake (same signature semantics as single MPC).

    def update_robot_state(self, robot_idx: int, robot_data: 'RobotData'):
        if not self.is_initialized[robot_idx]:
            self.current_state[robot_idx, :] = 0.0
            self.is_initialized[robot_idx] = True

        self._robot_data_list[robot_idx] = robot_data

        rpy_base = quat2ZYXangle(robot_data.quat_base)
        pos_base = np.asarray(robot_data.pos_base, dtype=np.float32)
        omega_base = np.asarray(robot_data.ang_vel_base, dtype=np.float32)
        vel_base = np.asarray(robot_data.lin_vel_base, dtype=np.float32)

        cs = self.current_state[robot_idx]
        cs[0:3] = rpy_base
        cs[3:6] = pos_base
        cs[6:9] = omega_base
        cs[9:12] = vel_base
        cs[12] = -self.gravity
        self.yaw[robot_idx] = rpy_base[2]
        self.pos_base_feet[robot_idx] = robot_data.pos_base_feet
        self.R_base[robot_idx] = np.asarray(robot_data.R_base, dtype=np.float32)

    def update_robot_state_batch(self,
                                  rpy_b: np.ndarray,
                                  pos_base_b: np.ndarray,
                                  lin_vel_b: np.ndarray,
                                  ang_vel_b: np.ndarray,
                                  pos_base_feet_b: np.ndarray,
                                  R_base_b: np.ndarray = None,
                                  robot_data_list=None) -> None:
        """Batched state intake (no Python per-robot loop)."""
        cs = self.current_state
        cs[:, 0:3]  = rpy_b
        cs[:, 3:6]  = pos_base_b
        cs[:, 6:9]  = ang_vel_b
        cs[:, 9:12] = lin_vel_b
        cs[:, 12]   = -self.gravity
        self.yaw[:]            = rpy_b[:, 2]
        self.pos_base_feet[:]  = pos_base_feet_b
        if R_base_b is not None:
            self.R_base[:] = R_base_b
        self.is_initialized[:] = True
        if robot_data_list is not None:
            self._robot_data_list = robot_data_list

    # ──────────────────────────────────────────────────────────────────────

    def update_mpc_if_needed_batch(self,
                                    iter_counter: int,
                                    base_vel_base_des: np.ndarray,
                                    yaw_turn_rate_des: Union[float, np.ndarray],
                                    gait_tables: np.ndarray) -> np.ndarray:
        """Solve all N MPC QPs in a single qpth call.

        Args:
            iter_counter: int (shared across robots)
            base_vel_base_des: (3,) common command in base frame, OR (B, 3) per-robot
            yaw_turn_rate_des: float (shared) OR (B,) per-robot
            gait_tables: (B, 4·horizon) contact schedule per robot

        Returns:
            contact_forces_first_horizon: (B, 12) first-horizon GRF per robot.
        """
        B = self.B
        H = self.horizon

        # Normalize cmd shapes to (B,3) and (B,) regardless of how the script
        # supplies them (scalar / 1D / per-robot).  Imported from tqpnet_mpc
        # so the two batch controllers stay in sync.
        from tqpnet_mpc import _normalize_cmd_shapes
        base_vel_base_des, yaw_turn_rate_des = _normalize_cmd_shapes(
            base_vel_base_des, yaw_turn_rate_des, B)

        # Vectorized desired-pose accumulator (replaces per-robot Python loop).
        v_des_world_all = np.einsum('bij,bj->bi', self.R_base,
                                     base_vel_base_des).astype(np.float32)        # (B,3)
        first = self.is_first_run
        yaw_rate = np.broadcast_to(yaw_turn_rate_des, (B,)).astype(np.float32)
        new_xpos = self.xpos_base_desired + self.dt_control * v_des_world_all[:, 0]
        new_ypos = self.ypos_base_desired + self.dt_control * v_des_world_all[:, 1]
        new_yaw  = self.yaw                + self.dt_control * yaw_rate
        self.xpos_base_desired = np.where(first, 0.0, new_xpos).astype(np.float32)
        self.ypos_base_desired = np.where(first, 0.0, new_ypos).astype(np.float32)
        self.yaw_desired       = np.where(first, self.yaw, new_yaw).astype(np.float32)
        self.is_first_run[:] = False

        # Decide whether MPC needs to be updated this iteration
        if iter_counter % self.iterations_between_mpc == 0:
            ref_traj_batch = self._generate_reference_trajectory_batch(
                base_vel_base_des, yaw_turn_rate_des)
            t0 = time.time()
            self.__contact_forces = self._solve_mpc_batch(ref_traj_batch, gait_tables)
            print(f"Batch MPC ({B} robots) solved in {time.time() - t0:.4f}s.")

        return self.__contact_forces[:, :12]

    # ──────────────────────────────────────────────────────────────────────
    # Reference trajectory generation (same logic as single MPC, vectorized).

    def _generate_reference_trajectory_batch(self,
                                              base_vel_base_des: np.ndarray,
                                              yaw_turn_rate_des: np.ndarray) -> np.ndarray:
        """Fully vectorized over robots — uses self.R_base (no robot_data_list)."""
        B, H = self.B, self.horizon
        cs = self.current_state                                                # (B, 13)

        from tqpnet_mpc import _normalize_cmd_shapes
        base_vel_base_des, yaw_turn_rate_des = _normalize_cmd_shapes(
            base_vel_base_des, yaw_turn_rate_des, B)

        v_des_world = np.einsum('bij,bj->bi', self.R_base,
                                 base_vel_base_des).astype(np.float32)         # (B, 3)
        yaw_rate = yaw_turn_rate_des                                            # already (B,)

        max_pos_err = 0.1
        np.clip(self.xpos_base_desired,
                cs[:, 3] - max_pos_err, cs[:, 3] + max_pos_err,
                out=self.xpos_base_desired)
        np.clip(self.ypos_base_desired,
                cs[:, 4] - max_pos_err, cs[:, 4] + max_pos_err,
                out=self.ypos_base_desired)
        cur_x = self.xpos_base_desired
        cur_y = self.ypos_base_desired

        vx, vy = cs[:, 9], cs[:, 10]
        pitch_mask = np.abs(vx) > 0.2
        if pitch_mask.any():
            self.pitch_init[pitch_mask] += (self.dt
                                              * (-cs[pitch_mask, 1])
                                              / vx[pitch_mask])
        roll_mask = np.abs(vy) > 0.1
        if roll_mask.any():
            self.roll_init[roll_mask] += (self.dt
                                            * (-cs[roll_mask, 0])
                                            / vy[roll_mask])
        np.clip(self.roll_init,  -0.25, 0.25, out=self.roll_init)
        np.clip(self.pitch_init, -0.25, 0.25, out=self.pitch_init)
        roll_comp  = vy * self.roll_init
        pitch_comp = vx * self.pitch_init

        ref = np.zeros((B, self.num_state * H), dtype=np.float32)
        k_arr = np.arange(H, dtype=np.float32)
        ref[:, 2::self.num_state]  = self.yaw_desired[:, None] + k_arr[None, :] * self.dt * yaw_rate[:, None]
        ref[:, 3::self.num_state]  = cur_x[:, None] + k_arr[None, :] * self.dt * v_des_world[:, 0:1]
        ref[:, 4::self.num_state]  = cur_y[:, None] + k_arr[None, :] * self.dt * v_des_world[:, 1:2]
        ref[:, 0::self.num_state]  = roll_comp[:, None]
        ref[:, 1::self.num_state]  = pitch_comp[:, None]
        ref[:, 5::self.num_state]  = self.com_height_des
        ref[:, 8::self.num_state]  = yaw_rate[:, None]
        ref[:, 9::self.num_state]  = v_des_world[:, 0:1]
        ref[:, 10::self.num_state] = v_des_world[:, 1:2]
        ref[:, 12::self.num_state] = -self.gravity
        return ref

    # ──────────────────────────────────────────────────────────────────────
    # Batched dynamics + condense + cost + constraints + solve.

    def _build_Ac_Bc_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 13, 13), (B, 13, 12) on device."""
        B = self.B
        Ac = torch.zeros(B, 13, 13, device=self.device)
        Bc = torch.zeros(B, 13, 12, device=self.device)

        yaw_t = torch.tensor(self.yaw, dtype=torch.float32, device=self.device)
        cy = torch.cos(yaw_t); sy = torch.sin(yaw_t)
        zero = torch.zeros_like(cy); one = torch.ones_like(cy)
        Rz = torch.stack([
            torch.stack([cy, -sy, zero], dim=-1),
            torch.stack([sy,  cy, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ], dim=-2)                                                     # (B, 3, 3)

        I_base = torch.tensor(self.base_inertia_base, dtype=torch.float32,
                               device=self.device).unsqueeze(0).expand(B, -1, -1)
        world_I = Rz @ I_base @ Rz.transpose(-1, -2)
        world_I_inv = torch.linalg.inv(world_I)

        Ac[:, 0:3, 6:9] = Rz.transpose(-1, -2)
        Ac[:, 3:6, 9:12] = torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1)
        Ac[:, 11, 12] = 1.0

        # vec2so3 batched
        feet = torch.tensor(self.pos_base_feet, dtype=torch.float32, device=self.device)  # (B,4,3)
        rx = feet[..., 0]; ry = feet[..., 1]; rz = feet[..., 2]
        zero_b = torch.zeros_like(rx)
        # (B, 4, 3, 3)
        skew = torch.stack([
            torch.stack([zero_b, -rz, ry], dim=-1),
            torch.stack([rz, zero_b, -rx], dim=-1),
            torch.stack([-ry, rx, zero_b], dim=-1),
        ], dim=-2)

        inv_m_eye3 = (torch.eye(3, device=self.device) / self.mass).unsqueeze(0).expand(B, -1, -1)
        for i in range(4):
            # B[6:9, 3i:3i+3] = world_I_inv @ skew_i
            Bc[:, 6:9, 3*i:3*(i+1)] = world_I_inv @ skew[:, i]
            Bc[:, 9:12, 3*i:3*(i+1)] = inv_m_eye3
        return Ac, Bc

    def _discretize_batch(self, Ac: torch.Tensor, Bc: torch.Tensor):
        """ZOH discretization via matrix exponential, batched."""
        B = self.B
        n, m = self.num_state, self.num_input
        dim = n + m
        sq = torch.zeros(B, dim, dim, device=self.device)
        sq[:, :n, :n] = Ac * self.dt
        sq[:, :n, n:] = Bc * self.dt
        # exp([[A, B];[0, 0]]·dt) → [[Ad, Bd]; [0, I]]
        E = torch.linalg.matrix_exp(sq)
        Ad = E[:, :n, :n]
        Bd = E[:, :n, n:]
        return Ad, Bd

    def _build_Sx_Su_batch(self, Ad: torch.Tensor, Bd: torch.Tensor):
        """(B, 13H, 13), (B, 13H, 12H)."""
        B = self.B
        n, m, H = self.num_state, self.num_input, self.horizon
        Sx = torch.zeros(B, n * H, n, device=self.device)
        Su = torch.zeros(B, n * H, m * H, device=self.device)
        # Build power_of_A: A^1, A^2, ..., A^H
        Apow = [torch.eye(n, device=self.device).unsqueeze(0).expand(B, -1, -1)]  # A^0
        for _ in range(H):
            Apow.append(Apow[-1] @ Ad)
        for i in range(H):
            Sx[:, n*i:n*(i+1), :] = Apow[i + 1]
            for j in range(H):
                if i >= j:
                    Su[:, n*i:n*(i+1), m*j:m*(j+1)] = Apow[i - j] @ Bd
        return Sx, Su

    def _solve_mpc_batch(self, ref_traj_np: np.ndarray, gait_tables_np: np.ndarray) -> np.ndarray:
        """Returns (B, 12·H)."""
        B = self.B
        H = self.horizon

        # 1. dynamics
        Ac, Bc = self._build_Ac_Bc_batch()
        Ad, Bd = self._discretize_batch(Ac, Bc)
        Sx, Su = self._build_Sx_Su_batch(Ad, Bd)                       # (B, n·H, n), (B, n·H, m·H)

        # 2. cost: H = 2(Su^T Q Su + R), g = 2 Su^T Q (Sx xt - Xref)
        xt = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)  # (B, 13)
        Xref = torch.tensor(ref_traj_np, dtype=torch.float32, device=self.device)        # (B, 13H)
        Q = self.Qbar.unsqueeze(0).expand(B, -1, -1)                  # (B, 13H, 13H)
        R = self.Rbar.unsqueeze(0).expand(B, -1, -1)                  # (B, 12H, 12H)

        SuT = Su.transpose(-1, -2)
        qpH = 2.0 * (SuT @ Q @ Su + R)                                 # (B, 12H, 12H)
        # add tiny Tikhonov for qpth numerical stability
        qpH = qpH + 1e-6 * torch.eye(qpH.shape[-1], device=self.device).unsqueeze(0)
        delta = (Sx @ xt.unsqueeze(-1)).squeeze(-1) - Xref              # (B, 13H)
        qpg = 2.0 * (SuT @ Q @ delta.unsqueeze(-1)).squeeze(-1)         # (B, 12H)

        # 3. inequality constraints: Gz <= h
        # G_batch is precomputed; build h: per (h, leg) → 6 entries
        gait_t = torch.tensor(gait_tables_np, dtype=torch.float32, device=self.device)  # (B, 4·H)
        h_batch = torch.zeros(B, 24 * H, device=self.device)
        # row 0..3 (friction cone) = 0; row 4 (fz upper) = gait·fz_max; row 5 (fz lower) = 0
        # For each (horizon h, leg i), index in h_batch is: 24*h + 6*i + {0..5}
        # We can vectorize: gait_t shape (B, 4·H); we want to put at indices 24*h + 6*leg + 4
        H_ = H
        for h_idx in range(H_):
            for leg_idx in range(4):
                gait_lh = gait_t[:, 4 * h_idx + leg_idx]             # (B,)
                h_batch[:, 24 * h_idx + 6 * leg_idx + 4] = gait_lh * self.fz_max

        # solve with osqp
        # return self._fallback_osqp(qpH, qpg, self.G_batch, h_batch)

        # 4. solve via qpth
        # qpth: min 1/2 z^T Q z + p^T z s.t. Gz <= h, Az = b
        # We need q_p (called "p" in qpth) instead of Q,p named differently here.
        e = torch.autograd.Variable(torch.Tensor())
        # try:
            
        # except Exception as e:
        #     print(f"[BatchMPC] qpth failed: {e}; falling back to per-robot OSQP.")
        #     # return self._fallback_osqp(qpH, qpg, self.G_batch, h_batch)
        z = QPFunction(eps=1e-10, verbose=0, check_Q_spd=False, maxIter=500)(
                qpH.double(), qpg.double(), self.G_batch.double(), h_batch.double(), e, e)
        return z.detach().cpu().numpy()                                # (B, 12H)

    # ──────────────────────────────────────────────────────────────────────

    def _fallback_osqp(self, qpH, qpg, G_batch, h_batch):
        """If qpth diverges, fall back to per-robot OSQP for safety."""
        """
        \min 1/2 x^T P x + q^T x
        s.t. l <= Ax <= u
        prob.setup(P, q, A, l, u)
        """
        from scipy import sparse
        import osqp
        B = self.B
        H = self.horizon
        out = np.zeros((B, 12 * H), dtype=np.float32)
        for i in range(B):
            P = sparse.csc_matrix(qpH[i].cpu().numpy())
            q = qpg[i].cpu().numpy()
            A = sparse.csc_matrix(G_batch[i].cpu().numpy())
            l = -np.inf * np.ones(24 * H)
            u = h_batch[i].cpu().numpy()
            prob = osqp.OSQP()
            prob.setup(P, q, A, l, u, verbose=False)
            res = prob.solve()
            if res.x is not None:
                out[i] = res.x
        return out
