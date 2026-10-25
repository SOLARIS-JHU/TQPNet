import argparse
import os
import sys
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

from isaacgym import gymapi, gymtorch
import numpy as np

from batch_mpc import BatchModelPredictiveController
from batch_kinematics_controller import BatchKinematicsController
from tqpnet_mpc import BatchTQPNetController
from gait import Gait
from leg_controller import LegController
from linear_mpc_configs import LinearMpcConfig
from isaacgym_utils import *
from kinematics import quat2matrix_batch, quat2ZYXangle_batch
from mpc import ModelPredictiveController
from robot_configs import A1Config, AliengoConfig
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


NUM_ROBOTS = int(os.environ.get('NUM_ROBOTS', 4096))
STATE_ESTIMATION = False

def reset(gym, sim, robot_config):
    base_state_tensor_discription = gym.acquire_actor_root_state_tensor(sim)
    base_state_tensor = gymtorch.wrap_tensor(base_state_tensor_discription)
    base_state_tensor[:, 0:3] = torch.zeros_like(base_state_tensor[:, 0:3])  # base position
    base_state_tensor[:, 2] = robot_config.base_height_des
    base_state_tensor[:, 3:7] = torch.zeros_like(base_state_tensor[:, 3:7])  # base orientation
    base_state_tensor[:, 6] = 1.
    base_state_tensor[:, 7:10] = torch.zeros_like(base_state_tensor[:, 7:10]) # base linear velocity
    base_state_tensor[:, 10:13] = torch.zeros_like(base_state_tensor[:, 10:13]) # base angular velocity
    base_state_tensor_discription = gymtorch.unwrap_tensor(base_state_tensor)
    gym.set_actor_root_state_tensor(sim, base_state_tensor_discription)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver",
                        choices=["drake", "osqp", "qpth", "tqpnet", "cold-admm"],
                        default="osqp",
                        help="QP backend.  'drake'/'osqp' = per-robot single solve; "
                             "'qpth' = batch GPU solve via qpth.QPFunction; "
                             "'tqpnet' = TQPNet warm start + K-step ADMM-ReLU refine; "
                             "'cold-admm' = batch cold-start ADMM (x=β=0) + K-step refine.")
    parser.add_argument("--cold-k", type=int, default=500,
                        help="ADMM iterations for --solver cold-admm (default 500)")
    parser.add_argument("--cold-rho", type=float, default=0.1,
                        help="ADMM penalty ρ for cold-admm. Empirically 0.1 converges "
                             "fastest on the actual MPC QP (cond ~5e3, n=12·H).")
    parser.add_argument("--headless", action="store_true",
                        help="run without GUI viewer (no rendering)")
    parser.add_argument("--max-iter", type=int, default=0,
                        help="stop after this many iterations (0 = run forever)")
    parser.add_argument("--tqpnet-weights", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '../linear_mpc/tqpnet_weights/tqpnet_a1.pth'),
                        help="path to trained TQPNet weights")
    parser.add_argument("--tqpnet-k-refine", type=int, default=10,
                        help="ADMM-ReLU refine steps after TQPNet warm start")
    parser.add_argument("--log-vx", type=str, default=None,
                        help="if set, write per-step (iter, t_sec, vx_mean, cmd_vx) "
                             "CSV to this path (used by velocity-tracking plot)")
    args = parser.parse_args()
    use_batch_solver = (args.solver in ("qpth", "tqpnet", "cold-admm"))
    use_tqpnet = (args.solver == "tqpnet")
    use_cold_admm = (args.solver == "cold-admm")
    print(f"Using QP solver: {args.solver}  (batch={use_batch_solver})  "
          f"headless={args.headless}")

    gym = gymapi.acquire_gym()
    sim = create_simulation(
        gym,
        dt=LinearMpcConfig.dt_control,
        use_gpu_pipeline=True
    )

    robot_config = A1Config

    asset_root_path = os.path.join(os.path.dirname(__file__), '../robot/')
    robot_path = 'a1/urdf/a1.urdf'
    urdf_path = os.path.join(asset_root_path, robot_path)

    # Load URDF once — same asset shared by all N actors.
    robot_model = load_model(gym, sim, asset_root_path, robot_path, is_base_fixed=False)

    # RobotData is per-robot pinocchio bookkeeping.  Only the single-robot
    # --solver=drake/osqp path needs it; the batch path uses analytic FK.
    if use_batch_solver:
        l_robot_data: List[RobotData] = []
    else:
        l_robot_data: List[RobotData] = [
            RobotData(urdf_path, state_estimation=STATE_ESTIMATION)
            for _ in range(NUM_ROBOTS)
        ]

    creat_ground_plane(gym, sim)

    # Create N envs + N actors in a single call (all share the same asset).
    envs, actors = create_envs_actors(
        gym, sim, robot_model, robot_config.base_height_des, num_envs=NUM_ROBOTS
    )

    viewer = None if args.headless else add_viewer(gym, sim)

    gym.prepare_sim(sim)

    reset(gym, sim, robot_config)

    # Single shared Gait instance — all NUM_ROBOTS robots run in lockstep, so
    # one set_iteration / get_swing_state / get_gait_table call covers all of them.
    gait_shared = Gait.TROTTING10
    l_gait = [gait_shared] * NUM_ROBOTS   # kept for back-compat with single-robot --solver=drake

    # Batch kinematics + leg controller + swing trajectory.  Replaces the
    # 4096× pinocchio + LegController + SwingFootTrajectoryGenerator path
    # for the batch solvers (qpth / tqpnet / cold-admm).
    batch_kin_ctrl = BatchKinematicsController(
        num_robots=NUM_ROBOTS, robot_config=robot_config,
        dt_control=LinearMpcConfig.dt_control, gravity=LinearMpcConfig.gravity)

    # Legacy stack only needed by the single-robot --solver=drake/osqp path.
    if not use_batch_solver:
        l_leg_controller: List[LegController] = [
            LegController(robot_config.Kp_swing, robot_config.Kd_swing)
            for _ in range(NUM_ROBOTS)]
        l_swing_foot_trajs = [
            [SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(4)]
            for _ in range(NUM_ROBOTS)]
    else:
        l_leg_controller   = None
        l_swing_foot_trajs = None

    # MPC controllers — single, batch qpth, batch tqpnet, or batch cold-ADMM
    if use_tqpnet:
        batch_mpc = BatchTQPNetController(
            LinearMpcConfig, A1Config, num_robots=NUM_ROBOTS,
            model_path=args.tqpnet_weights, K_refine=args.tqpnet_k_refine,
            device='cuda:0')
        l_predictive_controller = None
    elif use_cold_admm:
        batch_mpc = BatchTQPNetController(
            LinearMpcConfig, A1Config, num_robots=NUM_ROBOTS,
            model_path=None, K_refine=args.cold_k,
            disable_nn=True, rho_refine=args.cold_rho, device='cuda:0')
        l_predictive_controller = None
    elif args.solver == "qpth":
        batch_mpc = BatchModelPredictiveController(
            LinearMpcConfig, A1Config, num_robots=NUM_ROBOTS, device='cuda:0')
        l_predictive_controller = None
    else:
        l_predictive_controller: List[ModelPredictiveController] = [
            ModelPredictiveController(LinearMpcConfig, A1Config) for _ in range(NUM_ROBOTS)
        ]
        batch_mpc = None

    vel_base_des = np.tile([1.4, 0., 0.], (NUM_ROBOTS, 1)).astype(np.float32)
    yaw_turn_rate_des = np.tile([0.], (NUM_ROBOTS, 1)).astype(np.float32)

    # B * 3 array of desired foot velocities in world frame, used for swing foot trajectory generation.
    vel_base_des = np.random.uniform(low=-0.1, high=1.0, size=(NUM_ROBOTS, 3)).astype(np.float32)
    vel_base_des[:, 1] = np.random.uniform(low=-0.2, high=0.2, size=(NUM_ROBOTS,)).astype(np.float32)  # add some lateral velocity
    vel_base_des[:, 2] = 0.0
    yaw_turn_rate_des = np.random.uniform(low=-0.4, high=0.4, size=(NUM_ROBOTS, 1)).astype(np.float32)


    iter_counter = 0
    render_counter = 0
    torque_cmds_tensor = torch.zeros(NUM_ROBOTS, 12, device='cuda:0', dtype=torch.float32)

    # Velocity-tracking CSV (for plot 4).  One line per sim step.
    vx_csv = None
    if args.log_vx:
        vx_csv = open(args.log_vx, 'w')
        vx_csv.write("iter,t_sec,vx_mean,cmd_vx\n")

    while True:
        # exit conditions: viewer closed (GUI mode) or max-iter reached (any mode)
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break
        if args.max_iter > 0 and iter_counter >= args.max_iter:
            break

        # step graphics — only when running with viewer
        if viewer is not None and render_counter % 30 == 0:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            render_counter = 0
        render_counter += 1

        base_state_tensor_discription = gym.acquire_actor_root_state_tensor(sim)
        base_state_tensor: torch.Tensor = gymtorch.wrap_tensor(base_state_tensor_discription)
        joint_state_tensor_discription = gym.acquire_dof_state_tensor(sim)
        joint_state_tensor: torch.Tensor = gymtorch.wrap_tensor(joint_state_tensor_discription)

        # ── Phase 1: read all robot states from sim ───────────────────────
        # Single GPU→CPU transfer (instead of 4096× per-robot .cpu() syncs).
        base_state_np  = base_state_tensor.detach().cpu().numpy()              # (B, 13)
        joint_state_np = joint_state_tensor.detach().cpu().numpy()             # (B*12, 2)

        # Vectorized quaternion reorder (Isaac Gym im-re → mujoco re-im).
        quat_imre = base_state_np[:, 3:7]
        quat_reim = quat_imre[:, [3, 0, 1, 2]].astype(np.float32)              # (B, 4)
        pos_base_arr = base_state_np[:, 0:3].astype(np.float32)
        lin_vel_arr  = base_state_np[:, 7:10].astype(np.float32)
        ang_vel_arr  = base_state_np[:, 10:13].astype(np.float32)
        joint_q  = joint_state_np[:, 0].reshape(NUM_ROBOTS, 12).astype(np.float32)
        joint_qd = joint_state_np[:, 1].reshape(NUM_ROBOTS, 12).astype(np.float32)

        # Batched analytic FK + Jacobian (replaces 4096× pinocchio.update).
        R_base_arr = quat2matrix_batch(quat_reim)                              # (B, 3, 3)
        rpy_arr    = quat2ZYXangle_batch(quat_reim)                            # (B, 3)

        # Single shared Gait — set/get once, broadcast to per-robot tables.
        iters_per_mpc = (batch_mpc.iterations_between_mpc if use_batch_solver
                         else l_predictive_controller[0].iterations_between_mpc)
        gait_shared.set_iteration(iters_per_mpc, iter_counter)
        gait_table_one    = gait_shared.get_gait_table()                       # (4·H,)
        gait_tables_batch = np.broadcast_to(
            gait_table_one[None, :],
            (NUM_ROBOTS, 4 * LinearMpcConfig.horizon)).astype(np.float32)

        if use_batch_solver:
            # ── Run the batched analytic FK once for the whole population ──
            kin = batch_kin_ctrl.update_kinematics(
                q_b            = joint_q,
                qdot_b         = joint_qd,
                pos_base_b     = pos_base_arr,
                R_base_b       = R_base_arr,
                lin_vel_base_b = lin_vel_arr,
                ang_vel_base_b = ang_vel_arr,
            )
            pos_base_feet_arr = kin['pos_base_feet_world']                      # (B, 4, 3)

            # ── Phase 2: MPC solve (batched state intake + batched QP) ──
            batch_mpc.update_robot_state_batch(
                rpy_b           = rpy_arr,
                pos_base_b      = pos_base_arr,
                lin_vel_b       = lin_vel_arr,
                ang_vel_b       = ang_vel_arr,
                pos_base_feet_b = pos_base_feet_arr,
                R_base_b        = R_base_arr,
            )
            contact_forces_batch = batch_mpc.update_mpc_if_needed_batch(
                iter_counter, vel_base_des, yaw_turn_rate_des, gait_tables_batch)

            # ── Phase 3: batched leg controller + swing trajectory ──
            base_pos_des, base_vel_des, swing_states_shared = \
                batch_kin_ctrl.update_swing_targets(
                    gait_shared, np.asarray(vel_base_des, dtype=np.float32),
                    yaw_turn_rate_des)
            torque_cmds_np = batch_kin_ctrl.compute_torques(
                contact_forces_world   = contact_forces_batch,
                swing_states           = swing_states_shared,
                base_pos_swingfoot_des = base_pos_des,
                base_vel_swingfoot_des = base_vel_des,
            )
        else:
            # Single-robot --solver=drake/osqp path keeps the per-robot loop
            # (this branch is only used for tiny NUM_ROBOTS sanity runs).
            for robot_idx, robot_data in enumerate(l_robot_data):
                robot_data.update(
                    pos_base=pos_base_arr[robot_idx],
                    quat_base=quat_reim[robot_idx],
                    lin_vel_base=lin_vel_arr[robot_idx],
                    ang_vel_base=ang_vel_arr[robot_idx],
                    q=joint_q[robot_idx],
                    qdot=joint_qd[robot_idx],
                )
            contact_forces_batch = np.zeros((NUM_ROBOTS, 12), dtype=np.float32)
            for robot_idx in range(NUM_ROBOTS):
                pc = l_predictive_controller[robot_idx]
                pc.update_robot_state(l_robot_data[robot_idx])
                contact_forces_batch[robot_idx, :] = pc.update_mpc_if_needed(
                    iter_counter, vel_base_des, yaw_turn_rate_des,
                    gait_tables_batch[robot_idx], solver='drake',
                    debug=False, iter_debug=0)
            swing_states = gait_shared.get_swing_state()
            torque_cmds_np = np.empty((NUM_ROBOTS, 12), dtype=np.float32)
            for robot_idx, (robot_data, leg_controller, swing_foot_trajs) \
                in enumerate(zip(l_robot_data, l_leg_controller, l_swing_foot_trajs)):
                contact_forces = contact_forces_batch[robot_idx]
                pos_targets_swingfeet = np.zeros((4, 3))
                vel_targets_swingfeet = np.zeros((4, 3))
                for leg_idx in range(4):
                    if swing_states[leg_idx] > 0:
                        swing_foot_trajs[leg_idx].set_foot_placement(
                            robot_data, gait_shared, vel_base_des, yaw_turn_rate_des)
                        p_des, v_des = swing_foot_trajs[leg_idx].compute_traj_swingfoot(
                            robot_data, gait_shared)
                        pos_targets_swingfeet[leg_idx, :] = p_des
                        vel_targets_swingfeet[leg_idx, :] = v_des
                torque_cmds_np[robot_idx, :] = leg_controller.update(
                    robot_data, contact_forces, swing_states,
                    pos_targets_swingfeet, vel_targets_swingfeet)

        # Single host→device upload (replaces 4096× torch.tensor(...) GPU writes).
        torque_cmds_tensor.copy_(
            torch.from_numpy(torque_cmds_np).to(torque_cmds_tensor.device))

        torque_cmds_tensor_discription = gymtorch.unwrap_tensor(torque_cmds_tensor)
        gym.set_dof_actuation_force_tensor(sim, torque_cmds_tensor_discription)

        gym.fetch_results(sim, True)
        gym.simulate(sim)

        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)

        # Walking-quality monitor: every 100 steps print base z + vx mean
        if iter_counter % 100 == 0:
            z_mean  = base_state_tensor[:, 2].mean().item()
            z_min   = base_state_tensor[:, 2].min().item()
            vx_mean = base_state_tensor[:, 7].mean().item()
            # vel_base_des may be (3,) shared or (B, 3) per-robot — index either way.
            cmd_vx_print = float(np.asarray(vel_base_des, dtype=np.float32).reshape(-1, 3)[0, 0])
            print(f"[t={iter_counter:5d}]  z_mean={z_mean:.3f}  z_min={z_min:.3f}  "
                  f"vx_mean={vx_mean:+.3f}  (target z={robot_config.base_height_des:.2f}, "
                  f"target vx={cmd_vx_print:.2f})")

        # Per-step velocity log for plot 4.
        if vx_csv is not None:
            vx_now = base_state_tensor[:, 7].mean().item()
            t_sec = iter_counter * LinearMpcConfig.dt_control
            cmd_vx = float(np.asarray(vel_base_des, dtype=np.float32).reshape(-1, 3)[0, 0])
            vx_csv.write(f"{iter_counter},{t_sec:.4f},{vx_now:.5f},{cmd_vx:.4f}\n")

        iter_counter += 1

        if iter_counter == 20000:
            reset(gym, sim, robot_config)
            iter_counter = 0
            render_counter = 0

    if viewer is not None:
        gym.destroy_viewer(viewer)
    if vx_csv is not None:
        vx_csv.close()
    gym.destroy_sim(sim)

if __name__ == '__main__':
    main()