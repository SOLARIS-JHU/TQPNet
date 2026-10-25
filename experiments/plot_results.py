"""Convenience script: run all experiment plot scripts.

Individual scripts:
  python -m experiments.admm.plot_admm_convergence  # ADMM convergence (Fig 2)
  python -m experiments.ablation.plot_ablation  # Ablation study
  python -m experiments.uav.plot_uav            # UAV MPC closed-loop
  python -m experiments.plot_paper                  # Scalability latency/offline figures (Fig 5)
"""
from experiments.admm.plot_admm_convergence import main as plot_admm_convergence_main
from experiments.ablation.plot_ablation import main as plot_ablation_main
from experiments.uav.plot_uav     import plot_uav


if __name__ == "__main__":
    print("=== ADMM convergence ===")
    plot_admm_convergence_main()
    print("\n=== Ablation ===")
    plot_ablation_main()
    print("\n=== UAV MPC ===")
    plot_uav()
    print("\n=== Scalability ===")
    try:
        from experiments.scalability.scalability_mpc import plot_scalability_mpc
        plot_scalability_mpc()
    except Exception as e:
        print(f"  [skip] {e}")
    print("\nAll done.")
