# main.py — เปรียบเทียบ SA, SHO, WOA ด้วยค่ากลางชุดเดียว (equal-evaluations)
from pathlib import Path
from copy import deepcopy
from math import ceil
import argparse

from src.parser import load_ucla_benchmark
from src.algorithms.SA import simulated_annealing
from src.algorithms.SHO import spotted_hyena_optimizer
from src.algorithms.WOA import whale_optimizer

# -------------------------------
# 1) CONFIG (key-value)
# -------------------------------
CONFIG = {
    "shared": {
        "N_evals": 80,         # งบ evaluation เริ่มต้น (ปรับได้จาก CLI ด้วย --evals)
        "grid": (64, 64),
        "overlap_grid": (128, 128),
        "alpha": 0.5,
        "beta": 1e-3,
        "gamma": 0.1,
        "seed": 42,
        "verbose": True,
    },
    # SA: 1 evaluation ต่อ 1 iteration
    "sa": {
        "t0": None,           # ให้ SA auto-derive T0 จาก delta cost
        "t_min": 1e-6,
        "cooling": 0.99,
        "move_disp_prob": 0.7,
        "disp_scale_init": 0.02,
        "disp_scale_final": 0.002,
    },
    # SHO: ~pop_size evaluations ต่อ 1 iteration
    "sho": {
        "pop_size": 4,
        "local_prob": 0.10,
        "local_swap_k": 6,
    },
    # WOA: ~pop_size evaluations ต่อ 1 iteration
    "woa": {
        "pop_size": 1,
        "module_sample": 4096,
        "init_jitter": 0.002,
    },
    # เลือกว่าจะรันอะไรบ้าง
    "run": ["sa", "sho", "woa"],
}

# -------------------------------
# 2) Runners (equal-evals)
# -------------------------------
def run_sa(chip, shared, sa_cfg, N_evals):
    return simulated_annealing(
        deepcopy(chip),
        iters=N_evals,  # equal-evals
        grid_size=shared["grid"],
        overlap_grid=shared["overlap_grid"],
        alpha=shared["alpha"],
        beta=shared["beta"],
        gamma=shared["gamma"],
        t0=sa_cfg["t0"],
        t_min=sa_cfg["t_min"],
        cooling=sa_cfg["cooling"],
        move_disp_prob=sa_cfg["move_disp_prob"],
        disp_scale_init=sa_cfg["disp_scale_init"],
        disp_scale_final=sa_cfg["disp_scale_final"],
        seed=shared["seed"],
        verbose=shared["verbose"],
    )


def run_sho(chip, shared, sho_cfg, N_evals):
    pop = sho_cfg["pop_size"]
    iters = ceil(N_evals / pop)  # equal-evals (ประมาณ)
    return spotted_hyena_optimizer(
        deepcopy(chip),
        pop_size=pop,
        iters=iters,
        grid_size=shared["grid"],
        overlap_grid=shared["overlap_grid"],
        alpha=shared["alpha"],
        beta=shared["beta"],
        gamma=shared["gamma"],
        local_prob=sho_cfg["local_prob"],
        local_swap_k=sho_cfg["local_swap_k"],
        seed=shared["seed"],
        verbose=shared["verbose"],
    )


def run_woa(chip, shared, woa_cfg, N_evals):
    pop = woa_cfg["pop_size"]
    iters = ceil(N_evals / pop)  # equal-evals (ประมาณ)
    return whale_optimizer(
        deepcopy(chip),
        pop_size=pop,
        iters=iters,
        grid_size=shared["grid"],
        overlap_grid=shared["overlap_grid"],
        alpha=shared["alpha"],
        beta=shared["beta"],
        gamma=shared["gamma"],
        seed=shared["seed"],
        verbose=shared["verbose"],
        module_sample=woa_cfg["module_sample"],
        init_jitter=woa_cfg["init_jitter"],
    )


# -------------------------------
# 3) main()
# -------------------------------
def main():
    BASE = Path(__file__).resolve().parent / "data" / "ispd2005_benchmarks"

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=str, default=str(BASE / "bigblue1.inf.nodes"))
    parser.add_argument("--nets",  type=str, default=str(BASE / "bigblue1.nets"))
    parser.add_argument("--pl",    type=str, default=str(BASE / "bigblue1.pl"))

    parser.add_argument("--evals", type=int, default=CONFIG["shared"]["N_evals"],
                        help="equal-evaluations budget")

    parser.add_argument("--alpha", type=float, default=CONFIG["shared"]["alpha"])
    parser.add_argument("--beta",  type=float, default=CONFIG["shared"]["beta"])
    parser.add_argument("--gamma", type=float, default=CONFIG["shared"]["gamma"])

    parser.add_argument("--grid",  type=str,
                        default=f"{CONFIG['shared']['grid'][0]},{CONFIG['shared']['grid'][1]}")
    parser.add_argument("--ovlp",  type=str,
                        default=f"{CONFIG['shared']['overlap_grid'][0]},{CONFIG['shared']['overlap_grid'][1]}")
    parser.add_argument("--seed",  type=int, default=CONFIG["shared"]["seed"])

    parser.add_argument("--algo",  type=str, default="all",
                        choices=["sa", "sho", "woa", "all"])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # sync CLI -> CONFIG.shared
    GRID = tuple(int(x) for x in args.grid.split(","))
    OVLP = tuple(int(x) for x in args.ovlp.split(","))

    CONFIG["shared"].update({
        "N_evals": args.evals,
        "grid": GRID,
        "overlap_grid": OVLP,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "seed": args.seed,
        "verbose": args.verbose or CONFIG["shared"]["verbose"],
    })

    # Load chip
    chip = load_ucla_benchmark(
        nodes_file=args.nodes,
        nets_file=args.nets,
        pl_file=args.pl,
        chip_width=None,
        chip_height=None,
    )

    # เลือก algorithms
    algos = ["sa", "sho", "woa"] if args.algo == "all" else [args.algo]

    # Run & collect results
    results = {}
    for a in algos:
        print("=" * 60, f"\n>>> Running {a.upper()}  (N_evals={CONFIG['shared']['N_evals']})")
        if a == "sa":
            res = run_sa(chip, CONFIG["shared"], CONFIG["sa"], CONFIG["shared"]["N_evals"])
        elif a == "sho":
            res = run_sho(chip, CONFIG["shared"], CONFIG["sho"], CONFIG["shared"]["N_evals"])
        elif a == "woa":
            res = run_woa(chip, CONFIG["shared"], CONFIG["woa"], CONFIG["shared"]["N_evals"])
        else:
            raise ValueError(f"Unknown algo: {a}")
        results[a] = res

    # Summary table
    hdr = ["ALGO", "BestCost", "HPWL", "AvgCong", "Overflow", "Overlap", "Time(s)"]
    print("\n=== COMPARISON (equal-evaluations) ===")
    print("{:>6}  {:>10}  {:>12}  {:>12}  {:>8}  {:>8}  {:>8}".format(*hdr))

    for a in algos:
        r = results[a]
        print("{:>6}  {:>10.6f}  {:>12.0f}  {:>12.3e}  {:>8.6f}  {:>8.6f}  {:>8.2f}".format(
            a.upper(),
            r["best_cost"],
            r["hpwl"],
            r["avg_congestion"],
            r["overflow_ratio"] if r["overflow_ratio"] is not None else float("nan"),
            r["overlap_ratio"],
            r["execution_time"],
        ))


if __name__ == "__main__":
    main()
