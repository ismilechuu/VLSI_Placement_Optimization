# main.py — เปรียบเทียบ SA, SHO, WOA ด้วยค่ากลางชุดเดียว (equal-evaluations)
from pathlib import Path
from copy import deepcopy
from math import ceil
import argparse

from src.parser import load_ucla_benchmark
from src.algorithms.SA import simulated_annealing
from src.algorithms.SHO import spotted_hyena_optimizer
from src.algorithms.WOA import whale_optimizer
from src.utils.logger import ExperimentLogger


# -------------------------------
# 1) CONFIG (key-value)
# -------------------------------
CONFIG = {
    "shared": {
        "N_evals": 200,
        "grid": (32, 32),
        "overlap_grid": (64, 64),
        "alpha": 0.9,
        "beta": 1e-5,
        "gamma": 0.0003,
        "seed": 42,
        "verbose": True,
    },
    # SA: 1 evaluation ต่อ 1 iteration
    "sa": {
        "t0": None,           # ให้ SA auto-derive T0 จาก delta cost
        "t_min": 1e-6,
        "cooling": 0.985,
        "move_disp_prob": 0.85,
        "disp_scale_init": 0.05,
        "disp_scale_final": 0.003,
    },
    # SHO: ~pop_size evaluations ต่อ 1 iteration
    "sho": {
        "pop_size": 10,
        "local_prob": 0.3,
        "local_swap_k": 8,
        "movement_scale": 0.15,   
    },

    "woa": {
        "pop_size": 8,
        "module_sample": 800,
        "init_jitter": 0.002, 
        "movement_scale": 0.01, 
    },
    # เลือกว่าจะรันอะไรบ้าง
    "run": ["sa", "sho", "woa"],
}

# -------------------------------
# 2) Runners (equal-evals)
# -------------------------------
def run_sa(chip, shared, sa_cfg, N_evals):
    # ทำงานบนสำเนา chip เพื่อเก็บ layout ของ best
    work_chip = deepcopy(chip)
    res = simulated_annealing(
        work_chip,
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
    # แนบ chip ที่ถูก optimize แล้วเข้าไปในผลลัพธ์
    res["chip"] = work_chip
    return res, work_chip


def run_sho(chip, shared, sho_cfg, N_evals):
    pop = sho_cfg["pop_size"]
    iters = max(N_evals // pop, 1)

    work_chip = deepcopy(chip)

    res = spotted_hyena_optimizer(
        start_chip=chip,
        pop_size=pop,
        iters=iters,
        grid_size=shared["grid"],
        overlap_grid=shared["overlap_grid"],
        alpha=shared["alpha"],
        beta=shared["beta"],
        gamma=shared["gamma"],
        local_prob=sho_cfg["local_prob"],
        local_swap_k=sho_cfg["local_swap_k"],
        movement_scale=sho_cfg["movement_scale"],
        seed=shared["seed"],
        verbose=shared["verbose"],
    )
    res["chip"] = work_chip
    return res, work_chip


def run_woa(chip, shared, woa_cfg, N_evals):
    pop = woa_cfg["pop_size"]
    iters = max(N_evals // pop, 1)

    work_chip = deepcopy(chip)

    res = whale_optimizer(
        work_chip,
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
        movement_scale=woa_cfg["movement_scale"],
    )
    res["chip"] = work_chip
    return res, work_chip



# -------------------------------
# 3) main()
# -------------------------------
def main():
    BASE = Path(__file__).resolve().parent / "data" / "ispd2005_benchmarks"

    logger = ExperimentLogger(base_dir="results")

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

    # ชื่อ benchmark (เอาจากชื่อไฟล์ nodes)
    benchmark_name = Path(args.nodes).stem.split(".")[0]

    # เลือก algorithms
    algos = ["sa", "sho", "woa"] if args.algo == "all" else [args.algo]

    n_evals = CONFIG["shared"]["N_evals"]       

    # Run & collect results
    results = {}

    for a in algos:
        if a == "sa":
            res, chip_after = run_sa(chip, CONFIG["shared"], CONFIG["sa"], n_evals)
        elif a == "sho":
            res, chip_after = run_sho(chip, CONFIG["shared"], CONFIG["sho"], n_evals)
        elif a == "woa":
            res, chip_after = run_woa(chip, CONFIG["shared"], CONFIG["woa"], n_evals)

        results[a] = res

        # เซฟ layout จาก chip ที่ optimize แล้ว
        logger.save_layout(
            chip=chip_after,
            benchmark=benchmark_name,
            algo=a.upper(),
            tag=f"N{n_evals}",
        )


    # Summary table (หน้าจอ)
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

    # สร้างไฟล์ TXT summary + Markdown table สำหรับชุดรันนี้
    txt_path = logger.write_txt_summary(
        benchmark=benchmark_name,
        shared_cfg=CONFIG["shared"],
        n_evals=n_evals,
        results=results,
    )
    md_path = logger.write_markdown_comparison(
        benchmark=benchmark_name,
        shared_cfg=CONFIG["shared"],
        n_evals=n_evals,
        results=results,
    )
    print(f"\nSaved TXT summary to: {txt_path}")
    print(f"Saved Markdown comparison to: {md_path}")



if __name__ == "__main__":
    main()