# main.py
from pathlib import Path
from copy import deepcopy
from src.parser import load_ucla_benchmark
from src.evaluator import calculate_HPWL, calculate_cost_function
from src.algorithms.SHO import spotted_hyena_optimizer
from src.algorithms.SA import simulated_annealing

BASE = Path(__file__).resolve().parent / "data" / "ispd2005_benchmarks"

nodes = BASE / "bigblue1.inf.nodes"
nets  = BASE / "bigblue1.nets"
pl    = BASE / "bigblue1.pl"

chip = load_ucla_benchmark(
    nodes_file=nodes,
    nets_file=nets,
    pl_file=pl,
    chip_width=None,
    chip_height=None,
)

print("Total HPWL =", calculate_HPWL(chip))

# --- พารามิเตอร์ “ชุดเดียวกัน” ทั้ง SHO และ verify ---
GRID = (64, 64)
OVLP = (128, 128)
ALPHA, BETA, GAMMA = 0.5, 1e-3, 0.1
SEED = 42

# สร้างชิปแยกสำหรับแต่ละอัลกอริทึม (กันเขียนทับกัน)
chip_sa  = deepcopy(chip)
chip_sho = deepcopy(chip)

# --- SA ---
res_sa = simulated_annealing(
    chip_sa,
    iters=200,                 # เพิ่มงบให้ anneal จริง
    grid_size=GRID, overlap_grid=OVLP,
    alpha=ALPHA, beta=BETA, gamma=GAMMA,
    t0=1.0,                      # บังคับ T0 ให้ใหญ่พอ (หรือปล่อย None ก็ได้)
    t_min=1e-6, cooling=0.99,
    move_disp_prob=0.7,
    disp_scale_init=0.02, disp_scale_final=0.002,
    seed=SEED, verbose=True
)
metrics_sa = calculate_cost_function(chip_sa, grid_size=GRID, alpha=ALPHA, beta=BETA)
print("\n=== VERIFY FINAL SA ===")
print("Final HPWL :", metrics_sa["hpwl"])
print("Final Cost :", ALPHA*(metrics_sa["hpwl"]/max(res_sa['hpwl'],1e-12))
                  + BETA*(metrics_sa["avg_congestion"]/max(res_sa['avg_congestion'],1e-12))
                  + GAMMA*res_sa["overlap_ratio"])

# --- SHO ---
res_sho = spotted_hyena_optimizer(
    chip_sho,
    pop_size=20, iters=10,     # งบโดยประมาณ ~ เทียบ N eval ของ SA (20k / pop_size)
    grid_size=GRID, overlap_grid=OVLP,
    alpha=ALPHA, beta=BETA, gamma=GAMMA,
    local_prob=0.1, local_swap_k=6,
    seed=SEED, verbose=True
)
metrics_sho = calculate_cost_function(chip_sho, grid_size=GRID, alpha=ALPHA, beta=BETA)
print("\n=== VERIFY FINAL SHO ===")
print("Final HPWL :", metrics_sho["hpwl"])
print("Final Cost :", ALPHA*(metrics_sho["hpwl"]/max(res_sho['hpwl'],1e-12))
                  + BETA*(metrics_sho["avg_congestion"]/max(res_sho['avg_congestion'],1e-12))
                  + GAMMA*res_sho["overlap_ratio"])