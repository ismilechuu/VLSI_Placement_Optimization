import time
import random
from copy import deepcopy
from typing import Tuple, Dict, Any, List, Optional

import numpy as np

# relative imports into package
from src.evaluator import calculate_cost_function, calculate_congestion
from src.chip import Chip


def copy_chip_simple(ch: Chip) -> Chip:
    """Copy แบบประหยัดแรม: คัดลอกเฉพาะ modules/positions และ 'แชร์' nets ร่วมกัน"""
    new = Chip(ch.width, ch.height)
    # แชร์ nets (รายการ Net objects) ร่วมกัน เพราะเราไม่แก้ไข nets อยู่แล้ว
    new.nets = ch.nets

    # คัดลอก modules + ตำแหน่ง (ต้องแยก เพราะตำแหน่งของแต่ละ candidate ต่างกัน)
    for name, m in ch.modules.items():
        new.add_module(name, m.width, m.height, is_fixed=m.is_fixed)
        new.modules[name].set_position(m.x, m.y)

    return new


def _evaluate_chip(ch: Chip, grid_size: Tuple[int, int] = (20, 20), alpha=0.5, beta=0.5) -> Dict[str, Any]:
    """Wrap evaluate: calculate_cost_function returns normalized metrics etc."""
    res = calculate_cost_function(ch, grid_size=grid_size, alpha=alpha, beta=beta)
    # add hpwl raw and normalized etc are already present in calculate_cost_function result
    return res


def _approx_overlap_ratio(ch: Chip, grid_bins: Tuple[int, int] = (200, 200)) -> float:
    """
    Approximate overlap ratio by gridding the chip and accumulating module area per cell.
    overlap_area_estimate = sum(max(0, area_sum - cell_area) over cells)
    overlap_ratio = overlap_area_estimate / chip_area

    This is an approximation (faster than pairwise)
    """
    rows, cols = grid_bins
    chip_area = max(chip_area_local := (ch.width * ch.height), 1.0)

    cell_w = ch.width / cols
    cell_h = ch.height / rows
    cell_area = cell_w * cell_h

    grid = np.zeros((rows, cols), dtype=np.float64)

    # For each module, fill the grid cells it covers with the module area fraction
    for mod in ch.get_all_modules():
        # skip zero-size modules
        if mod.width <= 0 or mod.height <= 0:
            continue
        # bounding box in grid coords
        min_c = int(max(0, np.floor(mod.x / cell_w)))
        max_c = int(min(cols - 1, np.floor((mod.x + mod.width) / cell_w)))
        min_r = int(max(0, np.floor(mod.y / cell_h)))
        max_r = int(min(rows - 1, np.floor((mod.y + mod.height) / cell_h)))
        if max_c < min_c or max_r < min_r:
            continue
        # add module area equally to covered cells (approx)
        num_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
        if num_cells <= 0:
            continue
        area_per_cell = (mod.width * mod.height) / num_cells
        grid[min_r:max_r + 1, min_c:max_c + 1] += area_per_cell

    # estimate overlap: where grid cell sum > cell_area
    overlap_per_cell = np.maximum(0.0, grid - cell_area)
    overlap_area_est = float(np.sum(overlap_per_cell))

    overlap_ratio = overlap_area_est / chip_area
    return overlap_ratio


def _grid_overlap_ratio(chip: Chip, grid_bins: Tuple[int, int] = (200, 200)) -> float:
    """
    Compatibility layer: uses grid approximation to compute overlap ratio.
    Keep separate in case we want to change algorithm later.
    """
    return _approx_overlap_ratio(chip, grid_bins)


def _score_candidate(
    chip: Chip,
    grid_size: Tuple[int, int],
    overlap_grid: Tuple[int, int],
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[float, Dict[str, Any], float]:
    """
    Evaluate chip and return (score, meta, overlap_ratio_approx).
    Includes overlap penalty only when gamma > 0 to avoid extra work otherwise.
    """
    meta = _evaluate_chip(chip, grid_size=grid_size, alpha=alpha, beta=beta)
    overlap_ratio = _grid_overlap_ratio(chip, grid_bins=overlap_grid) if gamma != 0.0 else 0.0
    total_cost = meta['total_cost'] + gamma * overlap_ratio
    return total_cost, meta, overlap_ratio


def spotted_hyena_optimizer(
    start_chip: Chip,
    pop_size: int = 30,
    iters: int = 200,
    grid_size: Tuple[int, int] = (20, 20),        # congestion grid for evaluator
    overlap_grid: Tuple[int, int] = (200, 200),  # grid to approximate overlap
    alpha: float = 0.5,                          # cost weight for hpwl
    beta: float = 0.5,                           # cost weight for congestion
    gamma: float = 0.0,                          # cost weight for overlap penalty
    local_prob: float = 0.2,                     # probability to do local swap per candidate
    local_swap_k: int = 10,                      # how many modules to swap when local refine
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run SHO on a Chip object.
    Returns dict of metrics and ensures chip positions are updated to best found.
    """
    rng = random.Random(seed)

    # Create initial population by perturbing start_chip
    population: List[Chip] = []
    for i in range(pop_size):
        c = copy_chip_simple(start_chip)
        # small perturbation for movable modules
        for m in c.get_movable_modules():
            if rng.random() < 0.25:
                dx = rng.uniform(-0.01, 0.01) * c.width
                dy = rng.uniform(-0.01, 0.01) * c.height
                new_x = max(0.0, min(c.width - m.width, m.x + dx))
                new_y = max(0.0, min(c.height - m.height, m.y + dy))
                c.modules[m.name].set_position(new_x, new_y)
        population.append(c)

    # Evaluate initial population
    pop_scores: List[float] = []
    pop_meta: List[Dict[str, Any]] = []
    for c in population:
        score, meta, _ = _score_candidate(
            c,
            grid_size=grid_size,
            overlap_grid=overlap_grid,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        pop_meta.append(meta)
        pop_scores.append(score)

    best_idx = int(np.argmin(pop_scores))
    best_chip = copy_chip_simple(population[best_idx])
    best_meta = pop_meta[best_idx]
    best_cost = pop_scores[best_idx]

    start_time = time.time()
    if verbose:
        print(f"SHO: pop_size={pop_size}, iters={iters}, seed={seed}")

    # Main loop
    for t in range(1, iters + 1):
        # select leaders alpha, beta, delta by sorted cost
        order = sorted(range(len(population)), key=lambda i: pop_scores[i])
        alpha_chip = population[order[0]]
        beta_chip = population[order[1]] if len(population) > 1 else alpha_chip
        delta_chip = population[order[2]] if len(population) > 2 else beta_chip

        new_population: List[Chip] = []
        new_scores: List[float] = []
        new_meta: List[Dict[str, Any]] = []

        a_coef = 2.0 * (1.0 - t / iters)  # linearly decreasing (like many swarm algos)

        for i, ch in enumerate(population):
            # create copy to modify
            new_ch = copy_chip_simple(ch)

            # Update each movable module position guided by leaders
            for m in new_ch.get_movable_modules():
                # get corresponding leaders' positions for this module (if exist)
                # Some nets/modules may not be present in leader chips due to fixed etc.
                try:
                    la = alpha_chip.modules[m.name]
                    lb = beta_chip.modules[m.name]
                    ld = delta_chip.modules[m.name]
                except Exception:
                    # if module absent (shouldn't happen), skip
                    continue

                # positions of leaders
                lax, lay = la.get_position()
                lbx, lby = lb.get_position()
                ldx, ldy = ld.get_position()

                # SHO-inspired update (mix of leaders with random coefficients)
                r1, r2 = rng.random(), rng.random()
                A1 = 2.0 * r1 * a_coef - a_coef
                C1 = 2.0 * r2
                D_alpha_x = abs(C1 * lax - m.x)
                D_alpha_y = abs(C1 * lay - m.y)
                new_x1 = lax - A1 * D_alpha_x
                new_y1 = lay - A1 * D_alpha_y

                # beta influence
                r3, r4 = rng.random(), rng.random()
                A2 = 2.0 * r3 * a_coef - a_coef
                C2 = 2.0 * r4
                D_beta_x = abs(C2 * lbx - m.x)
                D_beta_y = abs(C2 * lby - m.y)
                new_x2 = lbx - A2 * D_beta_x
                new_y2 = lby - A2 * D_beta_y

                # delta influence
                r5, r6 = rng.random(), rng.random()
                A3 = 2.0 * r5 * a_coef - a_coef
                C3 = 2.0 * r6
                D_delta_x = abs(C3 * ldx - m.x)
                D_delta_y = abs(C3 * ldy - m.y)
                new_x3 = ldx - A3 * D_delta_x
                new_y3 = ldy - A3 * D_delta_y

                # combine influences (simple average)
                candidate_x = (new_x1 + new_x2 + new_x3) / 3.0
                candidate_y = (new_y1 + new_y2 + new_y3) / 3.0

                # add small random jitter to avoid stagnation
                jitter_scale = 0.002
                candidate_x += rng.uniform(-jitter_scale, jitter_scale) * new_ch.width
                candidate_y += rng.uniform(-jitter_scale, jitter_scale) * new_ch.height

                # clip within chip boundaries and ensure module stays within bounds
                candidate_x = max(0.0, min(new_ch.width - m.width, candidate_x))
                candidate_y = max(0.0, min(new_ch.height - m.height, candidate_y))

                new_ch.modules[m.name].set_position(candidate_x, candidate_y)

            # local refinement (random swaps)
            if rng.random() < local_prob:
                movable = new_ch.get_movable_modules()
                if len(movable) >= 2:
                    # shuffle and swap coordinates for a small number of pairs
                    rng.shuffle(movable)
                    k = min(local_swap_k, max(2, len(movable)//50))
                    for j in range(0, min(len(movable)-1, k), 2):
                        a = movable[j]; b = movable[j+1]
                        xa, ya = a.get_position(); xb, yb = b.get_position()
                        new_ch.modules[a.name].set_position(xb, yb)
                        new_ch.modules[b.name].set_position(xa, ya)

            # evaluate new candidate
            score_new, meta_new, _ = _score_candidate(
                new_ch,
                grid_size=grid_size,
                overlap_grid=overlap_grid,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            new_population.append(new_ch)
            new_scores.append(score_new)
            new_meta.append(meta_new)

            # update global best if improved
            if score_new < best_cost:
                best_cost = score_new
                best_chip = copy_chip_simple(new_ch)
                best_meta = meta_new

        # replace population
        population = new_population
        pop_scores = new_scores
        pop_meta = new_meta

        if verbose:
            print(f"Iteration {t}/{iters} | Best Cost: {best_cost:.6f}")

    end_time = time.time()
    exec_time = end_time - start_time

    # ensure best_chip positions are written into start_chip (so downstream evaluates the best)
    # copy best positions back into start_chip
    for name, mod in best_chip.modules.items():
        if name in start_chip.modules and not start_chip.modules[name].is_fixed:
            start_chip.modules[name].set_position(mod.x, mod.y)

    # compute final metrics for the best chip
    final_meta = _evaluate_chip(best_chip, grid_size=grid_size, alpha=alpha, beta=beta)

    # overlap ratio (approx)
    overlap_ratio = _grid_overlap_ratio(best_chip, grid_bins=overlap_grid)

    result: Dict[str, Any] = {
        'best_cost': best_cost,
        'hpwl': final_meta.get('hpwl', None),
        'hpwl_normalized': final_meta.get('hpwl_normalized', None),
        'congestion_penalty': final_meta.get('congestion_penalty', None),
        'congestion_normalized': final_meta.get('congestion_normalized', None),
        'max_congestion': final_meta.get('max_congestion', None),
        'avg_congestion': final_meta.get('avg_congestion', None),
        'overflow_ratio': final_meta.get('overflow_ratio', None),
        'overlap_ratio': overlap_ratio,
        'execution_time': exec_time,
        'iterations': iters,
    }

    if verbose:
        print("\n==== SHO RESULT ====")
        print(f"Best Cost      : {result['best_cost']}")
        print(f"HPWL           : {result['hpwl']}")
        print(f"Max Congestion : {result['max_congestion']}")
        print(f"Avg Congestion : {result['avg_congestion']}")
        print(f"Overflow Ratio : {result['overflow_ratio']}")
        print(f"Overlap Ratio  : {result['overlap_ratio']}")
        print(f"Execution Time : {result['execution_time']:.2f} sec")

    return result


# If run as script for quick smoke test (requires parser & data)
if __name__ == "__main__":
    # quick demo when running this file directly (not typical in package)
    import argparse
    from pathlib import Path
    from src.parser import load_ucla_benchmark as load_chip

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "ispd2005_benchmarks"))
    parser.add_argument("--nodes", type=str, default="bigblue1.inf.nodes")
    parser.add_argument("--nets", type=str, default="bigblue1.nets")
    parser.add_argument("--pl", type=str, default="bigblue1.pl")
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    base = Path(args.data_dir)
    nodes = base / args.nodes
    nets = base / args.nets
    pl = base / args.pl

    chip = load_chip(nodes_file=nodes, nets_file=nets, pl_file=pl, chip_width=None, chip_height=None)
    res = spotted_hyena_optimizer(chip, pop_size=args.pop, iters=args.iters, verbose=True)
    print(res)
