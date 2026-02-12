"""
Genetic Algorithm optimizer for VLSI placement (GA)

Place this file at: src/optimizer_ga.py

Provides:
    genetic_optimizer(start_chip, pop_size=40, generations=200, ...)

Return: dict with keys similar to SHO:
    'best_cost','hpwl','hpwl_normalized','max_congestion','avg_congestion',
    'congestion_penalty','overflow_ratio','overlap_ratio','execution_time','generations'

The implementation uses:
- relative imports: .evaluator.calculate_cost_function, .chip.Chip
- simple tournament selection, uniform crossover, mutation (perturb + swap), elitism
- grid-based approximate overlap to compute overlap_ratio

Note: This is a straightforward, readable GA suited for experimentation and comparison with SHO.
"""

import time
import random
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from ..chip import Chip
from ..evaluator import calculate_cost_function


def copy_chip_simple(ch: Chip) -> Chip:
    new = Chip(ch.width, ch.height)
    for name, m in ch.modules.items():
        new.add_module(name, m.width, m.height, is_fixed=m.is_fixed)
        new.modules[name].set_position(m.x, m.y)
    for net in ch.nets:
        new.add_net(net.name, list(net.modules))
    return new


def _approx_overlap_ratio(chip: Chip, grid_bins: Tuple[int, int] = (200, 200)) -> float:
    # same approach as in optimizer_sho: approximate overlap by gridding module area
    rows, cols = grid_bins
    chip_area = max((chip.width * chip.height), 1.0)

    cell_w = chip.width / cols
    cell_h = chip.height / rows
    cell_area = cell_w * cell_h

    grid = np.zeros((rows, cols), dtype=np.float64)
    for mod in chip.get_all_modules():
        if mod.width <= 0 or mod.height <= 0:
            continue
        min_c = int(max(0, np.floor(mod.x / cell_w)))
        max_c = int(min(cols - 1, np.floor((mod.x + mod.width) / cell_w)))
        min_r = int(max(0, np.floor(mod.y / cell_h)))
        max_r = int(min(rows - 1, np.floor((mod.y + mod.height) / cell_h)))
        if max_c < min_c or max_r < min_r:
            continue
        num_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
        if num_cells <= 0:
            continue
        area_per_cell = (mod.width * mod.height) / num_cells
        grid[min_r:max_r + 1, min_c:max_c + 1] += area_per_cell

    overlap_per_cell = np.maximum(0.0, grid - cell_area)
    overlap_area_est = float(np.sum(overlap_per_cell))
    overlap_ratio = overlap_area_est / chip_area
    return overlap_ratio


def _evaluate_chip(ch: Chip, grid_size: Tuple[int, int] = (20, 20), alpha=0.5, beta=0.5) -> Dict[str, Any]:
    return calculate_cost_function(ch, grid_size=grid_size, alpha=alpha, beta=beta)


def _tournament_select(pop_scores: List[float], k: int, rng: random.Random) -> int:
    # return index of winner
    idxs = rng.sample(range(len(pop_scores)), min(k, len(pop_scores)))
    best = min(idxs, key=lambda i: pop_scores[i])
    return best


def _crossover_uniform(parent_a: Chip, parent_b: Chip, rng: random.Random) -> Chip:
    # offspring inherits per-module position either from A, from B, or average
    child = copy_chip_simple(parent_a)
    for m in child.get_movable_modules():
        if m.name not in parent_b.modules:
            continue
        if rng.random() < 0.45:
            # copy from A (already set)
            continue
        elif rng.random() < 0.9:
            # copy from B
            bx, by = parent_b.modules[m.name].get_position()
            child.modules[m.name].set_position(bx, by)
        else:
            # average
            ax, ay = parent_a.modules[m.name].get_position()
            bx, by = parent_b.modules[m.name].get_position()
            child.modules[m.name].set_position((ax + bx) / 2.0, (ay + by) / 2.0)
    return child


def _mutate_chip(chip: Chip, mutation_rate: float, perturb_scale: float, swap_prob: float, rng: random.Random):
    # position perturbation
    for m in chip.get_movable_modules():
        if rng.random() < mutation_rate:
            dx = rng.uniform(-perturb_scale, perturb_scale) * chip.width
            dy = rng.uniform(-perturb_scale, perturb_scale) * chip.height
            nx = max(0.0, min(chip.width - m.width, m.x + dx))
            ny = max(0.0, min(chip.height - m.height, m.y + dy))
            chip.modules[m.name].set_position(nx, ny)

    # random swaps
    movable = chip.get_movable_modules()
    n = len(movable)
    if n >= 2 and rng.random() < swap_prob:
        # do a few random swaps
        swaps = max(1, n // 100)  # scale with problem size
        for _ in range(swaps):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b:
                continue
            ma, mb = movable[a], movable[b]
            ax, ay = ma.get_position(); bx, by = mb.get_position()
            chip.modules[ma.name].set_position(bx, by)
            chip.modules[mb.name].set_position(ax, ay)


def genetic_optimizer(
    start_chip: Chip,
    pop_size: int = 40,
    generations: int = 200,
    elitism: int = 2,
    tournament_k: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.02,
    perturb_scale: float = 0.01,  # fraction of chip dim
    swap_prob: float = 0.2,
    grid_size: Tuple[int, int] = (20, 20),
    overlap_grid: Tuple[int, int] = (200, 200),
    alpha: float = 0.5,
    beta: float = 0.5,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a simple GA for placement optimization.

    The function updates start_chip positions to the best found layout.
    Returns a dict with metrics similar to SHO's result.
    """
    rng = random.Random(seed)

    # initialize population
    population: List[Chip] = []
    for i in range(pop_size):
        c = copy_chip_simple(start_chip)
        # small perturbation
        for m in c.get_movable_modules():
            if rng.random() < 0.3:
                dx = rng.uniform(-perturb_scale, perturb_scale) * c.width
                dy = rng.uniform(-perturb_scale, perturb_scale) * c.height
                nx = max(0.0, min(c.width - m.width, m.x + dx))
                ny = max(0.0, min(c.height - m.height, m.y + dy))
                c.modules[m.name].set_position(nx, ny)
        population.append(c)

    # evaluate
    pop_scores: List[float] = []
    pop_meta: List[Dict[str, Any]] = []
    for c in population:
        meta = _evaluate_chip(c, grid_size=grid_size, alpha=alpha, beta=beta)
        pop_meta.append(meta)
        pop_scores.append(meta['total_cost'])

    best_idx = int(np.argmin(pop_scores))
    best_chip = copy_chip_simple(population[best_idx])
    best_meta = pop_meta[best_idx]
    best_cost = pop_scores[best_idx]

    start_time = time.time()
    if verbose:
        print(f"GA: pop={pop_size}, gens={generations}, seed={seed}")

    for g in range(1, generations + 1):
        # create new generation
        new_pop: List[Chip] = []
        new_scores: List[float] = []
        new_meta: List[Dict[str, Any]] = []

        # elitism: keep top-k
        order = sorted(range(len(population)), key=lambda i: pop_scores[i])
        elites = [population[i] for i in order[:elitism]]
        for e in elites:
            new_pop.append(copy_chip_simple(e))

        # produce offspring until population full
        while len(new_pop) < pop_size:
            # selection
            a_idx = _tournament_select(pop_scores, tournament_k, rng)
            b_idx = _tournament_select(pop_scores, tournament_k, rng)
            parent_a = population[a_idx]
            parent_b = population[b_idx]

            if rng.random() < crossover_rate:
                child = _crossover_uniform(parent_a, parent_b, rng)
            else:
                child = copy_chip_simple(parent_a)

            # mutation
            _mutate_chip(child, mutation_rate, perturb_scale, swap_prob, rng)

            # evaluate child
            meta_c = _evaluate_chip(child, grid_size=grid_size, alpha=alpha, beta=beta)
            score_c = meta_c['total_cost']

            new_pop.append(child)
            new_scores.append(score_c)
            new_meta.append(meta_c)

            # track best
            if score_c < best_cost:
                best_cost = score_c
                best_chip = copy_chip_simple(child)
                best_meta = meta_c

        # if new_scores shorter than expected (due to elites), re-evaluate those elites
        if len(new_scores) < len(new_pop):
            # fill missing meta/score entries for elites
            fill_count = len(new_pop) - len(new_scores)
            for i in range(fill_count):
                e_meta = _evaluate_chip(new_pop[i], grid_size=grid_size, alpha=alpha, beta=beta)
                new_meta.insert(i, e_meta)
                new_scores.insert(i, e_meta['total_cost'])

        population = new_pop
        pop_scores = new_scores
        pop_meta = new_meta

        if verbose and (g % max(1, generations // 10) == 0 or g <= 5):
            print(f"Generation {g}/{generations} | Best Cost: {best_cost:.6f}")

    end_time = time.time()
    exec_time = end_time - start_time

    # copy best back into start_chip
    for name, mod in best_chip.modules.items():
        if name in start_chip.modules and not start_chip.modules[name].is_fixed:
            start_chip.modules[name].set_position(mod.x, mod.y)

    final_meta = _evaluate_chip(best_chip, grid_size=grid_size, alpha=alpha, beta=beta)
    overlap_ratio = _approx_overlap_ratio(best_chip, grid_bins=overlap_grid)

    result: Dict[str, Any] = {
        'best_cost': best_cost,
        'hpwl': final_meta.get('hpwl', None),
        'hpwl_normalized': final_meta.get('hpwl_normalized', None),
        'max_congestion': final_meta.get('max_congestion', None),
        'avg_congestion': final_meta.get('avg_congestion', None),
        'congestion_penalty': final_meta.get('congestion_penalty', None),
        'congestion_normalized': final_meta.get('congestion_normalized', None),
        'overflow_ratio': final_meta.get('overflow_ratio', None),
        'overlap_ratio': overlap_ratio,
        'execution_time': exec_time,
        'generations': generations,
    }

    # if verbose:
    #     print("\n==== GA RESULT ====")
    #     print(f"Best Cost      : {result['best_cost']}")
    #     print(f"HPWL           : {result['hpwl']}")
    #     print(f"Max Congestion : {result['max_congestion']}")
    #     print(f"Avg Congestion : {result['avg_congestion']}")
    #     print(f"Overflow Ratio : {result['overflow_ratio']}")
    #     print(f"Overlap Ratio  : {result['overlap_ratio']}")
    #     print(f"Execution Time : {result['execution_time']:.2f} sec")

    return result


# Quick demo when run directly
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from src.parser import load_ucla_benchmark as load_chip

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "ispd2005_benchmarks"))
    parser.add_argument("--nodes", type=str, default="bigblue1.inf.nodes")
    parser.add_argument("--nets", type=str, default="bigblue1.nets")
    parser.add_argument("--pl", type=str, default="bigblue1.pl")
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=10)
    args = parser.parse_args()

    base = Path(args.data_dir)
    nodes = base / args.nodes
    nets = base / args.nets
    pl = base / args.pl

    chip = load_chip(nodes_file=nodes, nets_file=nets, pl_file=pl, chip_width=None, chip_height=None)
    res = genetic_optimizer(chip, pop_size=args.pop, generations=args.gens, verbose=True)
    print(res)
