import time
import math
import random
from typing import Tuple, Dict, Any, Optional, List

import numpy as np

from src.chip import Chip
from src.evaluator import calculate_cost_function


def copy_chip_simple(ch: Chip) -> Chip:
    """
    Lightweight copy: duplicate modules & positions; share nets list.
    """
    new = Chip(ch.width, ch.height)
    new.nets = ch.nets
    for name, m in ch.modules.items():
        new.add_module(name, m.width, m.height, is_fixed=m.is_fixed)
        new.modules[name].set_position(m.x, m.y)
    return new


def _approx_overlap_ratio(ch: Chip, grid_bins: Tuple[int, int] = (200, 200)) -> float:
    """
    Approximate overlap ratio with a coarse grid (fast, like in SHO).
    overlap_area_estimate = sum(max(0, area_sum - cell_area) over cells) / chip_area
    """
    rows, cols = grid_bins
    chip_area = max(ch.width * ch.height, 1.0)

    cell_w = ch.width / cols
    cell_h = ch.height / rows
    cell_area = cell_w * cell_h

    grid = np.zeros((rows, cols), dtype=np.float64)

    for mod in ch.get_all_modules():
        if mod.width <= 0 or mod.height <= 0:
            continue
        min_c = int(max(0, math.floor(mod.x / cell_w)))
        max_c = int(min(cols - 1, math.floor((mod.x + mod.width) / cell_w)))
        min_r = int(max(0, math.floor(mod.y / cell_h)))
        max_r = int(min(rows - 1, math.floor((mod.y + mod.height) / cell_h)))
        if max_c < min_c or max_r < min_r:
            continue
        num_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
        if num_cells <= 0:
            continue
        area_per_cell = (mod.width * mod.height) / num_cells
        grid[min_r:max_r + 1, min_c:max_c + 1] += area_per_cell

    overlap = np.maximum(0.0, grid - cell_area)
    return float(np.sum(overlap)) / chip_area


def _score(
    ch: Chip,
    grid_size: Tuple[int, int],
    overlap_grid: Tuple[int, int],
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[float, Dict[str, Any], float]:
    meta = calculate_cost_function(ch, grid_size=grid_size, alpha=alpha, beta=beta)
    ov = _approx_overlap_ratio(ch, overlap_grid) if gamma != 0.0 else 0.0
    total = meta['total_cost'] + gamma * ov
    return total, meta, ov


def _random_displacement(rng: random.Random, ch: Chip, m, scale: float = 0.02):
    dx = rng.uniform(-scale, scale) * ch.width
    dy = rng.uniform(-scale, scale) * ch.height
    nx = max(0.0, min(ch.width - m.width, m.x + dx))
    ny = max(0.0, min(ch.height - m.height, m.y + dy))
    ch.modules[m.name].set_position(nx, ny)


def _swap_positions(ch: Chip, a, b):
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    ch.modules[a.name].set_position(bx, by)
    ch.modules[b.name].set_position(ax, ay)


def simulated_annealing(
    start_chip: Chip,
    iters: int = 20000,
    grid_size: Tuple[int, int] = (20, 20),
    overlap_grid: Tuple[int, int] = (200, 200),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,
    # annealing params
    t0: Optional[float] = None,           # if None -> auto init
    t_min: float = 1e-4,
    cooling: float = 0.98,                # geometric cooling factor
    move_disp_prob: float = 0.7,          # prob to do displacement; else swap
    disp_scale_init: float = 0.02,        # initial displacement scale
    disp_scale_final: float = 0.002,      # final displacement scale
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Basic SA with two move types: (1) random displacement, (2) position swap.
    """

    rng = random.Random(seed)

    current = copy_chip_simple(start_chip)
    # small random jitter to diversify start
    for m in current.get_movable_modules():
        if rng.random() < 0.25:
            _random_displacement(rng, current, m, scale=0.01)

    cur_cost, cur_meta, cur_ov = _score(current, grid_size, overlap_grid, alpha, beta, gamma)

    best = copy_chip_simple(current)
    best_cost, best_meta, best_ov = cur_cost, cur_meta, cur_ov

    # auto-initialize T0 if not provided (based on typical cost delta)
    if t0 is None:
        # probe a few random moves to estimate delta scale
        deltas = []
        probe = copy_chip_simple(current)
        movable = probe.get_movable_modules()
        for _ in range(32):
            if len(movable) >= 2 and rng.random() < 0.5:
                a, b = rng.sample(movable, 2)
                _swap_positions(probe, a, b)
            else:
                if movable:
                    _random_displacement(rng, probe, rng.choice(movable), scale=0.05)
            score_p, _, _ = _score(probe, grid_size, overlap_grid, alpha, beta, gamma)
            deltas.append(abs(score_p - cur_cost))
            # revert back to current
            probe = copy_chip_simple(current)
        est = (sum(deltas) / max(len(deltas), 1)) if deltas else 1.0
        t0 = max(est, 1e-6)

    T = float(t0)
    start_t = time.time()

    if verbose:
        print(f"SA: iters={iters}, seed={seed}, T0={T:.6g}, cool={cooling}")

    movable = current.get_movable_modules()

    for i in range(1, iters + 1):
        candidate = copy_chip_simple(current)

        # choose and apply a move
        if movable and (rng.random() < move_disp_prob or len(movable) < 2):
            # displacement; scale decays over time
            frac = i / max(iters, 1)
            scale = disp_scale_init * ((1 - frac) + frac * (disp_scale_final / max(disp_scale_init, 1e-12)))
            _random_displacement(rng, candidate, rng.choice(movable), scale=scale)
        else:
            a, b = rng.sample(movable, 2)
            _swap_positions(candidate, a, b)

        cand_cost, cand_meta, cand_ov = _score(candidate, grid_size, overlap_grid, alpha, beta, gamma)

        delta = cand_cost - cur_cost
        accept = delta <= 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12))

        if accept:
            current = candidate
            cur_cost, cur_meta, cur_ov = cand_cost, cand_meta, cand_ov

            if cur_cost < best_cost:
                best = copy_chip_simple(current)
                best_cost, best_meta, best_ov = cur_cost, cur_meta, cur_ov

        # cool
        T = max(T * cooling, t_min)

        if verbose and (i % max(1, iters // 10) == 0):
            print(f"Iter {i}/{iters} | T={T:.3g} | Best: {best_cost:.6f} | Cur: {cur_cost:.6f}")

    exec_time = time.time() - start_t

    # write best back to start_chip (like SHO) so downstream uses the best
    for name, mod in best.modules.items():
        if name in start_chip.modules and not start_chip.modules[name].is_fixed:
            start_chip.modules[name].set_position(mod.x, mod.y)

    result = {
        'best_cost': best_cost,
        'hpwl': best_meta.get('hpwl', None),
        'hpwl_normalized': best_meta.get('hpwl_normalized', None),
        'congestion_penalty': best_meta.get('congestion_penalty', None),
        'congestion_normalized': best_meta.get('congestion_normalized', None),
        'max_congestion': best_meta.get('max_congestion', None),
        'avg_congestion': best_meta.get('avg_congestion', None),
        'overflow_ratio': best_meta.get('overflow_ratio', None),
        'overlap_ratio': best_ov,
        'execution_time': exec_time,
        'iterations': iters,
    }

    if verbose:
        print("\n==== SA RESULT ====")
        print(f"Best Cost      : {result['best_cost']}")
        print(f"HPWL           : {result['hpwl']}")
        print(f"Max Congestion : {result['max_congestion']}")
        print(f"Avg Congestion : {result['avg_congestion']}")
        print(f"Overflow Ratio : {result['overflow_ratio']}")
        print(f"Overlap Ratio  : {result['overlap_ratio']}")
        print(f"Execution Time : {result['execution_time']:.2f} sec")

    return result

