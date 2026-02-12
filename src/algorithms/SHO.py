# ============================
#  SHO.py — Spotted Hyena Optimizer (FIXED / TUNED VERSION)
# ============================
import time
import random
import math
from copy import deepcopy

from src.evaluator import calculate_cost_function


def copy_chip_simple(chip):
    """
    Clone chip ทั้งก้อนแบบง่าย ๆ
    """
    return deepcopy(chip)


def approx_overlap_ratio(chip, grid_bins):
    """
    Lightweight overlap approximation (bin-occupancy).
    """
    W = chip.width
    H = chip.height
    gx, gy = grid_bins

    bin_w = W / gx
    bin_h = H / gy

    bins = [[0.0 for _ in range(gy)] for _ in range(gx)]

    total_area = 0.0
    overlap_area = 0.0

    for m in chip.get_movable_modules():
        mw = max(m.width, 1.0)
        mh = max(m.height, 1.0)
        area = mw * mh
        total_area += area

        # range of bins
        x0 = max(0, int(m.x // bin_w))
        x1 = min(gx - 1, int((m.x + m.width) // bin_w))
        y0 = max(0, int(m.y // bin_h))
        y1 = min(gy - 1, int((m.y + m.height) // bin_h))

        if x1 < x0 or y1 < y0:
            continue

        n_cells = (x1 - x0 + 1) * (y1 - y0 + 1)
        if n_cells <= 0:
            continue
        add_val = area / n_cells

        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                prev = bins[ix][iy]
                newv = prev + add_val
                bins[ix][iy] = newv

    if total_area <= 0:
        return 0.0

    cell_area = bin_w * bin_h
    for ix in range(gx):
        for iy in range(gy):
            if bins[ix][iy] > cell_area:
                overlap_area += (bins[ix][iy] - cell_area)

    return overlap_area / (chip.width * chip.height + 1e-9)


def spotted_hyena_optimizer(
    start_chip,
    pop_size=4,
    iters=10,
    grid_size=(64, 64),
    overlap_grid=(128, 128),
    alpha=0.5,
    beta=1e-3,
    gamma=0.1,
    local_prob=0.10,
    local_swap_k=6,
    seed=42,
    verbose=True,
    movement_scale=0.05,  # limit movement per iteration (fraction of chip size)
):
    """
    Tuned SHO สำหรับ VLSI placement:
    - ใช้ cost เดียวกับ SA (HPWL + Congestion) + gamma * overlap
    - ใช้ equal-eval ผ่าน pop_size * iters ประมาณ N_evals
    - มี local refinement + SA-like acceptance
    """
    rng = random.Random(seed)

    # -------------------------
    # Create population
    # -------------------------
    population = [copy_chip_simple(start_chip) for _ in range(pop_size)]
    scores = []

    def eval_chip(chip):
        meta = calculate_cost_function(
            chip, grid_size=grid_size, alpha=alpha, beta=beta
        )
        ov = approx_overlap_ratio(chip, overlap_grid)
        score = meta["total_cost"] + gamma * ov
        return score, meta, ov

    # evaluate initial population
    for ind in population:
        score, meta, ov = eval_chip(ind)
        scores.append((score, meta, ov))

    # sort by best
    population_scores = list(zip(population, scores))
    population_scores.sort(key=lambda x: x[1][0])  # based on total cost
    population = [ps[0] for ps in population_scores]
    scores = [ps[1] for ps in population_scores]

    best_chip = copy_chip_simple(population[0])
    best_score, best_meta, best_ovlp = scores[0]

    if verbose:
        print(f"SHO: pop_size={pop_size}, iters={iters}, seed={seed}")
        print(f"     movement_scale={movement_scale}")
        print(f"Initial best cost: {best_score:.6f}")

    t_start = time.time()

    # Cache chip dimensions
    W = float(start_chip.width)
    H = float(start_chip.height)
    max_move = movement_scale * min(W, H)

    # =============================
    # Main loop of SHO
    # =============================
    for it in range(1, iters + 1):
        # a: encircle / attack model coefficient (decays)
        a = 2 - 2 * (it / iters)

        new_population = []

        for i in range(pop_size):
            # 1) local refinement (random small moves + swaps)
            if rng.random() < local_prob:
                new_ind = copy_chip_simple(population[i])
                movable = new_ind.get_movable_modules()
                if len(movable) > 0:
                    k = min(local_swap_k, len(movable))
                    chosen = rng.sample(movable, k)

                    for m in chosen:
                        dx = rng.uniform(-0.5, 0.5) * max_move
                        dy = rng.uniform(-0.5, 0.5) * max_move
                        new_x = max(0.0, min(W - m.width, m.x + dx))
                        new_y = max(0.0, min(H - m.height, m.y + dy))
                        m.set_position(new_x, new_y)

                    # optional random pair swaps
                    if len(movable) >= 2:
                        for _ in range(max(1, k // 2)):
                            m1, m2 = rng.sample(movable, 2)
                            temp_x, temp_y = m1.x, m1.y
                            new_m1_x = max(0.0, min(m2.x, W - m1.width))
                            new_m1_y = max(0.0, min(m2.y, H - m1.height))
                            new_m2_x = max(0.0, min(temp_x, W - m2.width))
                            new_m2_y = max(0.0, min(temp_y, H - m2.height))
                            m1.set_position(new_m1_x, new_m1_y)
                            m2.set_position(new_m2_x, new_m2_y)

                new_population.append(new_ind)
                continue

            # 2) SHO main update — chase best
            best = population[0]
            ind = copy_chip_simple(population[i])
            nb = best  # already best individual

            for name, m_i in ind.modules.items():
                if m_i.is_fixed:
                    continue

                m_b = nb.modules.get(name, None)
                if m_b is None:
                    continue

                curr_x = m_i.x
                curr_y = m_i.y

                r1 = rng.random()
                r2 = rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2

                # SHO-like encircling
                D_x = abs(C * m_b.x - curr_x)
                D_y = abs(C * m_b.y - curr_y)
                new_x = m_b.x - A * D_x
                new_y = m_b.y - A * D_y

                # limit movement length
                dx = new_x - curr_x
                dy = new_y - curr_y
                move_dist = math.hypot(dx, dy)

                if move_dist > max_move and move_dist > 1e-9:
                    scale = max_move / move_dist
                    dx *= scale
                    dy *= scale
                    new_x = curr_x + dx
                    new_y = curr_y + dy

                new_x = max(0.0, min(new_x, W - m_i.width))
                new_y = max(0.0, min(new_y, H - m_i.height))
                m_i.set_position(new_x, new_y)

            new_population.append(ind)

        # evaluate new population + SA-like acceptance
        new_scores = []
        for ind in new_population:
            s, meta, ov = eval_chip(ind)
            new_scores.append((s, meta, ov))

        # Combine old & new and keep best pop_size (elitist)
        combined = list(zip(population, scores)) + list(zip(new_population, new_scores))
        combined.sort(key=lambda x: x[1][0])
        combined = combined[:pop_size]

        population = [c[0] for c in combined]
        scores = [c[1] for c in combined]

        # update global best
        if scores[0][0] < best_score:
            best_score, best_meta, best_ovlp = scores[0]
            best_chip = copy_chip_simple(population[0])

        if verbose:
            print(f"Iteration {it}/{iters} | SHO best cost: {best_score:.6f}")

    t_end = time.time()

    # write-back best layout ลง start_chip
    # for name, m_best in best_chip.modules.items():
    #     m_target = start_chip.get_module(name)
    #     if m_target is not None:
    #         m_target.set_position(m_best.x, m_best.y)

    return {
        "best_cost": best_score,
        "hpwl": best_meta["hpwl"],
        "max_congestion": best_meta["max_congestion"],
        "avg_congestion": best_meta["avg_congestion"],
        "overflow_ratio": best_meta["overflow_ratio"],
        "overlap_ratio": best_ovlp,
        "execution_time": t_end - t_start,
    }
