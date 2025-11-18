import time
import math
import random
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from copy import deepcopy
from src.chip import Chip
from src.evaluator import calculate_cost_function


def copy_chip_simple(ch: Chip) -> Chip:
    """
    คัดลอก chip ทั้งก้อนแบบปลอดภัย:
    - modules
    - nets
    - ตำแหน่งทุกอย่าง
    ใช้ deepcopy ตรง ๆ (ชัดกว่าและไม่สับสน)
    """
    return deepcopy(ch)


def _approx_overlap_ratio(ch: Chip, grid_bins: Tuple[int, int]) -> float:
    """
    Approximate overlap ratio ด้วยการ gridding:

    overlap_area_estimate = sum(max(0, area_sum - cell_area) over all cells)
    overlap_ratio = overlap_area_estimate / chip_area
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
    overlap_area = float(np.sum(overlap))
    return overlap_area / chip_area


def _score(
    ch: Chip,
    grid_size: Tuple[int, int],
    overlap_grid: Tuple[int, int],
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[float, Dict[str, Any], float]:
    """
    คำนวณ cost เหมือน SA / SHO:
    total_cost (จาก calculate_cost_function) + gamma * overlap_ratio
    """
    meta = calculate_cost_function(ch, grid_size=grid_size, alpha=alpha, beta=beta)
    ov = _approx_overlap_ratio(ch, overlap_grid) if gamma != 0.0 else 0.0
    score = meta["total_cost"] + gamma * ov
    return score, meta, ov


def whale_optimizer(
    start_chip: Chip,
    pop_size: int = 10,
    iters: int = 200,
    grid_size: Tuple[int, int] = (20, 20),
    overlap_grid: Tuple[int, int] = (64, 64),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,
    movement_scale: float = 0.02,  # base movement (fraction of chip, จะลดลงเรื่อย ๆ ตาม t)
    seed: Optional[int] = None,
    verbose: bool = True,
    module_sample: int = 384,   # จำนวนโมดูลต่อ whale ต่อรอบ
    init_jitter: float = 0.003, # scale ของ jitter เริ่มต้น (fraction ของ chip)
) -> Dict[str, Any]:
    """
    Whale Optimization Algorithm (WOA) เวอร์ชัน tuned สำหรับ placement:

    - warm-start จาก layout เดิม
    - ขยับ subset ของ movable modules ต่อรอบ (module_sample)
    - ใช้ normalized coordinate [0,1] แล้ว clamp ระยะขยับต่อรอบ
    - ใช้ cost เดียวกับ SA/SHO (HPWL + Congestion + gamma*overlap)
    - ใช้ acceptance rule แบบ SA-ish (ยอมรับ solution แย่ลงได้บ้างช่วงต้น)
    """

    t_start = time.time()
    rng = random.Random(seed)

    pop_size = max(int(pop_size), 1)
    iters = max(int(iters), 1)
    module_sample = max(int(module_sample), 1)
    init_jitter = float(max(0.0, init_jitter))

    movable = start_chip.get_movable_modules()
    if not movable:
        base_score, base_meta, base_ov = _score(
            start_chip, grid_size, overlap_grid, alpha, beta, gamma
        )
        t_end = time.time()
        return {
            "best_cost": base_score,
            "hpwl": float(base_meta["hpwl"]),
            "max_congestion": float(base_meta["max_congestion"]),
            "avg_congestion": float(base_meta["avg_congestion"]),
            "overflow_ratio": float(base_meta["overflow_ratio"]),
            "overlap_ratio": base_ov,
            "execution_time": t_end - t_start,
        }

    W = float(start_chip.width)
    H = float(start_chip.height)

    movable_names = [m.name for m in movable]

    # -------- initial population (warm start + jitter) --------
    population: List[Chip] = []
    scores: List[float] = []
    metas: List[Dict[str, Any]] = []
    ov_list: List[float] = []

    for _ in range(pop_size):
        ch = copy_chip_simple(start_chip)

        # small jitter รอบ layout เดิม
        for name in movable_names:
            if rng.random() < 0.5:
                m = ch.modules[name]
                dx = rng.uniform(-init_jitter, init_jitter) * ch.width
                dy = rng.uniform(-init_jitter, init_jitter) * ch.height
                nx = max(0.0, min(ch.width - m.width, m.x + dx))
                ny = max(0.0, min(ch.height - m.height, m.y + dy))
                m.set_position(nx, ny)

        s, meta, ov = _score(ch, grid_size, overlap_grid, alpha, beta, gamma)
        population.append(ch)
        scores.append(s)
        metas.append(meta)
        ov_list.append(ov)

    best_idx = min(range(pop_size), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_meta = metas[best_idx]
    best_ov = ov_list[best_idx]
    best_chip = copy_chip_simple(population[best_idx])

    if verbose:
        print(
            f"WOA: pop_size={pop_size}, iters={iters}, seed={seed}\n"
            f"     initial best cost={best_score:.6f}"
        )

    # -------- main loop --------
    for t in range(1, iters + 1):
        # a ลดแบบ quadratic -> exploit ช่วงท้ายแรงขึ้น
        a = 1.2 * ((iters - t) / float(iters)) ** 2.0

        # temperature-like factor สำหรับ acceptance (สูงช่วงต้น, ต่ำช่วงท้าย)
        temp_factor = max(0.2, (iters - t + 1) / float(iters))  # 1.0 -> 0.2

        for i in range(pop_size):
            if i == best_idx:
                continue  # elitism

            ch = copy_chip_simple(population[i])

            movable_now = ch.get_movable_modules()
            if not movable_now:
                continue

            sample_k = min(len(movable_now), module_sample)
            sample = rng.sample(movable_now, sample_k)

            for m in sample:
                mx = m.x / W
                my = m.y / H

                # best module
                m_best = best_chip.get_module(m.name)
                if m_best is None:
                    continue
                bx = m_best.x / W
                by = m_best.y / H

                r1 = rng.random()
                r2 = rng.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                p = rng.random()
                l = 0.6 * rng.uniform(-1.0, 1.0)

                if p < 0.5:
                    # encircling / exploring
                    if abs(A) < 1.0:
                        # เข้าหา best
                        Dx = abs(C * bx - mx)
                        Dy = abs(C * by - my)
                        nx = bx - A * Dx
                        ny = by - A * Dy
                    else:
                        # explore รอบ ๆ whale อื่น
                        j = rng.randrange(pop_size)
                        if j == i:
                            j = (j + 1) % pop_size
                        m_ref = population[j].get_module(m.name)
                        if m_ref is None:
                            continue
                        rx = m_ref.x / W
                        ry = m_ref.y / H
                        Dx = abs(C * rx - mx)
                        Dy = abs(C * ry - my)
                        nx = rx - A * Dx
                        ny = ry - A * Dy
                else:
                    # spiral update รอบ best
                    dist = math.hypot(bx - mx, by - my)
                    b_sp = 1.0
                    nx = (
                        dist * math.exp(b_sp * l) * math.cos(2.0 * math.pi * l) + bx
                    )
                    ny = (
                        dist * math.exp(b_sp * l) * math.sin(2.0 * math.pi * l) + by
                    )

                # ----- limit step size (movement clamp) -----
                # ปรับ movement_scale ตามเวลา (แรงช่วงต้น, เบากช่วงท้าย)
                eff_move = movement_scale * temp_factor  # temp_factor อยู่ใน [0.2, 1.0]

                step_x = nx - mx
                step_y = ny - my
                limit = eff_move

                if step_x > limit:
                    step_x = limit
                elif step_x < -limit:
                    step_x = -limit

                if step_y > limit:
                    step_y = limit
                elif step_y < -limit:
                    step_y = -limit

                nx = mx + step_x
                ny = my + step_y

                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))

                x_new = min(max(nx * W, 0.0), W - m.width)
                y_new = min(max(ny * H, 0.0), H - m.height)
                m.set_position(x_new, y_new)

            new_score, new_meta, new_ov = _score(
                ch, grid_size, overlap_grid, alpha, beta, gamma
            )

            old_score = scores[i]
            accept = False
            if new_score < old_score:
                accept = True
            else:
                # ยอมรับบางส่วนด้วยโอกาสมากขึ้นช่วง iteration แรก ๆ
                delta = new_score - old_score
                if delta < 1e-9:
                    accept = True
                else:
                    denom = max(abs(old_score), 1e-6)
                    # temp_factor สูง -> ยอมรับง่ายช่วงต้น, ยากช่วงท้าย
                    prob = math.exp(-delta / (denom * temp_factor))
                    # base factor 0.1 แทน 0.01 ให้โอกาสขยับมากขึ้น
                    if rng.random() < 0.1 * prob:
                        accept = True

            if accept:
                population[i] = ch
                scores[i] = new_score
                metas[i] = new_meta
                ov_list[i] = new_ov

                if new_score < best_score:
                    best_score = new_score
                    best_meta = new_meta
                    best_ov = new_ov
                    best_chip = copy_chip_simple(ch)
                    best_idx = i

        if verbose:
            print(
                f"Iteration {t}/{iters} | WOA best cost: {best_score:.6f}"
            )

    t_end = time.time()

    # เขียน best layout กลับไปที่ start_chip
    for m_best in best_chip.get_all_modules():
        m_target = start_chip.get_module(m_best.name)
        if m_target is not None:
            m_target.set_position(m_best.x, m_best.y)

    return {
        "best_cost": float(best_score),
        "hpwl": float(best_meta["hpwl"]),
        "max_congestion": float(best_meta["max_congestion"]),
        "avg_congestion": float(best_meta["avg_congestion"]),
        "overflow_ratio": float(best_meta["overflow_ratio"]),
        "overlap_ratio": float(best_ov),
        "execution_time": t_end - t_start,
    }
