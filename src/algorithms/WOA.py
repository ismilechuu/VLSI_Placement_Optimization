import time
import math
import random
from typing import Tuple, Dict, Any, Optional, List

import numpy as np

from src.chip import Chip
from src.evaluator import calculate_cost_function


# -------------------------------
# utils (คล้าย SA / SHO)
# -------------------------------
def copy_chip_simple(ch: Chip) -> Chip:
    """
    Copy แบบประหยัด: คัดลอก modules + position, แชร์ nets
    """
    new = Chip(ch.width, ch.height)
    new.nets = ch.nets  # แชร์ได้เพราะเราไม่แก้ nets
    for name, m in ch.modules.items():
        new.add_module(name, m.width, m.height, is_fixed=m.is_fixed)
        new.modules[name].set_position(m.x, m.y)
    return new


def _approx_overlap_ratio(ch: Chip, grid_bins: Tuple[int, int] = (200, 200)) -> float:
    """
    Approximate overlap ratio ด้วยการ gridding (เหมือน SA/SHO)

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
    total = meta["total_cost"] + gamma * ov
    return total, meta, ov


def _jitter_small(rng: random.Random, ch: Chip, scale: float = 0.01):
    """
    warm-start: เขยิบตำแหน่งเดิมเล็กน้อย (แทนการสุ่มทั้งชิป)
    """
    for m in ch.get_movable_modules():
        nx = m.x + rng.uniform(-scale, scale) * ch.width
        ny = m.y + rng.uniform(-scale, scale) * ch.height
        nx = max(0.0, min(ch.width - m.width, nx))
        ny = max(0.0, min(ch.height - m.height, ny))
        ch.modules[m.name].set_position(nx, ny)


# -------------------------------
#  WOA main
# -------------------------------
def whale_optimizer(
    start_chip: Chip,
    pop_size: int = 2,                      # ลด pop เพื่อได้ iters สูงขึ้น
    iters: int = 200,
    grid_size: Tuple[int, int] = (20, 20),
    overlap_grid: Tuple[int, int] = (200, 200),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = True,
    module_sample: int = 256,               # ขยับเฉพาะ K modules ต่อ whale ต่อรอบ
    init_jitter: float = 0.01,              # เขยิบเลย์เอาต์เริ่มต้นนิดเดียว
) -> Dict[str, Any]:
    """
    Whale Optimization แบบ warm-start + local moves
    ให้ผลในรูปแบบเดียวกับ SA / SHO

    - warm-start: เริ่มจากเลย์เอาต์เดิม แล้ว jitter เล็กน้อย
    - local moves: แต่ละรอบ ขยับเฉพาะ module_sample ตัว (แทนขยับทั้ง 2 แสน+ โมดูล)
    """
    rng = random.Random(seed)

    # ---------- init population ----------
    population: List[Chip] = []
    for _ in range(pop_size):
        c = copy_chip_simple(start_chip)
        _jitter_small(rng, c, scale=init_jitter)
        population.append(c)

    # evaluate initial population
    pop_scores: List[float] = []
    pop_meta: List[Dict[str, Any]] = []
    pop_ov: List[float] = []
    for c in population:
        s, m, ov = _score(c, grid_size, overlap_grid, alpha, beta, gamma)
        pop_scores.append(s)
        pop_meta.append(m)
        pop_ov.append(ov)

    best_idx = int(np.argmin(pop_scores))
    best_chip = copy_chip_simple(population[best_idx])
    best_cost = pop_scores[best_idx]
    best_meta = pop_meta[best_idx]
    best_ov = pop_ov[best_idx]

    start_t = time.time()
    if verbose:
        print(f"WOA: pop_size={pop_size}, iters={iters}, seed={seed}")

    movable_names = [m.name for m in start_chip.get_movable_modules()]

    # ---------- main loop ----------
    for t in range(1, iters + 1):
        a = 2.0 * (1.0 - t / max(1, iters))  # standard WOA coefficient
        new_population: List[Chip] = []
        new_scores: List[float] = []
        new_meta: List[Dict[str, Any]] = []
        new_ov: List[float] = []

        for i, ch in enumerate(population):
            new_ch = copy_chip_simple(ch)

            # เลือก subset ของโมดูลที่จะขยับรอบนี้ (local move)
            if len(movable_names) == 0:
                sample_names = []
            else:
                k = min(module_sample, len(movable_names))
                sample_names = rng.sample(movable_names, k) if k < len(movable_names) else movable_names

            for name in sample_names:
                m = new_ch.modules[name]
                bx, by = best_chip.modules[name].get_position()

                r1 = rng.random()
                r2 = rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = rng.random()
                l = rng.uniform(-1.0, 1.0)

                if p < 0.5:
                    # encircling / search
                    if abs(A) < 1.0:
                        Dx = abs(C * bx - m.x)
                        Dy = abs(C * by - m.y)
                        x_new = bx - A * Dx
                        y_new = by - A * Dy
                    else:
                        # ใช้วาฬตัวอื่นในประชากร
                        j = rng.randrange(len(population))
                        r_chip = population[j]
                        rx, ry = r_chip.modules[name].get_position()
                        Dx = abs(C * rx - m.x)
                        Dy = abs(C * ry - m.y)
                        x_new = rx - A * Dx
                        y_new = ry - A * Dy
                else:
                    # bubble-net (spiral)
                    dist = math.hypot(bx - m.x, by - m.y)
                    x_new = dist * math.exp(1.0 * l) * math.cos(2 * math.pi * l) + bx
                    y_new = dist * math.exp(1.0 * l) * math.sin(2 * math.pi * l) + by

                # boundary check
                x_new = max(0.0, min(new_ch.width - m.width, x_new))
                y_new = max(0.0, min(new_ch.height - m.height, y_new))
                new_ch.modules[name].set_position(x_new, y_new)

            # คำนวณคะแนนของ candidate ตัวนี้
            s, m_meta, ov = _score(new_ch, grid_size, overlap_grid, alpha, beta, gamma)
            new_population.append(new_ch)
            new_scores.append(s)
            new_meta.append(m_meta)
            new_ov.append(ov)

            # อัปเดต global best
            if s < best_cost:
                best_chip = copy_chip_simple(new_ch)
                best_cost = s
                best_meta = m_meta
                best_ov = ov

        # Elitism: เอา best ไปแทนตัวที่แย่สุด
        worst_idx = int(np.argmax(new_scores))
        new_population[worst_idx] = copy_chip_simple(best_chip)
        new_scores[worst_idx] = best_cost
        new_meta[worst_idx] = best_meta
        new_ov[worst_idx] = best_ov

        population = new_population
        pop_scores = new_scores
        pop_meta = new_meta
        pop_ov = new_ov

        if verbose and (t % max(1, iters // 10) == 0):
            print(f"Iter {t}/{iters} | Best: {best_cost:.6f}")

    exec_time = time.time() - start_t

    # เขียนตำแหน่ง best กลับลง start_chip (เหมือน SA/SHO)
    for name, mod in best_chip.modules.items():
        if name in start_chip.modules and not start_chip.modules[name].is_fixed:
            start_chip.modules[name].set_position(mod.x, mod.y)

    result: Dict[str, Any] = {
        "best_cost": best_cost,
        "hpwl": best_meta.get("hpwl", None),
        "hpwl_normalized": best_meta.get("hpwl_normalized", None),
        "congestion_penalty": best_meta.get("congestion_penalty", None),
        "congestion_normalized": best_meta.get("congestion_normalized", None),
        "max_congestion": best_meta.get("max_congestion", None),
        "avg_congestion": best_meta.get("avg_congestion", None),
        "overflow_ratio": best_meta.get("overflow_ratio", None),
        "overlap_ratio": best_ov,
        "execution_time": exec_time,
        "iterations": iters,
    }

    if verbose:
        print("\n==== WOA RESULT ====")
        print(f"Best Cost      : {result['best_cost']}")
        print(f"HPWL           : {result['hpwl']}")
        print(f"Max Congestion : {result['max_congestion']}")
        print(f"Avg Congestion : {result['avg_congestion']}")
        print(f"Overflow Ratio : {result['overflow_ratio']}")
        print(f"Overlap Ratio  : {result['overlap_ratio']}")
        print(f"Execution Time : {result['execution_time']:.2f} sec")

    return result
