# =======================
#  SA.py (Final Version)
# =======================
import time
import math
import random
from copy import deepcopy
from src.evaluator import calculate_cost_function


def copy_chip_simple(chip):
    """Clone chip (modules + pos) but share nets for memory saving."""
    new_chip = deepcopy(chip)
    return new_chip


def _approx_overlap_ratio(chip, grid_bins):
    """Approximate overlap ratio using simple bin occupancy."""
    W = chip.width
    H = chip.height
    gx, gy = grid_bins

    bin_w = W / gx
    bin_h = H / gy

    bins = [[0 for _ in range(gy)] for _ in range(gx)]

    total_area = 0
    overlap_area = 0

    for m in chip.modules.values():
        x = int(m.x // bin_w)
        y = int(m.y // bin_h)
        if 0 <= x < gx and 0 <= y < gy:
            bins[x][y] += m.width * m.height
            total_area += m.width * m.height

    for x in range(gx):
        for y in range(gy):
            if bins[x][y] > (bin_w * bin_h):
                overlap_area += (bins[x][y] - bin_w * bin_h)

    if total_area == 0:
        return 0.0
    return overlap_area / total_area


def simulated_annealing(
    start_chip,
    iters=100,
    grid_size=(64, 64),
    overlap_grid=(128, 128),
    alpha=0.5,
    beta=1e-3,
    gamma=0.1,
    t0=None,
    t_min=1e-6,
    cooling=0.99,
    move_disp_prob=0.7,
    disp_scale_init=0.02,
    disp_scale_final=0.002,
    seed=42,
    verbose=True,
):
    rng = random.Random(seed)

    # ทำงานบนสำเนา
    chip = copy_chip_simple(start_chip)
    best_chip = copy_chip_simple(start_chip)

    # ใช้ “โมดูลที่ขยับได้” จริง ๆ (ไม่ยุ่งกับ fixed)
    movable = [m for m in chip.modules.values() if not m.is_fixed]
    if not movable:
        # ไม่มีอะไรให้ optimize
        meta0 = calculate_cost_function(
            chip, grid_size=grid_size, overlap_grid=overlap_grid,
            alpha=alpha, beta=beta
        )
        ov0 = _approx_overlap_ratio(chip, overlap_grid)
        return {
            "best_cost": float(meta0["total_cost"] + gamma * ov0),
            "hpwl": float(meta0["hpwl"]),
            "max_congestion": float(meta0["max_congestion"]),
            "avg_congestion": float(meta0["avg_congestion"]),
            "overflow_ratio": float(meta0["overflow_ratio"]),
            "overlap_ratio": float(ov0),
            "execution_time": 0.0,
        }

    # ---------- initial cost ----------
    init_meta = calculate_cost_function(
        chip, grid_size=grid_size, overlap_grid=overlap_grid,
        alpha=alpha, beta=beta
    )
    init_overlap = _approx_overlap_ratio(chip, overlap_grid)
    init_cost = init_meta["total_cost"] + gamma * init_overlap
    best_cost = init_cost

    # ---------- auto T0 ----------
    if t0 is None:
        trials = []
        for _ in range(32):
            m = rng.choice(movable)
            oldx, oldy = m.x, m.y

            dx = (rng.random() - 0.5) * chip.width * 0.05
            dy = (rng.random() - 0.5) * chip.height * 0.05
            m.x = max(0, min(chip.width, m.x + dx))
            m.y = max(0, min(chip.height, m.y + dy))

            meta_try = calculate_cost_function(
                chip, grid_size=grid_size, overlap_grid=overlap_grid,
                alpha=alpha, beta=beta
            )
            ovlp_try = _approx_overlap_ratio(chip, overlap_grid)
            c_try = meta_try["total_cost"] + gamma * ovlp_try
            trials.append(abs(c_try - init_cost))

            # revert
            m.x, m.y = oldx, oldy

        avg_d = sum(trials) / (len(trials) + 1e-9)
        t0 = avg_d * 2.0 if avg_d > 0 else 1e-3

    T = t0
    disp_scale = disp_scale_init
    disp_decay = (disp_scale_final / disp_scale_init) ** (1 / max(1, iters))

    t_start = time.time()

    # ---------- main loop ----------
    for it in range(1, iters + 1):
        m = rng.choice(movable)
        oldx, oldy = m.x, m.y

        # backup สำหรับกรณี swap
        m2 = None
        oldx2 = oldy2 = None

        if rng.random() < move_disp_prob:
            # displacement move
            dx = (rng.random() - 0.5) * chip.width * disp_scale
            dy = (rng.random() - 0.5) * chip.height * disp_scale
            m.x = max(0, min(chip.width,  m.x + dx))
            m.y = max(0, min(chip.height, m.y + dy))
        else:
            # swap กับอีกโมดูลที่ขยับได้
            m2 = rng.choice(movable)
            oldx2, oldy2 = m2.x, m2.y
            m.x, m2.x = m2.x, m.x
            m.y, m2.y = m2.y, m.y

        meta2 = calculate_cost_function(
            chip, grid_size=grid_size, overlap_grid=overlap_grid,
            alpha=alpha, beta=beta
        )
        ov2 = _approx_overlap_ratio(chip, overlap_grid)
        cand_cost = meta2["total_cost"] + gamma * ov2

        dcost = cand_cost - init_cost
        if dcost < 0 or rng.random() < math.exp(-dcost / (T + 1e-12)):
            # accept
            init_cost = cand_cost
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_chip = copy_chip_simple(chip)
        else:
            # reject → revert move ทั้งคู่ (ถ้า swap)
            m.x, m.y = oldx, oldy
            if m2 is not None:
                m2.x, m2.y = oldx2, oldy2

        T = max(t_min, T * cooling)
        disp_scale *= disp_decay

        if verbose and (it % max(1, iters // 10) == 0 or it <= 5):
            print(f"Iteration {it}/{iters} | SA best cost: {best_cost:.6f}")

    t_end = time.time()

    # ---------- copy best positions กลับไปที่ start_chip ----------
    for name, m_src in best_chip.modules.items():
        if name in start_chip.modules and not start_chip.modules[name].is_fixed:
            start_chip.modules[name].set_position(m_src.x, m_src.y)

    final_meta = calculate_cost_function(
        best_chip, grid_size=grid_size, overlap_grid=overlap_grid,
        alpha=alpha, beta=beta
    )
    final_ovlp = _approx_overlap_ratio(best_chip, overlap_grid)

    # 🔊 พิมพ์ผลเหมือนเดิม (เฉพาะตอน verbose=True)
    # if verbose:
    #     print("\n==== SA RESULT ====")
    #     print(f"Best Cost      : {float(best_cost)}")
    #     print(f"HPWL           : {float(final_meta['hpwl'])}")
    #     print(f"Max Congestion : {float(final_meta['max_congestion'])}")
    #     print(f"Avg Congestion : {float(final_meta['avg_congestion'])}")
    #     print(f"Overflow Ratio : {float(final_meta['overflow_ratio'])}")
    #     print(f"Overlap Ratio  : {float(final_ovlp)}")
    #     print(f"Execution Time : {t_end - t_start:.2f} sec")

    return {
        "best_cost": float(best_cost),
        "hpwl": float(final_meta["hpwl"]),
        "max_congestion": float(final_meta["max_congestion"]),
        "avg_congestion": float(final_meta["avg_congestion"]),
        "overflow_ratio": float(final_meta["overflow_ratio"]),
        "overlap_ratio": float(final_ovlp),
        "execution_time": t_end - t_start,
    }

    

