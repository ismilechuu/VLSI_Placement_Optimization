# =========================================================
# evaluator.py (Final Version — with overlap_grid, gamma)
# =========================================================
import numpy as np
from typing import List, Optional
from .chip import Chip, Net


# -------------------------
# HPWL
# -------------------------
def calculate_HPWL(chip: Chip, net: Optional[Net] = None) -> float:
    def _hpwl_one(n: Net) -> float:
        xs = []
        ys = []
        for name in n.modules:
            mod = chip.modules.get(name)
            x, y = mod.get_position()
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            return 0.0
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    if net is not None:
        return _hpwl_one(net)

    return sum(_hpwl_one(n) for n in chip.get_all_nets())


# -------------------------
# Congestion (RUDY)
# -------------------------
def calculate_congestion(chip, grid_size=(10, 10)):
    rows, cols = grid_size
    cmap = np.zeros((rows, cols))

    cell_w = chip.width / cols
    cell_h = chip.height / rows

    for net in chip.nets:
        xs = []
        ys = []
        for name in net.modules:
            m = chip.get_module(name)
            xs.append(m.x)
            ys.append(m.y)
        if len(xs) < 2:
            continue

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        hpwl_net = (max_x - min_x) + (max_y - min_y)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        bbox_area = max(bbox_w * bbox_h, 1e-6)

        density = hpwl_net / bbox_area

        sc = int(min_x / cell_w)
        ec = int(max_x / cell_w)
        sr = int(min_y / cell_h)
        er = int(max_y / cell_h)

        sc = max(0, min(sc, cols - 1))
        ec = max(0, min(ec, cols - 1))
        sr = max(0, min(sr, rows - 1))
        er = max(0, min(er, rows - 1))

        num = (er - sr + 1) * (ec - sc + 1)
        num = max(num, 1)

        inc = density / num

        for r in range(sr, er + 1):
            for c in range(sc, ec + 1):
                cmap[r, c] += inc

    max_c = np.max(cmap)
    avg_c = np.mean(cmap)
    overflow = np.sum(cmap > 2 * avg_c) / (rows * cols)

    return cmap, max_c, avg_c, overflow


# -------------------------
# Overlap (bin-based)
# -------------------------
def calculate_overlap_ratio(chip, grid_size=(128,128)):
    gx, gy = grid_size
    W, H = chip.width, chip.height
    bw = W / gx
    bh = H / gy

    bins = [[0 for _ in range(gy)] for _ in range(gx)]
    total_area = 0

    for m in chip.modules.values():
        x = int(m.x // bw)
        y = int(m.y // bh)
        if 0 <= x < gx and 0 <= y < gy:
            bins[x][y] += m.width * m.height
        total_area += m.width * m.height

    overlap = 0
    cap = bw * bh
    for x in range(gx):
        for y in range(gy):
            if bins[x][y] > cap:
                overlap += (bins[x][y] - cap)

    if total_area == 0:
        return 0.0

    return overlap / total_area


# -------------------------
# Unified Cost Function
# -------------------------
def calculate_cost_function(
    chip,
    grid_size=(10,10),
    overlap_grid=(128,128),
    alpha=0.5,
    beta=1e-3,
    gamma=0.1
):
    # HPWL
    hpwl = calculate_HPWL(chip)

    # Congestion (RUDY)
    cmap, max_c, avg_c, overflow = calculate_congestion(chip, grid_size)
    rows, cols = grid_size
    cong_penalty = np.sum(np.where(cmap > avg_c, cmap**2, cmap))

    # Overlap
    overlap_ratio = calculate_overlap_ratio(chip, overlap_grid)

    # Normalization
    max_hpwl = len(chip.nets) * (chip.width + chip.height)
    max_hpwl = max(max_hpwl, 1)

    max_cong = rows * cols * (max_c ** 2)
    max_cong = max(max_cong, 1)

    hpwl_norm = hpwl / max_hpwl
    cong_norm = cong_penalty / max_cong

    # Total cost
    total_cost = alpha*hpwl_norm + beta*cong_norm


    return {
        "total_cost": total_cost,
        "hpwl": hpwl,
        "max_congestion": max_c,
        "avg_congestion": avg_c,
        "overflow_ratio": overflow,
        "overlap_ratio": overlap_ratio,
        "congestion_penalty": cong_penalty,
        "execution_cost": total_cost,
    }
