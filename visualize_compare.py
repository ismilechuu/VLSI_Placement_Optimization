import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------- helpers ----------------

def load_modules(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_lookup(mods: pd.DataFrame):
    cx = mods["x"] + mods["width"] / 2.0
    cy = mods["y"] + mods["height"] / 2.0
    return {name: (x, y) for name, x, y in zip(mods["name"], cx, cy)}


def hpwl(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def load_short_nets(nets_path: str, mods: pd.DataFrame, max_short: int = 200):
    """อ่าน *_nets.json แล้วคืน list ของ list[(x,y)] ของ net ที่ HPWL ต่ำสุด"""
    p = Path(nets_path)
    if not p.exists():
        print(f"[WARN] nets file not found: {p}")
        return []

    with open(p, "r", encoding="utf-8") as f:
        nets = json.load(f)

    lookup = build_lookup(mods)
    net_hpwl = []

    for net in nets:
        # logger ของเราควรใช้ key "modules"
        if "modules" in net:
            names = net["modules"]
        elif "pins" in net:
            # เผื่อกรณีเก่า ๆ
            names = []
            for pin in net["pins"]:
                if isinstance(pin, dict):
                    if "module" in pin:
                        names.append(pin["module"])
                    elif "name" in pin:
                        names.append(pin["name"])
                else:
                    names.append(pin)
        else:
            continue

        pts = [lookup[n] for n in names if n in lookup]
        if len(pts) >= 2:
            net_hpwl.append((hpwl(pts), pts))

    net_hpwl.sort(key=lambda x: x[0])
    short = [pts for (_, pts) in net_hpwl[:max_short]]
    print(f"[INFO] {p.name}: using {len(short)} shortest-HPWL nets")
    return short


def plot_one(ax, title: str, mods: pd.DataFrame, short_nets, chip_margin_scale=1.02):
    # แยก movable / fixed
    mov = mods[mods["is_fixed"] == 0]
    fix = mods[mods["is_fixed"] == 1]

    # plot nets ก่อน
    for pts in short_nets:
        xs_n = [p[0] for p in pts]
        ys_n = [p[1] for p in pts]
        ax.plot(xs_n, ys_n, c="limegreen", lw=0.4, alpha=0.4)

    # plot modules
    ax.scatter(
        mov["x"] + mov["width"] / 2.0,
        mov["y"] + mov["height"] / 2.0,
        s=1,
        c="#4dacff",
        alpha=0.5,
        label="movable",
    )
    ax.scatter(
        fix["x"] + fix["width"] / 2.0,
        fix["y"] + fix["height"] / 2.0,
        s=6,
        c="#ff8c42",
        alpha=0.9,
        label="fixed",
    )

    max_x = float((mods["x"] + mods["width"]).max())
    max_y = float((mods["y"] + mods["height"]).max())
    ax.set_xlim(0, max_x * chip_margin_scale)
    ax.set_ylim(0, max_y * chip_margin_scale)

    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# ---------------- main ----------------

def main():
    # แก้ชื่อไฟล์ตาม N_evals ที่ใช้จริงได้เลย
    SA_mod = "results/layouts/bigblue1_SA_N120_modules.csv"
    SA_net = "results/layouts/bigblue1_SA_N120_nets.json"

    SHO_mod = "results/layouts/bigblue1_SHO_N120_modules.csv"
    SHO_net = "results/layouts/bigblue1_SHO_N120_nets.json"

    WOA_mod = "results/layouts/bigblue1_WOA_N120_modules.csv"
    WOA_net = "results/layouts/bigblue1_WOA_N120_nets.json"

    # โหลด modules
    mods_SA = load_modules(SA_mod)
    mods_SHO = load_modules(SHO_mod)
    mods_WOA = load_modules(WOA_mod)

    # โหลด short nets (ใช้ HPWL ต่ำสุด)
    nets_SA = load_short_nets(SA_net, mods_SA, max_short=200)
    nets_SHO = load_short_nets(SHO_net, mods_SHO, max_short=200)
    nets_WOA = load_short_nets(WOA_net, mods_WOA, max_short=200)

    # วาด 3 รูปเรียงกัน
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    plot_one(axes[0], "SA Placement", mods_SA, nets_SA)
    plot_one(axes[1], "SHO Placement", mods_SHO, nets_SHO)
    plot_one(axes[2], "WOA Placement", mods_WOA, nets_WOA)

    # legend รวมไว้ที่รูปขวา
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", fontsize=10)

    plt.tight_layout()
    out_path = Path("results/layouts/bigblue1_compare_SA_SHO_WOA.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved comparison figure to {out_path}")


if __name__ == "__main__":
    main()
