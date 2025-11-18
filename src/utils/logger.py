# src/utils/logger.py
import csv
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ExperimentLogger:
    """
    เก็บผลการรันอัลกอริทึม:
    - summary_runs.csv : เก็บผลทุก run (ใช้ plot ทีหลัง)
    - layouts/         : เก็บตำแหน่งโมดูล + nets ของแต่ละอัลกอ
    - reports/         : txt + markdown summary ต่อ 1 ชุดการทดลอง
    """

    def __init__(self, base_dir: str = "results", enable_summary: bool = False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # summary CSV
        self.summary_path = self.base_dir / "summary_runs.csv" if enable_summary else None
        if self.summary_path and not self.summary_path.exists():
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "benchmark",
                    "algo",
                    "n_evals",
                    "seed",
                    "alpha",
                    "beta",
                    "gamma",
                    "grid",
                    "overlap_grid",
                    "best_cost",
                    "hpwl",
                    "max_congestion",
                    "avg_congestion",
                    "overflow_ratio",
                    "overlap_ratio",
                    "exec_time_sec",
                ])

    # ---------- utils for formatting ----------
    @staticmethod
    def _fmt_float(v, digits=6):
        if v is None:
            return "NA"
        return f"{float(v):.{digits}f}"

    @staticmethod
    def _fmt_time(v):
        if v is None:
            return "NA"
        return f"{float(v):.2f} s"

    @staticmethod
    def _fmt_int(v):
        if v is None:
            return "NA"
        return f"{int(round(float(v))):,}"

    @staticmethod
    def _fmt_sci(v, digits=2):
        if v is None:
            return "NA"
        return f"{float(v):.{digits}e}"

    # ---------- summary_runs.csv ----------
    def log_run(
        self,
        benchmark: str,
        algo: str,
        shared_cfg: Dict[str, Any],
        n_evals: int,
        seed: int,
        result: Dict[str, Any],
    ):
        if not self.summary_path:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            ts,
            benchmark,
            algo,
            n_evals,
            seed,
            shared_cfg.get("alpha"),
            shared_cfg.get("beta"),
            shared_cfg.get("gamma"),
            f"{shared_cfg.get('grid')}",
            f"{shared_cfg.get('overlap_grid')}",
            result.get("best_cost"),
            result.get("hpwl"),
            result.get("max_congestion"),
            result.get("avg_congestion"),
            result.get("overflow_ratio"),
            result.get("overlap_ratio"),
            result.get("execution_time"),
        ]
        with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ---------- save layout (modules + nets) ----------
    def save_layout(self, chip, benchmark: str, algo: str, tag: str = ""):
        """
        เซฟ layout + nets:
        - modules: results/layouts/{benchmark}_{algo}_{tag}_modules.csv
        - nets:    results/layouts/{benchmark}_{algo}_{tag}_nets.json
        """
        layout_dir = self.base_dir / "layouts"
        layout_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        modules_path = layout_dir / f"{benchmark}_{algo}{suffix}_modules.csv"
        nets_path = layout_dir / f"{benchmark}_{algo}{suffix}_nets.json"

        # modules
        with open(modules_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "x", "y", "width", "height", "is_fixed"])
            for m in chip.get_all_modules():
                writer.writerow([
                    m.name,
                    m.x,
                    m.y,
                    m.width,
                    m.height,
                    int(getattr(m, "is_fixed", False)),
                ])

        # nets (ในโปรเจกต์นี้ Net มี field 'modules' = รายชื่อโมดูล)
        nets_data = []
        for net in chip.nets:
            nets_data.append({
                "name": net.name,
                "modules": list(net.modules),
            })
        with open(nets_path, "w", encoding="utf-8") as f:
            json.dump(nets_data, f, indent=2)

        return modules_path, nets_path

    # ---------- write TXT summary per experiment ----------
    def write_txt_summary(
        self,
        benchmark: str,
        shared_cfg: Dict[str, Any],
        n_evals: int,
        results: Dict[str, Dict[str, Any]],
    ):
        """
        สร้างไฟล์ TXT สรุปผลการรันชุดปัจจุบัน
        เช่น results/reports/bigblue1_N100_summary.txt
        """
        report_dir = self.base_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        path = report_dir / f"{benchmark}_N{n_evals}_summary.txt"

        lines = []
        lines.append(f"Benchmark : {benchmark}")
        lines.append(f"N_evals   : {n_evals}")
        lines.append(f"alpha/beta/gamma : {shared_cfg.get('alpha')}, "
                     f"{shared_cfg.get('beta')}, {shared_cfg.get('gamma')}")
        lines.append(f"grid      : {shared_cfg.get('grid')}")
        lines.append(f"overlap   : {shared_cfg.get('overlap_grid')}")
        lines.append("")
        lines.append("=== Algorithm Results ===")

        for key in ["sa", "sho", "woa"]:
            if key not in results:
                continue
            algo_name = key.upper()
            r = results[key]
            lines.append("")
            lines.append(f"---- {algo_name} ----")
            lines.append(f"Best Cost      : {self._fmt_float(r.get('best_cost'), 6)}")
            lines.append(f"HPWL           : {self._fmt_int(r.get('hpwl'))}")
            lines.append(f"Max Congestion : {self._fmt_sci(r.get('max_congestion'), 2)}")
            lines.append(f"Avg Congestion : {self._fmt_sci(r.get('avg_congestion'), 2)}")
            lines.append(f"Overflow Ratio : {self._fmt_float(r.get('overflow_ratio'), 6)}")
            lines.append(f"Overlap Ratio  : {self._fmt_float(r.get('overlap_ratio'), 6)}")
            lines.append(f"Execution Time : {self._fmt_time(r.get('execution_time'))}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path

    # ---------- write Markdown comparison table ----------
    def write_markdown_comparison(
        self,
        benchmark: str,
        shared_cfg: Dict[str, Any],
        n_evals: int,
        results: Dict[str, Dict[str, Any]],
    ):
        """
        สร้างไฟล์ Markdown ตารางเปรียบเทียบแบบที่ต้องการ
        เช่น results/reports/bigblue1_N100_comparison.md
        """
        report_dir = self.base_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{benchmark}_N{n_evals}_comparison.md"

        # map key -> label
        algo_order = [k for k in ["sa", "sho", "woa"] if k in results]
        label = {"sa": "SA", "sho": "SHO", "woa": "WOA"}

        # ดึงค่าที่ต้องใช้
        def getv(algo_key, field):
            r = results.get(algo_key, {})
            v = r.get(field, None)
            return None if v is None else float(v)

        # สร้าง dict metrics
        metrics = [
            ("Best Cost", "best_cost", "min"),
            ("HPWL", "hpwl", "min"),
            ("Avg. Congestion", "avg_congestion", "min"),
            ("Max Congestion", "max_congestion", "min"),
            ("Overflow Ratio", "overflow_ratio", "min"),
            ("Overlap Ratio", "overlap_ratio", "min"),
            ("Time", "execution_time", "min"),
        ]

        # เตรียมค่าที่ format แล้ว
        fmt_values: Dict[str, Dict[str, str]] = {m[0]: {} for m in metrics}
        for mname, field, _ in metrics:
            for a in algo_order:
                v = getv(a, field)
                if mname in ("Best Cost", "Overflow Ratio", "Overlap Ratio"):
                    s = self._fmt_float(v, 6) if v is not None else "NA"
                elif mname in ("HPWL",):
                    s = self._fmt_int(v)
                elif mname in ("Avg. Congestion", "Max Congestion"):
                    s = self._fmt_sci(v, 2)
                elif mname in ("Time",):
                    s = self._fmt_time(v)
                else:
                    s = str(v)
                fmt_values[mname][a] = s

        # หาผู้ชนะของแต่ละ metric
        winners: Dict[str, str] = {}
        eps = 1e-12
        for mname, field, mode in metrics:
            vals = {a: getv(a, field) for a in algo_order if getv(a, field) is not None}
            if not vals:
                winners[mname] = "NA"
                continue
            if mode == "min":
                best = min(vals.values())
                win_list = [a for a, v in vals.items() if abs(v - best) <= eps]
            else:
                best = max(vals.values())
                win_list = [a for a, v in vals.items() if abs(v - best) <= eps]

            # กรณี Overflow เท่ากันหมด → ใช้ "="
            if mname == "Overflow Ratio" and len(win_list) == len(algo_order):
                winners[mname] = "="
                continue

            # กรณี Overlap/อื่น ๆ เสมอกันหลายตัว → join ด้วย "/"
            if len(win_list) > 1:
                winners[mname] = "/".join(label[a] for a in win_list)
            else:
                # ใส่ 🟢 นำหน้า
                algo_label = label[win_list[0]]
                if mname == "Time":
                    winners[mname] = f"🟢 {algo_label} (fastest)"
                else:
                    winners[mname] = f"🟢 {algo_label}"

        # เขียน markdown
        lines = []
        lines.append(f"# Comparison: {benchmark} (N_evals={n_evals})")
        lines.append("")
        header = "| Metric | " + " | ".join(f"**{label[a]}**" for a in algo_order) + " | Winner |"
        sep = "| " + " | ".join(["---"] * (len(algo_order) + 2)) + " |"
        lines.append(header)
        lines.append(sep)

        for mname, _, _ in metrics:
            row = [f"**{mname}**"]
            for a in algo_order:
                row.append(fmt_values[mname][a])
            row.append(winners[mname])
            lines.append("| " + " | ".join(row) + " |")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path
