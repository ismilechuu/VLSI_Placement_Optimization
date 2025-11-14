import sys
import os

# เพิ่ม path ของโปรเจ็ค
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import random
import math
import matplotlib.pyplot as plt
from evaluator import calculate_HPWL
from chip import Chip


class WhaleOptimizationPlacement:
    def __init__(self, chip: Chip, num_whales=10, max_iter=30, overlap_penalty=1.0):
        """
        Improved Whale Optimization Algorithm (WOA) for VLSI placement
        """
        self.chip = chip
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.overlap_penalty = overlap_penalty
        self.bounds = (0, chip.width, 0, chip.height)

    # ✅ 1. Initialize population
    def initialize_population(self):
        """สุ่มตำแหน่งเริ่มต้นของแต่ละวาฬ"""
        population = []
        movable = self.chip.get_movable_modules()
        for _ in range(self.num_whales):
            whale = []
            for module in movable:
                x = random.uniform(0, self.chip.width - module.width)
                y = random.uniform(0, self.chip.height - module.height)
                whale.append((x, y))
            population.append(whale)
        return population

    # ✅ 2. Improved Overlap Handling (Legalization-based)
    def compute_overlap_penalty(self, whale):
        """คำนวณค่า Penalty จากการซ้อนทับ"""
        penalty = 0.0
        modules = self.chip.get_movable_modules()

        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                x1, y1 = whale[i]
                x2, y2 = whale[j]
                mod1 = modules[i]
                mod2 = modules[j]

                overlap_x = max(0, min(x1 + mod1.width, x2 + mod2.width) - max(x1, x2))
                overlap_y = max(0, min(y1 + mod1.height, y2 + mod2.height) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                penalty += overlap_area

        return penalty * self.overlap_penalty

    # ✅ 3. Fitness Function (รวม HPWL + Overlap Penalty)
    def evaluate(self, whale):
        """คำนวณ Fitness รวมค่า HPWL และ Penalty จาก Overlap"""
        self.apply_position(whale)
        hpwl = calculate_HPWL(self.chip)
        penalty = self.compute_overlap_penalty(whale)
        return hpwl + penalty  # ยิ่งน้อยยิ่งดี

    def apply_position(self, whale):
        """นำตำแหน่งของโมดูลใน whale ไปใช้จริงกับ chip"""
        movable = self.chip.get_movable_modules()
        for module, pos in zip(movable, whale):
            module.x, module.y = pos

    # ✅ 4. Main Optimization Loop (WOA Core)
    def optimize(self):
        population = self.initialize_population()
        fitness = [self.evaluate(w) for w in population]

        best_idx = fitness.index(min(fitness))
        best_whale = [pos for pos in population[best_idx]]
        best_fitness = fitness[best_idx]

        movable = self.chip.get_movable_modules()

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.num_whales):
                whale = population[i]
                new_whale = []

                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = random.random()
                l = random.uniform(-1, 1)

                for idx, (x, y) in enumerate(whale):
                    module = movable[idx]

                    if p < 0.5:
                        if abs(A) < 1:
                            bx, by = best_whale[idx]
                            x_new = bx - A * abs(C * bx - x)
                            y_new = by - A * abs(C * by - y)
                        else:
                            rand_whale = random.choice(population)
                            rx, ry = rand_whale[idx]
                            x_new = rx - A * abs(C * rx - x)
                            y_new = ry - A * abs(C * ry - y)
                    else:
                        # Bubble-net attack (วงกลม)
                        bx, by = best_whale[idx]
                        dist = abs(bx - x)
                        x_new = dist * math.exp(1 * l) * math.cos(2 * math.pi * l) + bx
                        y_new = dist * math.exp(1 * l) * math.sin(2 * math.pi * l) + by

                    # ✅ Boundary check
                    x_new = max(0, min(x_new, self.chip.width - module.width))
                    y_new = max(0, min(y_new, self.chip.height - module.height))

                    new_whale.append((x_new, y_new))

                new_fitness = self.evaluate(new_whale)

                if new_fitness < fitness[i]:
                    population[i] = new_whale
                    fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    best_whale = [pos for pos in new_whale]
                    best_fitness = new_fitness

            print(f"Iteration {t+1}/{self.max_iter} | Best Fitness (HPWL+Penalty) = {best_fitness:.2f}")

        self.apply_position(best_whale)
        return best_whale, best_fitness

        # ✅ Visualization
    def visualize(self):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(0, self.chip.width)
        ax.set_ylim(0, self.chip.height)
        ax.set_title("VLSI Placement (Improved Whale Optimization)")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")

        # 🎨 วาดโมดูลแต่ละตัว
        for module in self.chip.get_all_modules():
            rect = plt.Rectangle(
                (module.x, module.y),
                module.width,
                module.height,
                fill=True,
                color=(random.random(), random.random(), random.random()),
                alpha=0.5
            )
            ax.add_patch(rect)
            plt.text(module.x + module.width / 2,
                     module.y + module.height / 2,
                     module.name,
                     ha='center', va='center', fontsize=10, weight='bold')

        # 🔗 วาดเส้นเชื่อมระหว่างโมดูลตาม net
        for net in self.chip.nets:
            points = []
            for mod_name in net.modules:
                mod = next((m for m in self.chip.get_all_modules() if m.name == mod_name), None)
                if mod:
                    # จุดศูนย์กลางของโมดูล
                    cx = mod.x + mod.width / 2
                    cy = mod.y + mod.height / 2
                    points.append((cx, cy))

            # ถ้ามีโมดูลมากกว่า 1 ตัวใน net ให้เชื่อมเส้น
            if len(points) >= 2:
                xs, ys = zip(*points)
                ax.plot(xs, ys, 'k--', linewidth=1.0, alpha=0.6)

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()


def main():
    # ✅ สร้างชิปจำลองขนาด 100x100
    chip = Chip(100, 100)
    chip.add_module("A", 10, 10)
    chip.add_module("B", 15, 15)
    chip.add_module("C", 20, 20)

    # ✅ สร้างการเชื่อมโยงระหว่างโมดูล
    chip.add_net("Net1", ["A", "B"])
    chip.add_net("Net2", ["B", "C"])

    # ✅ เรียกใช้ Whale Optimization Placement
    woa = WhaleOptimizationPlacement(
        chip,
        num_whales=20,       # จำนวนวาฬ (population)
        max_iter=50,         # จำนวนรอบการเรียนรู้
        overlap_penalty=0.2  # ค่าบทลงโทษจากการซ้อนทับ
    )

    # ✅ เริ่มการ optimize
    best_solution, best_fitness = woa.optimize()

    # ✅ แสดงผลลัพธ์สุดท้าย
    print(f"\nBest Fitness (HPWL+Penalty): {best_fitness:.2f}")
    for mod in chip.get_all_modules():
        print(f"{mod.name}: x={mod.x:.2f}, y={mod.y:.2f}")

    # ✅ แสดงภาพผลลัพธ์
    woa.visualize()


if __name__ == "__main__":
    main()
