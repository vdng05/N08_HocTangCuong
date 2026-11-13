"""
=============================================================================
PHÂN TÍCH CHUYÊN SÂU ĐỘ HỘI TỤ VÀ BẢN ĐỒ NHIỆT HÀM GIÁ TRỊ V(s)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên: Vũ Hải Đăng (Thuật toán) & Vương Tiến Dũng (Môi trường)
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
Mục đích:
1. Vẽ đồ thị hàm sai số Bellman (Delta) và số trạng thái thay đổi chính sách.
2. Xuất 4 bản đồ nhiệt (Heatmaps) của hàm giá trị trạng thái V(s) tương ứng
   với 4 giai đoạn giao hàng của tác tử:
   - Giai đoạn 0: Chưa giao điểm nào (Mask = 0b000)
   - Giai đoạn 1: Đã giao C2, đang hướng tới C3 (Mask = 0b010)
   - Giai đoạn 2: Đã giao C2 và C3, đang hướng tới C1 (Mask = 0b110)
   - Giai đoạn 3: Đã giao đủ cả 3 điểm, đang tìm đường về Kho (Mask = 0b111)
3. Lưu các hình ảnh phân tích khoa học phục vụ Báo cáo Chương V.
=============================================================================
"""

import sys
import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from environment.delivery_env import DeliveryGridEnv
from algorithms.policy_iteration import PolicyIteration


def run_convergence_analysis():
    print("=" * 75)
    print("  BẮT ĐẦU PHÂN TÍCH CHUYÊN SÂU ĐỘ HỘI TỤ & HÀM GIÁ TRỊ V(s) - VŨ HẢI ĐĂNG")
    print("=" * 75)

    env = DeliveryGridEnv()
    solver = PolicyIteration(env, gamma=config.GAMMA, theta=config.THETA)
    history = solver.train(verbose=False)

    print(f"-> Thuật toán đã hội tụ hoàn toàn sau {history['pi_iterations']} vòng lặp.")
    print(f"-> Thời gian huấn luyện: {history['training_time_seconds']:.4f} giây.")

    # 1. Vẽ đồ thị hội tụ của thuật toán (Convergence Curves)
    iters = list(range(1, history["pi_iterations"] + 1))
    sweeps = history["eval_sweeps_per_iter"]
    changes = history["policy_changes_per_iter"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Biểu đồ 1: Số lượt quét Bellman (Policy Evaluation Sweeps)
    ax1.plot(iters, sweeps, marker="o", color="#2563eb", linewidth=2.2, markersize=6, label="Số lượt quét Bellman")
    ax1.set_title("Số Lượt Quét Bellman Qua Từng Vòng Lặp PI\n(Policy Evaluation Sweeps)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Vòng Lặp Policy Iteration", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Số lượt Bellman Sweeps", fontsize=10, fontweight="bold")
    ax1.set_xticks(iters)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(frameon=True)

    # Biểu đồ 2: Số trạng thái thay đổi hành động (Policy Changes)
    ax2.plot(iters, changes, marker="s", color="#dc2626", linewidth=2.2, markersize=6, label="Số trạng thái đổi hành động")
    ax2.set_title("Tốc Độ Ổn Định Chính Sách\n(Policy Changes Per Iteration)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Vòng Lặp Policy Iteration", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Số trạng thái thay đổi", fontsize=10, fontweight="bold")
    ax2.set_xticks(iters)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    curve_path = os.path.join(config.FIGURES_DIR, "convergence_curves.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[XUẤT BÁO CÁO] Đã lưu đồ thị hội tụ vào: {curve_path}")

    # 2. Xuất Bản đồ nhiệt Hàm giá trị trạng thái V(s) cho 4 giai đoạn
    # Giai đoạn 0: mask = 0 (0b000)
    # Giai đoạn 1: mask = 2 (0b010 - Đã giao C2)
    # Giai đoạn 2: mask = 6 (0b110 - Đã giao C2 và C3)
    # Giai đoạn 3: mask = 7 (0b111 - Đã giao đủ cả 3 khách)
    stage_masks = [
        (0, "Giai Đoạn 1: Xuất phát tại Kho (Chưa giao khách nào)\nMục tiêu tối ưu: Điểm C2 (7, 4)"),
        (2, "Giai Đoạn 2: Đã giao C2 (7, 4)\nMục tiêu tối ưu: Điểm C3 (7, 9)"),
        (6, "Giai Đoạn 3: Đã giao C2 & C3\nMục tiêu tối ưu: Điểm C1 (2, 9)"),
        (7, "Giai Đoạn 4: Đã giao xong tất cả 3 điểm\nMục tiêu tối ưu: Về Kho Depot (3, 1)")
    ]

    fig_heat, axes = plt.subplots(2, 2, figsize=(15, 14))
    axes = axes.flatten()

    for idx, (m, stage_title) in enumerate(stage_masks):
        ax = axes[idx]
        grid_v = np.full((config.GRID_SIZE, config.GRID_SIZE), np.nan)

        for r in range(config.GRID_SIZE):
            for c in range(config.GRID_SIZE):
                state = (r, c, m)
                if state in env.state_to_idx:
                    s_idx = env.state_to_idx[state]
                    grid_v[r, c] = solver.V[s_idx]

        # Vẽ Heatmap bằng Seaborn
        cmap = "viridis" if m != 7 else "plasma"
        sns.heatmap(
            grid_v, ax=ax, cmap=cmap, cbar=True,
            linewidths=0.5, linecolor="#cbd5e1",
            cbar_kws={"label": "Giá trị trạng thái V(s)"}
        )

        # Đánh dấu các vật cản bằng màu xám đen
        for r, c in env.obstacles:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor="#334155", edgecolor="#0f172a", hatch="//"))

        # Đánh dấu Kho hàng
        dr, dc = env.depot_pos
        ax.scatter(dc + 0.5, dr + 0.5, color="#3b82f6", s=180, edgecolors="white", linewidths=2, marker="D", zorder=5)
        ax.text(dc + 0.5, dr + 0.5, "D", color="white", fontsize=9, fontweight="bold", ha="center", va="center", zorder=6)

        # Đánh dấu 3 khách hàng
        for k, (cr, cc) in enumerate(env.customer_positions):
            is_done = bool(m & (1 << k))
            marker_color = "#22c55e" if is_done else "#f97316"
            ax.scatter(cc + 0.5, cr + 0.5, color=marker_color, s=200, edgecolors="white", linewidths=1.5, marker="*", zorder=5)
            ax.text(cc + 0.5, cr + 0.15, f"C{k+1}", color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)

        ax.set_title(stage_title, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("Cột (x)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Hàng (y)", fontsize=9, fontweight="bold")

    plt.tight_layout()
    heatmap_path = os.path.join(config.FIGURES_DIR, "value_function_heatmaps.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig_heat)
    print(f"[XUẤT BÁO CÁO] Đã lưu bản đồ nhiệt V(s) 4 giai đoạn vào: {heatmap_path}")

    # 3. Xuất bảng số liệu hội tụ ra CSV trong thư mục outputs/data/
    metrics_data = []
    for it in range(len(iters)):
        metrics_data.append({
            "Vòng lặp PI": iters[it],
            "Số lượt Bellman Sweeps": sweeps[it],
            "Số trạng thái thay đổi hành động": changes[it]
        })
    df_metrics = pd.DataFrame(metrics_data)
    csv_metrics_path = os.path.join(config.DATA_DIR, "policy_iteration_convergence_metrics.csv")
    df_metrics.to_csv(csv_metrics_path, index=False, encoding="utf-8-sig")
    print(f"[XUẤT BÁO CÁO] Đã lưu bảng thống kê số liệu hội tụ vào: {csv_metrics_path}")

    print("=" * 75)
    print("  HOÀN TẤT PHÂN TÍCH CHUYÊN SÂU!")
    print("=" * 75)


if __name__ == "__main__":
    run_convergence_analysis()
