"""
=============================================================================
MODULE TRỰC QUAN HÓA ĐỒ HỌA CHUYÊN NGHIỆP (ADVANCED GRID VISUALIZER)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên: Vũ Hải Đăng (MSV: 2351170580) & Vương Tiến Dũng (MSV: 2351170586)
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
Tính năng nâng cao:
1. Giao diện 2 khung (Dual-Panel HUD): Bản đồ khuôn viên trường & Bảng telemetry.
2. Nội suy chuyển động mượt mà (Smooth Motion Interpolation) - không giật cục.
3. Mô hình xe giao hàng thực tế (Delivery Van): Thân xe, cabin, 4 bánh xe,
   đèn pha chiếu sáng (headlights beam) xoay chuẩn theo góc rẽ.
4. Hiệu ứng vết đường đi (Visited Carpet) và chuyển đổi trạng thái đơn hàng thời gian thực.
5. Nhãn trực quan cho từng tòa nhà trong khuôn viên trường ĐH Thủy Lợi.
=============================================================================
"""

import sys
import os
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, Wedge
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np

# Thêm thư mục gốc vào path để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GridVisualizer:
    """
    Bộ kết xuất đồ họa và hoạt ảnh mô phỏng chất lượng cao cho bài toán giao hàng.
    """

    def __init__(
        self,
        grid_size: int = config.GRID_SIZE,
        depot_pos: Tuple[int, int] = config.DEPOT_POS,
        customer_positions: List[Tuple[int, int]] = config.CUSTOMER_POSITIONS,
        obstacles: set = config.OBSTACLES,
        campus_buildings: Dict[str, List[Tuple[int, int]]] = config.CAMPUS_BUILDINGS
    ):
        self.grid_size = grid_size
        self.depot_pos = depot_pos
        self.customer_positions = customer_positions
        self.obstacles = obstacles
        self.campus_buildings = campus_buildings

    def draw_campus_map(self, ax: plt.Axes, visited_cells: Optional[set] = None, mask: int = 0):
        """Vẽ bản đồ khuôn viên trường học chi tiết với các tòa nhà, kho và khách hàng."""
        ax.clear()
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect("equal")

        ax.set_xlabel("Tọa độ Cột (x)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Tọa độ Hàng (y)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, which="both", color="#cbd5e1", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.invert_yaxis()  # Hàng 0 ở trên cùng

        # 0. Nền chung của đường đi
        bg = Rectangle((-0.5, -0.5), self.grid_size, self.grid_size, facecolor="#f8fafc", zorder=0)
        ax.add_patch(bg)

        # 1. Vẽ các ô đã đi qua (Visited Path / Vết bánh xe)
        if visited_cells:
            for r, c in visited_cells:
                # Bỏ qua kho và điểm khách hàng để giữ màu nổi bật
                if (r, c) != self.depot_pos and (r, c) not in self.customer_positions:
                    carpet = Rectangle(
                        (c - 0.48, r - 0.48), 0.96, 0.96,
                        facecolor="#dcfce7", edgecolor="#86efac", linewidth=1.2, zorder=1
                    )
                    ax.add_patch(carpet)

        # 2. Vẽ các tòa nhà khuôn viên (Vật cản)
        building_colors = {
            "KTX Khu Tây": "#475569",
            "Nhà A1 (Hiệu Bộ)": "#334155",
            "Dãy C3 (Phòng học)": "#475569",
            "KTX Khu Nam": "#475569",
            "Thư Viện Trung Tâm": "#1e293b",
            "Giảng Đường C1": "#334155",
            "Sân Vận Động": "#065f46",
            "Xưởng Kỹ Thuật": "#334155",
            "Khu Thực Hành Điện": "#475569"
        }

        for b_name, cells in self.campus_buildings.items():
            color = building_colors.get(b_name, "#475569")
            edge_color = "#0f172a" if color != "#065f46" else "#022c22"
            for r, c in cells:
                b_rect = Rectangle(
                    (c - 0.5, r - 0.5), 1.0, 1.0,
                    facecolor=color, edgecolor=edge_color, linewidth=1.5, zorder=2
                )
                ax.add_patch(b_rect)

            # Đặt nhãn tên ở tâm của khối nhà
            center_r = sum(c[0] for c in cells) / len(cells)
            center_c = sum(c[1] for c in cells) / len(cells)
            # Rút gọn tên cho vừa ô
            short_name = b_name.replace("Trung Tâm", "").replace("Khu", "").strip()
            ax.text(
                center_c, center_r, short_name,
                color="white", fontsize=8.5, fontweight="bold",
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f172a", alpha=0.75, edgecolor="none")
            )

        # 3. Vẽ Kho hàng cố định (Depot) tại (3, 1)
        dr, dc = self.depot_pos
        depot_rect = Rectangle(
            (dc - 0.48, dr - 0.48), 0.96, 0.96,
            facecolor="#dbeafe", edgecolor="#2563eb", linewidth=2.5, linestyle="--", zorder=4
        )
        ax.add_patch(depot_rect)
        ax.text(
            dc, dr - 0.12, "DEPOT",
            color="#1d4ed8", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5
        )
        ax.text(
            dc, dr + 0.18, "KHO (3,1)",
            color="#1e40af", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5
        )

        # 4. Vẽ 3 Điểm khách hàng
        for k, (cr, cc) in enumerate(self.customer_positions):
            is_delivered = bool(mask & (1 << k))
            if is_delivered:
                # Đã giao thành công: Hộp màu xanh lá có dấu tích
                c_rect = Rectangle(
                    (cc - 0.45, cr - 0.45), 0.9, 0.9,
                    facecolor="#bbf7d0", edgecolor="#16a34a", linewidth=2.0, zorder=4
                )
                ax.add_patch(c_rect)
                ax.text(
                    cc, cr, f"C{k+1} ✓",
                    color="#15803d", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5
                )
            else:
                # Chưa giao: Ngôi sao màu cam rực rỡ
                ax.scatter(
                    cc, cr,
                    color="#ea580c", marker="*", s=420, zorder=5,
                    edgecolors="#7c2d12", linewidths=1.8
                )
                ax.text(
                    cc, cr - 0.38, f"Khách C{k+1}",
                    color="#9a3412", fontsize=9, fontweight="bold", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffedd5", edgecolor="#ea580c", alpha=0.95)
                )

    def draw_vehicle(self, ax: plt.Axes, c: float, r: float, dc: float, dr: float):
        """
        Vẽ mô hình xe giao hàng thực tế (Delivery Van):
        Có thân xe, cabin kính, đèn pha chiếu sáng phía trước và 4 bánh xe xoay theo hướng di chuyển.
        """
        # Chuẩn hóa vector chỉ hướng di chuyển
        length = np.hypot(dc, dr)
        if length > 1e-4:
            ux, uy = dc / length, dr / length
        else:
            ux, uy = 0.0, 1.0  # Mặc định hướng xuống

        # Vector vuông góc (bên phải của xe)
        wx, wy = -uy, ux

        def to_world(p_parallel, p_lateral):
            return (c + p_parallel * ux + p_lateral * wx, r + p_parallel * uy + p_lateral * wy)

        # 1. Chùm ánh sáng đèn pha (Headlights Cone Beam)
        light_length = 0.65
        light_spread = 0.35
        p_front_l = to_world(0.35, -0.20)
        p_front_r = to_world(0.35, 0.20)
        p_beam_l = to_world(0.35 + light_length, -light_spread)
        p_beam_r = to_world(0.35 + light_length, light_spread)

        beam_poly = Polygon(
            [p_front_l, p_beam_l, p_beam_r, p_front_r],
            closed=True, facecolor="#fef08a", edgecolor="none", alpha=0.35, zorder=7
        )
        ax.add_patch(beam_poly)

        # 2. Bốn bánh xe (4 Wheels)
        wheel_positions = [
            (0.22, -0.27), (0.22, 0.27),    # 2 bánh trước
            (-0.22, -0.27), (-0.22, 0.27)   # 2 bánh sau
        ]
        for wp_par, wp_lat in wheel_positions:
            w_center = to_world(wp_par, wp_lat)
            # Bánh xe dạng hình chữ nhật nhỏ xoay theo hướng xe
            w_p1 = (w_center[0] + 0.10 * ux - 0.04 * wx, w_center[1] + 0.10 * uy - 0.04 * wy)
            w_p2 = (w_center[0] + 0.10 * ux + 0.04 * wx, w_center[1] + 0.10 * uy + 0.04 * wy)
            w_p3 = (w_center[0] - 0.10 * ux + 0.04 * wx, w_center[1] - 0.10 * uy + 0.04 * wy)
            w_p4 = (w_center[0] - 0.10 * ux - 0.04 * wx, w_center[1] - 0.10 * uy - 0.04 * wy)
            wheel = Polygon([w_p1, w_p2, w_p3, w_p4], closed=True, facecolor="#111827", zorder=8)
            ax.add_patch(wheel)

        # 3. Thân xe giao hàng (Vehicle Body - Bo góc khí động học)
        body_local = [
            (0.36, -0.20), (0.38, 0.0), (0.36, 0.20),  # Mũi xe
            (-0.35, 0.22), (-0.38, 0.15), (-0.38, -0.15), (-0.35, -0.22) # Đuôi xe
        ]
        body_world = [to_world(p[0], p[1]) for p in body_local]
        body = Polygon(body_world, closed=True, facecolor="#2563eb", edgecolor="#1e3a8a", linewidth=1.8, zorder=9)
        ax.add_patch(body)

        # 4. Kính chắn gió Cabin (Windshield)
        windshield_local = [(0.15, -0.16), (0.28, -0.14), (0.28, 0.14), (0.15, 0.16)]
        windshield_world = [to_world(p[0], p[1]) for p in windshield_local]
        windshield = Polygon(windshield_world, closed=True, facecolor="#93c5fd", edgecolor="#1d4ed8", linewidth=1.0, zorder=10)
        ax.add_patch(windshield)

        # 5. Hai đèn pha sáng vàng (Yellow Headlights)
        hl_l = to_world(0.35, -0.16)
        hl_r = to_world(0.35, 0.16)
        ax.scatter([hl_l[0], hl_r[0]], [hl_l[1], hl_r[1]], color="#facc15", s=35, zorder=11, edgecolors="#ca8a04")

    def draw_environment_legend_hud(self, ax_hud: plt.Axes):
        """
        Vẽ Bảng chú thích Môi trường mô phỏng (Environment Specification & Legend)
        theo đúng yêu cầu của nhóm trưởng Hoa Đinh:
        - Biểu đồ kích thước bao nhiêu (12x12)
        - Ô đen là vật cản, ô trắng là đường đi
        - Kho hàng cố định tọa độ nào (3,1)
        - 3 điểm giao hàng cố định ở tọa độ nào
        - Các thông số không gian MDP và hàm phần thưởng cố định
        """
        ax_hud.clear()
        ax_hud.set_xlim(0, 10)
        ax_hud.set_ylim(0, 10)
        ax_hud.axis("off")

        # Nền bảng chú thích màu trắng viền xám trang nhã
        hud_bg = Rectangle((0.2, 0.2), 9.6, 9.6, facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5, zorder=0)
        ax_hud.add_patch(hud_bg)

        # Tiêu đề bảng chú thích
        ax_hud.text(
            5.0, 9.4, "CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG",
            fontsize=11.5, fontweight="bold", color="#0f172a", ha="center"
        )
        ax_hud.text(
            5.0, 9.05, "(Đề tài: Tối ưu hóa lộ trình giao hàng)",
            fontsize=8.5, color="#64748b", ha="center", style="italic"
        )
        ax_hud.axhline(8.85, xmin=0.08, xmax=0.92, color="#cbd5e1", linestyle="-", linewidth=1.2)

        # 1. Kích thước & Quy ước màu sắc ô
        ax_hud.text(0.6, 8.45, "1. BẢN ĐỒ & QUY ƯỚC Ô LƯỚI:", fontsize=9.5, fontweight="bold", color="#1e3a8a")
        ax_hud.text(0.9, 8.05, f"• Kích thước: Lưới {self.grid_size}x{self.grid_size} (Tổng 144 ô)", fontsize=9, color="#1e293b")
        
        # Icon ô đen
        blk_box = Rectangle((0.9, 7.55), 0.35, 0.35, facecolor="#475569", edgecolor="#0f172a")
        ax_hud.add_patch(blk_box)
        ax_hud.text(1.4, 7.6, f"Ô Đen/Xám: Vật cản ({len(self.obstacles)} ô công trình)", fontsize=8.5, color="#334155")

        # Icon ô trắng
        wht_box = Rectangle((0.9, 7.10), 0.35, 0.35, facecolor="#ffffff", edgecolor="#94a3b8")
        ax_hud.add_patch(wht_box)
        ax_hud.text(1.4, 7.15, f"Ô Trắng: Đường đi hợp lệ ({self.grid_size*self.grid_size - len(self.obstacles)} ô)", fontsize=8.5, color="#334155")

        ax_hud.axhline(6.85, xmin=0.08, xmax=0.92, color="#e2e8f0", linestyle="--", linewidth=0.8)

        # 2. Tọa độ các điểm mốc trên lưới
        ax_hud.text(0.6, 6.55, f"2. TỌA ĐỘ KHO & {len(self.customer_positions)} ĐIỂM GIAO HÀNG:", fontsize=9.5, fontweight="bold", color="#1e3a8a")
        
        # Kho Depot
        d_box = Rectangle((0.9, 6.10), 0.35, 0.35, facecolor="#dbeafe", edgecolor="#2563eb", linestyle="--")
        ax_hud.add_patch(d_box)
        ax_hud.text(1.4, 6.15, f"Kho hàng Depot (D): Tọa độ ({self.depot_pos[0]}, {self.depot_pos[1]})", fontsize=8.5, fontweight="bold", color="#1d4ed8")

        # Các điểm khách hàng (Hỗ trợ hiển thị linh hoạt > 2 điểm, tự động chia 2 cột nếu > 3 điểm)
        num_c = len(self.customer_positions)
        if num_c <= 3:
            for k, cust in enumerate(self.customer_positions):
                y_c = 5.65 - k * 0.42
                ax_hud.scatter(1.08, y_c + 0.15, marker="*", color="#ea580c", s=80, edgecolors="#7c2d12")
                ax_hud.text(1.4, y_c + 0.05, f"Điểm giao C{k+1}: Tọa độ ({cust[0]}, {cust[1]})", fontsize=8.5, color="#c2410c")
        else:
            for k, cust in enumerate(self.customer_positions):
                col_idx = k // 3
                row_idx = k % 3
                x_star = 1.08 + col_idx * 4.4
                x_text = 1.40 + col_idx * 4.4
                y_c = 5.65 - row_idx * 0.42
                ax_hud.scatter(x_star, y_c + 0.15, marker="*", color="#ea580c", s=75, edgecolors="#7c2d12")
                ax_hud.text(x_text, y_c + 0.05, f"C{k+1}: ({cust[0]}, {cust[1]})", fontsize=8.5, color="#c2410c")

        ax_hud.axhline(4.25, xmin=0.08, xmax=0.92, color="#e2e8f0", linestyle="--", linewidth=0.8)

        # 3. Không gian MDP (Markov Decision Process)
        num_valid = self.grid_size * self.grid_size - len(self.obstacles)
        num_states_dyn = num_valid * (1 << num_c)
        ax_hud.text(0.6, 3.95, "3. MÔ HÌNH HÓA TOÁN HỌC (MDP):", fontsize=9.5, fontweight="bold", color="#1e3a8a")
        ax_hud.text(0.9, 3.55, f"• Trạng thái S = (x, y, mask) | {num_states_dyn} states", fontsize=8.5, color="#334155")
        ax_hud.text(0.9, 3.15, "• Hành động A = {0: Lên, 1: Xuống, 2: Trái, 3: Phải}", fontsize=8.5, color="#334155")

        ax_hud.axhline(2.85, xmin=0.08, xmax=0.92, color="#e2e8f0", linestyle="--", linewidth=0.8)

        # 4. Hàm phần thưởng cố định (Fixed Reward Shaping)
        ax_hud.text(0.6, 2.55, "4. HÀM PHẦN THƯỞNG CỐ ĐỊNH (DOCS):", fontsize=9.5, fontweight="bold", color="#1e3a8a")
        ax_hud.text(0.9, 2.15, "• Chi phí di chuyển mỗi bước: R = -1", fontsize=8.5, color="#475569")
        ax_hud.text(0.9, 1.75, "• Phạt đâm vật cản / vượt biên: R = -10", fontsize=8.5, color="#dc2626")
        ax_hud.text(0.9, 1.35, "• Thưởng giao hàng thành công: R = +50", fontsize=8.5, fontweight="bold", color="#16a34a")
        ax_hud.text(0.9, 0.95, "• Thưởng hoàn thành về Kho: R = +100", fontsize=8.5, fontweight="bold", color="#2563eb")

        # Chú thích cuối
        ax_hud.text(5.0, 0.45, "(* Môi trường độc lập theo chuẩn Gymnasium *)", fontsize=8, color="#64748b", ha="center", style="italic")

    def plot_environment_layout(
        self,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Vẽ và lưu sơ đồ Môi trường mô phỏng ĐỘC LẬP
        kèm Bảng chú thích môi trường (KHÔNG có thông tin thuật toán).
        Dành riêng để nộp và trình bày phần Môi trường cho nhóm trưởng Hoa Đinh.
        """
        if save_path is None:
            save_path = os.path.join(config.FIGURES_DIR, "environment_layout_12x12.png")

        fig = plt.figure(figsize=(15, 8.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.15)
        ax_map = fig.add_subplot(gs[0])
        ax_hud = fig.add_subplot(gs[1])

        # Vẽ bản đồ môi trường tĩnh (chưa có xe chạy, chưa có lộ trình)
        self.draw_campus_map(ax_map, visited_cells=None, mask=0)

        # Vẽ xe xuất phát tại Kho hàng
        dr, dc = self.depot_pos
        self.draw_vehicle(ax_map, float(dc), float(dr), 0.0, 1.0)

        ax_map.set_title(
            f"MÔ HÌNH MÔI TRƯỜNG GIAO HÀNG TRÊN LƯỚI 12x12 (CAMPUS GRIDWORLD)\n"
            f"Khuôn viên Đại học Thủy Lợi | Kho: ({dr}, {dc}) | 3 Điểm giao cố định",
            fontsize=11.5, fontweight="bold", pad=12, color="#0f172a"
        )

        # Vẽ Bảng chú thích môi trường bên phải
        self.draw_environment_legend_hud(ax_hud)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[ĐỒ HỌA MÔI TRƯỜNG] Đã lưu sơ đồ môi trường độc lập vào: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def draw_telemetry_hud(
        self,
        ax_hud: plt.Axes,
        current_step: int,
        total_steps: int,
        current_pos: Tuple[int, int],
        current_action: str,
        mask: int,
        total_reward: float,
        event_message: str
    ):
        """Vẽ bảng điều khiển Telemetry thời gian thực ở cột bên phải."""
        ax_hud.clear()
        ax_hud.set_xlim(0, 10)
        ax_hud.set_ylim(0, 10)
        ax_hud.axis("off")

        # Nền bảng điều khiển màu ghi sáng công nghệ
        hud_bg = Rectangle((0.2, 0.2), 9.6, 9.6, facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.5, zorder=0)
        ax_hud.add_patch(hud_bg)

        # Tiêu đề bảng điều khiển
        ax_hud.text(
            5.0, 9.3, "BẢNG ĐIỀU KHIỂN TELEMETRY",
            fontsize=12, fontweight="bold", color="#0f172a", ha="center"
        )
        ax_hud.axhline(9.0, xmin=0.08, xmax=0.92, color="#94a3b8", linestyle="-", linewidth=1.0)

        # 1. Trạng thái hoạt động
        status_color = "#16a34a" if mask == 7 and current_pos == self.depot_pos else "#2563eb"
        status_text = "HOÀN TẤT NHIỆM VỤ ✓" if mask == 7 and current_pos == self.depot_pos else "ĐANG VẬN HÀNH..."
        ax_hud.text(0.8, 8.4, "TRẠNG THÁI:", fontsize=10, fontweight="bold", color="#475569")
        ax_hud.text(9.2, 8.4, status_text, fontsize=10, fontweight="bold", color=status_color, ha="right")

        # 2. Đồng hồ tiến trình bước đi
        progress_pct = (current_step / max(1, total_steps)) * 100
        ax_hud.text(0.8, 7.8, "TIẾN TRÌNH:", fontsize=10, fontweight="bold", color="#475569")
        ax_hud.text(9.2, 7.8, f"{current_step}/{total_steps} bước ({progress_pct:.0f}%)", fontsize=10, fontweight="bold", color="#0f172a", ha="right")

        # Thanh tiến trình đồ họa
        bar_bg = Rectangle((0.8, 7.3), 8.4, 0.35, facecolor="#e2e8f0", edgecolor="none")
        bar_fill = Rectangle((0.8, 7.3), 8.4 * (progress_pct / 100.0), 0.35, facecolor="#3b82f6", edgecolor="none")
        ax_hud.add_patch(bar_bg)
        ax_hud.add_patch(bar_fill)

        # 3. Tọa độ & Hành động hiện tại
        ax_hud.text(0.8, 6.7, f"TỌA ĐỘ HIỆN TẠI: ({current_pos[0]}, {current_pos[1]})", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax_hud.text(0.8, 6.2, f"HƯỚNG DI CHUYỂN: {current_action}", fontsize=9.5, fontweight="bold", color="#2563eb")

        ax_hud.axhline(5.8, xmin=0.08, xmax=0.92, color="#cbd5e1", linestyle="--", linewidth=0.8)

        # 4. Trạng thái từng đơn hàng giao
        ax_hud.text(0.8, 5.4, "DANH SÁCH ĐƠN HÀNG:", fontsize=10, fontweight="bold", color="#0f172a")

        for k, cust in enumerate(self.customer_positions):
            y_pos = 4.8 - k * 0.65
            is_done = bool(mask & (1 << k))
            badge_color = "#16a34a" if is_done else "#ea580c"
            badge_text = "ĐÃ GIAO ✓" if is_done else "CHỜ GIAO..."

            ax_hud.text(0.8, y_pos, f"Đơn C{k+1} {cust}:", fontsize=9, color="#334155")
            ax_hud.text(9.2, y_pos, badge_text, fontsize=9, fontweight="bold", color=badge_color, ha="right")

        ax_hud.axhline(3.0, xmin=0.08, xmax=0.92, color="#cbd5e1", linestyle="--", linewidth=0.8)

        # 5. Phần thưởng tích lũy (Reward Gauge)
        ax_hud.text(0.8, 2.5, "PHẦN THƯỞNG TÍCH LŨY:", fontsize=10, fontweight="bold", color="#0f172a")
        ax_hud.text(9.2, 2.5, f"{total_reward:+.1f} điểm", fontsize=11, fontweight="bold", color="#16a34a" if total_reward > 0 else "#dc2626", ha="right")

        # 6. Hộp thông báo sự kiện nổi bật (Toast / Event Banner)
        toast_bg = Rectangle((0.8, 0.6), 8.4, 1.4, facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1.2, linestyle="--")
        ax_hud.add_patch(toast_bg)
        ax_hud.text(5.0, 1.5, "THÔNG BÁO SỰ KIỆN", fontsize=8.5, fontweight="bold", color="#1e40af", ha="center")
        ax_hud.text(5.0, 0.95, event_message, fontsize=9, fontweight="bold", color="#1d4ed8", ha="center")

    def animate_route_smooth(
        self,
        route_result: Dict[str, Any],
        substeps: int = 8,
        interval_ms: int = 40,
        save_gif_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Mô phỏng xe di chuyển mượt mà (Smooth Motion Animation) qua từng ô lưới
        kèm Bảng điều khiển Telemetry thời gian thực.
        """
        route_coords = route_result["route_coords"]
        states_seq = route_result["states_sequence"]
        action_names = route_result["action_names"]
        rewards = route_result["rewards"]
        total_steps = route_result["total_steps"]

        # Xây dựng danh sách các frame nội suy mượt mà
        frames_data = []
        visited_cells_tracker = {route_coords[0]}
        accum_reward = 0.0

        for i in range(len(route_coords) - 1):
            r1, c1 = route_coords[i]
            r2, c2 = route_coords[i + 1]
            act_name = action_names[i]
            rew = rewards[i]
            accum_reward += rew
            mask_current = states_seq[i + 1][2]
            visited_cells_tracker.add((r2, c2))

            # Xác định thông báo sự kiện
            event_msg = f"Đang di chuyển: {act_name}"
            for k, cust in enumerate(self.customer_positions):
                if (r2, c2) == cust and (mask_current & (1 << k)) and not (states_seq[i][2] & (1 << k)):
                    event_msg = f"★ GIAO THÀNH CÔNG C{k+1} (+20đ)!"
            if (r2, c2) == self.depot_pos and mask_current == config.TARGET_MASK:
                event_msg = "★ HOÀN TẤT & VỀ KHO (+100đ)!"

            dr = r2 - r1
            dc = c2 - c1

            for s in range(substeps):
                t = s / substeps
                curr_r = r1 + t * dr
                curr_c = c1 + t * dc
                frames_data.append({
                    "r": curr_r,
                    "c": curr_c,
                    "dr": dr,
                    "dc": dc,
                    "step_idx": i + 1,
                    "act_name": act_name,
                    "mask": states_seq[i][2] if t < 0.5 else mask_current,
                    "visited_cells": set(visited_cells_tracker),
                    "accum_reward": accum_reward,
                    "event_msg": event_msg
                })

        # Frame cuối cùng tại đích
        last_r, last_c = route_coords[-1]
        frames_data.append({
            "r": float(last_r),
            "c": float(last_c),
            "dr": float(route_coords[-1][0] - route_coords[-2][0]),
            "dc": float(route_coords[-1][1] - route_coords[-2][1]),
            "step_idx": total_steps,
            "act_name": "KẾT THÚC",
            "mask": config.TARGET_MASK,
            "visited_cells": set(visited_cells_tracker),
            "accum_reward": accum_reward,
            "event_msg": "★ HOÀN TẤT & VỀ KHO (+100đ)!"
        })

        # Thiết lập Figure 2 khung hình
        fig = plt.figure(figsize=(15, 8.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.15)
        ax_map = fig.add_subplot(gs[0])
        ax_hud = fig.add_subplot(gs[1])

        def init():
            self.draw_campus_map(ax_map)
            self.draw_telemetry_hud(
                ax_hud, 0, total_steps, self.depot_pos, "XUẤT PHÁT", 0, 0.0, "Xe bắt đầu rời kho..."
            )
            return []

        def update(frame_idx):
            data = frames_data[frame_idx]
            r, c = data["r"], data["c"]
            dr, dc = data["dr"], data["dc"]
            step_i = data["step_idx"]
            mask_val = data["mask"]
            visited = data["visited_cells"]
            act_n = data["act_name"]
            rew_val = data["accum_reward"]
            evt_msg = data["event_msg"]

            # Vẽ bản đồ
            self.draw_campus_map(ax_map, visited_cells=visited, mask=mask_val)

            # Vẽ xe giao hàng đang di chuyển
            self.draw_vehicle(ax_map, c, r, dc, dr)

            ax_map.set_title(
                f"MÔ PHỎNG LỘ TRÌNH GIAO HÀNG TỰ HÀNH (POLICY ITERATION)\n"
                f"Khuôn viên Đại học Thủy Lợi | Lưới 12x12 | Bước: {step_i}/{total_steps}",
                fontsize=11.5, fontweight="bold", pad=10, color="#0f172a"
            )

            # Vẽ bảng telemetry
            self.draw_telemetry_hud(
                ax_hud, step_i, total_steps, (int(round(r)), int(round(c))),
                act_n, mask_val, rew_val, evt_msg
            )

            return []

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames_data),
            init_func=init, interval=interval_ms, blit=False, repeat=True
        )

        if save_gif_path:
            print(f"[ĐỒ HỌA] Đang kết xuất GIF animation mượt mà ({len(frames_data)} frames) vào: {save_gif_path} ...")
            ani.save(save_gif_path, writer="pillow", fps=1000 // interval_ms)
            print(f"[ĐỒ HỌA] Đã lưu thành công file GIF mượt mà: {save_gif_path}")

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_optimal_route(
        self,
        route_result: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """Vẽ lộ trình tĩnh độ nét cao kèm số thứ tự từng bước đi."""
        if save_path is None:
            save_path = os.path.join(config.FIGURES_DIR, "optimal_route_12x12.png")
        fig = plt.figure(figsize=(15, 8.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.15)
        ax_map = fig.add_subplot(gs[0])
        ax_hud = fig.add_subplot(gs[1])

        route_coords = route_result["route_coords"]
        total_steps = route_result["total_steps"]
        total_reward = route_result["total_reward"]

        # Vẽ nền bản đồ với toàn bộ các ô đã ghé
        visited_set = set(route_coords)
        self.draw_campus_map(ax_map, visited_cells=visited_set, mask=config.TARGET_MASK)

        # Vẽ đường nối lộ trình di chuyển (Màu tím neon nổi bật có viền trắng)
        rows = [p[0] for p in route_coords]
        cols = [p[1] for p in route_coords]

        ax_map.plot(
            cols, rows, color="#4338ca", linestyle="-", linewidth=3.5, alpha=0.9, zorder=6
        )

        # Đánh dấu mũi tên và số bước tại từng chặng
        for i in range(len(route_coords) - 1):
            r1, c1 = route_coords[i]
            r2, c2 = route_coords[i + 1]
            ax_map.annotate(
                "", xy=(c2, r2), xytext=(c1, r1),
                arrowprops=dict(arrowstyle="->", color="#312e81", lw=2.2, mutation_scale=14),
                zorder=7
            )
            # Đánh số bước đi nhỏ trên mỗi ô
            mid_c = (c1 + c2) / 2.0
            mid_r = (r1 + r2) / 2.0
            ax_map.text(
                mid_c, mid_r, f"{i+1}", fontsize=7.5, fontweight="bold",
                color="#ffffff", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="#4338ca", edgecolor="none", alpha=0.9)
            )

        ax_map.set_title(
            f"BẢN ĐỒ LỘ TRÌNH GIAO HÀNG TỐI ƯU (POLICY ITERATION)\n"
            f"Tổng quãng đường: {total_steps} bước | Tổng phần thưởng: {total_reward:+.1f} điểm",
            fontsize=12, fontweight="bold", pad=12, color="#0f172a"
        )

        # Vẽ telemetry HUD tổng kết
        self.draw_telemetry_hud(
            ax_hud, total_steps, total_steps, self.depot_pos, "HOÀN TẤT VỀ KHO",
            config.TARGET_MASK, total_reward, "★ LỘ TRÌNH ĐẠT TỐI ƯU TOÀN CỤC!"
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[ĐỒ HỌA] Đã lưu sơ đồ lộ trình chi tiết vào: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
