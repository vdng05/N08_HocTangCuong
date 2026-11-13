"""
=============================================================================
CHƯƠNG TRÌNH THỰC NGHIỆM CHÍNH (MAIN EXPERIMENT ENTRYPOINT)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên: Vũ Hải Đăng (MSV: 2351170580) & Vương Tiến Dũng (MSV: 2351170586)
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Thêm thư mục hiện tại vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from environment.delivery_env import DeliveryGridEnv
from algorithms.policy_iteration import PolicyIteration
from visualization.visualizer import GridVisualizer


def run_experiment(animate: bool = False, save_gif: bool = False):
    print("=" * 80)
    print("   BÀI TẬP LỚN: HỌC TĂNG CƯỜNG VÀ ỨNG DỤNG - NHÓM 8")
    print("   ĐỀ TÀI: TỐI ƯU HÓA LỘ TRÌNH GIAO HÀNG TRÊN LƯỚI 12x12")
    print("   Thành viên: Vũ Hải Đăng (Thuật toán) & Vương Tiến Dũng (Môi trường)")
    print("=" * 80)

    # 1. Khởi tạo Môi trường mô phỏng (Phần việc của Vương Tiến Dũng & Vũ Hải Đăng)
    print("\n[BƯỚC 1] Khởi tạo Môi trường Mô phỏng Lưới 12x12 (Gymnasium Environment)...")
    env = DeliveryGridEnv()
    print(f"  + Kích thước lưới: {env.grid_size} x {env.grid_size}")
    print(f"  + Vị trí Kho hàng cố định (Depot): {env.depot_pos}")
    print(f"  + Vị trí các điểm giao hàng cố định: {env.customer_positions}")
    print(f"  + Số lượng ô vật cản (Obstacles): {len(env.obstacles)}")
    print(f"  + Tổng số trạng thái rời rạc trong MDP: {env.num_states}")
    print("  -> Khởi tạo môi trường thành công!")

    # 2. Khởi tạo và Huấn luyện Thuật toán Policy Iteration (Phần việc của Vũ Hải Đăng)
    print("\n[BƯỚC 2] Huấn luyện Thuật toán Policy Iteration (Dynamic Programming)...")
    solver = PolicyIteration(env=env, gamma=config.GAMMA, theta=config.THETA)
    history = solver.train(verbose=True)

    # 3. Trích xuất lộ trình giao hàng tối ưu từ chính sách pi*
    print("\n[BƯỚC 3] Trích xuất Lộ trình Giao hàng Tối ưu từ Chính sách pi*...")
    route_result = solver.extract_optimal_route(max_steps=100)

    if not route_result["is_completed"]:
        print("  [CẢNH BÁO] Tác tử chưa hoàn tất chu trình hoặc vượt quá số bước tối đa!")
        return

    # In chi tiết kết quả thực nghiệm
    print("\n" + "=" * 80)
    print("                      KẾT QUẢ THỰC NGHIỆM CHI TIẾT")
    print("=" * 80)
    print(f"  * Trạng thái hoàn thành:            THÀNH CÔNG (100% các điểm đã giao và về kho an toàn)")
    print(f"  * Tổng số bước di chuyển (Quãng đường): {route_result['total_steps']} bước")
    print(f"  * Tổng phần thưởng tích lũy (Reward): {route_result['total_reward']:.2f} điểm")
    print(f"  * Số vòng lặp Policy Iteration:       {history['pi_iterations']} vòng")
    print(f"  * Thời gian huấn luyện:               {history['training_time_seconds']:.4f} giây")
    
    print("\n  * Thứ tự ghé thăm các điểm khách hàng:")
    for item in route_result["visited_customers_order"]:
        print(f"    - Điểm {item['customer_id']} tại tọa độ {item['position']} -> Đạt được ở Bước {item['step']}")

    # 4. Lưu bảng kết quả chi tiết từng bước ra file CSV trong thư mục outputs/data/
    csv_path = os.path.join(config.DATA_DIR, "route_steps_log.csv")
    steps_data = []
    for step_i in range(len(route_result["actions_taken"])):
        curr_p = route_result["route_coords"][step_i]
        next_p = route_result["route_coords"][step_i + 1]
        act_name = route_result["action_names"][step_i]
        rew = route_result["rewards"][step_i]
        mask_val = route_result["states_sequence"][step_i][2]
        steps_data.append({
            "Bước": step_i + 1,
            "Tọa độ hiện tại": f"({curr_p[0]}, {curr_p[1]})",
            "Hành động": act_name,
            "Tọa độ kế tiếp": f"({next_p[0]}, {next_p[1]})",
            "Phần thưởng": rew,
            "Bitmask đã giao": bin(mask_val)
        })
    df_steps = pd.DataFrame(steps_data)
    df_steps.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[DỮ LIỆU] Đã lưu chi tiết từng bước di chuyển vào: {csv_path}")

    # 5. Lưu bảng tổng hợp số liệu để phục vụ Báo cáo Chương V vào outputs/data/
    summary_path = os.path.join(config.DATA_DIR, "algorithm_benchmark_summary.csv")
    summary_data = [{
        "Thuật toán": "Policy Iteration (Dynamic Programming)",
        "Thành viên phụ trách": "Vũ Hải Đăng",
        "Kích thước lưới": "12x12",
        "Số điểm giao": 3,
        "Tổng quãng đường (bước)": route_result["total_steps"],
        "Tổng phần thưởng": route_result["total_reward"],
        "Số vòng lặp hội tụ": history["pi_iterations"],
        "Thời gian thực thi (s)": round(history["training_time_seconds"], 4),
        "Thứ tự giao hàng": " -> ".join([c["customer_id"] for c in route_result["visited_customers_order"]])
    }]
    pd.DataFrame(summary_data).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[DỮ LIỆU] Đã lưu bảng tổng hợp chỉ số thuật toán vào: {summary_path}")

    # 6. Trực quan hóa Đồ họa nâng cao (Lưu vào outputs/figures/)
    print("\n[BƯỚC 4] Kết xuất Đồ họa Trực quan hóa Lộ trình (Dual-Panel Map & Telemetry)...")
    visualizer = GridVisualizer(
        grid_size=env.grid_size,
        depot_pos=env.depot_pos,
        customer_positions=env.customer_positions,
        obstacles=env.obstacles,
        campus_buildings=config.CAMPUS_BUILDINGS
    )

    img_save_path = os.path.join(config.FIGURES_DIR, "optimal_route_12x12.png")
    visualizer.plot_optimal_route(route_result, save_path=img_save_path, show=False)

    # Nếu người dùng yêu cầu bật Animation hoặc lưu GIF (Lưu vào outputs/animations/)
    if animate or save_gif:
        gif_path = os.path.join(config.ANIMATIONS_DIR, "delivery_animation.gif") if save_gif else None
        print("\n[BƯỚC 5] Đang xử lý Hoạt ảnh mô phỏng chuyển động MƯỢT MÀ (Smooth Motion Animation)...")
        visualizer.animate_route_smooth(
            route_result, substeps=6, interval_ms=45, save_gif_path=gif_path, show=animate
        )
    else:
        print("\n[GỢI Ý] Để xem hoạt ảnh mô phỏng xe chuyển động mượt mà (Animation), hãy chạy lệnh:")
        print("        python main.py --animate")

    print("\n" + "=" * 80)
    print("   HOÀN TẤT THỰC NGHIỆM CHO VŨ HẢI ĐĂNG & VƯƠNG TIẾN DŨNG!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chương trình tối ưu lộ trình giao hàng lưới 12x12 bằng Policy Iteration")
    parser.add_argument("--animate", action="store_true", help="Hiển thị cửa sổ hoạt ảnh Matplotlib chuyển động từng bước của xe")
    parser.add_argument("--save-gif", action="store_true", help="Lưu hoạt ảnh thành file delivery_animation.gif")
    args = parser.parse_args()

    run_experiment(animate=args.animate, save_gif=args.save_gif)
