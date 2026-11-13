"""
=============================================================================
CHƯƠNG TRÌNH KHỞI CHẠY VÀ TRỰC QUAN HÓA MÔI TRƯỜNG ĐỘC LẬP
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên phụ trách: Vương Tiến Dũng (MSV: 2351170586) & Vũ Hải Đăng
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
YÊU CẦU TỪ NHÓM TRƯỞNG HOA ĐINH:
- Môi trường hoàn toàn độc lập, KHÔNG gắn thuật toán Policy Iteration vào.
- Hiển thị bản đồ lưới 12x12 kèm BẢNG CHÚ THÍCH MÔI TRƯỜNG chi tiết:
  + Kích thước lưới (12x12)
  + Ô đen là vật cản, ô trắng là đường đi
  + Tọa độ Kho hàng cố định (3, 1)
  + Tọa độ 3 điểm giao hàng cố định: (2, 9), (7, 4), (7, 9)
  + Hàm phần thưởng cố định theo Docs: (-1, -10, +50, +100)
- Kiểm thử các hàm tương tác chuẩn Gymnasium: reset(), step(), render()
=============================================================================
"""

import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from environment.delivery_env import DeliveryGridEnv
from visualization.visualizer import GridVisualizer


def run_environment_demo(show_gui: bool = True, test_steps: bool = True):
    print("=" * 80)
    print("   BÀI TẬP LỚN: HỌC TĂNG CƯỜNG VÀ ỨNG DỤNG - NHÓM 8")
    print("   MODULE MÔ PHỎNG MÔI TRƯỜNG GIAO HÀNG LƯỚI 12x12 (ĐỘC LẬP)")
    print("   Thành viên: Vương Tiến Dũng (MSV: 2351170586) & Vũ Hải Đăng")
    print("=" * 80)

    # 1. Khởi tạo môi trường
    print("\n[1] KHỞI TẠO MÔI TRƯỜNG GYMNASIUM...")
    env = DeliveryGridEnv()
    print(f"  + Kích thước lưới:          {env.grid_size} x {env.grid_size} (Tổng {env.grid_size * env.grid_size} ô)")
    print(f"  + Kho hàng cố định (Depot): {env.depot_pos}")
    print(f"  + 3 Điểm giao cố định:      {env.customer_positions}")
    print(f"  + Số ô vật cản (ô đen):     {len(env.obstacles)} ô")
    print(f"  + Số ô đường đi (ô trắng):  {len(env.valid_cells)} ô")
    print(f"  + Không gian trạng thái S:  {env.num_states} states rời rạc")
    print(f"  + Không gian hành động A:   4 hướng (0: Lên, 1: Xuống, 2: Trái, 3: Phải)")
    print(f"  + Cấu hình Phần thưởng:     Step={config.REWARD_STEP}, Collision={config.REWARD_COLLISION}, Delivery={config.REWARD_DELIVERY}, Terminal={config.REWARD_TERMINAL}")

    # 2. Kiểm thử reset và các hàm Gym
    print("\n[2] KIỂM THỬ GIAO DIỆN CHUẨN GYMNASIUM...")
    obs, info = env.reset()
    print(f"  -> Gọi env.reset(): Trạng thái bắt đầu tại Kho {info['state_tuple']}, Mask={info['state_tuple'][2]}")

    if test_steps:
        print("\n  -> Thực hiện 3 bước di chuyển thử nghiệm để test phản hồi của môi trường:")
        # Thử đi Xuống (1)
        next_obs, reward, term, trunc, info = env.step(config.ACTION_DOWN)
        print(f"     * Bước 1: Hành động XUỐNG -> Vị trí mới: {info['state_tuple'][:2]}, Thưởng: {reward}")

        # Thử đi Phải (3)
        next_obs, reward, term, trunc, info = env.step(config.ACTION_RIGHT)
        print(f"     * Bước 2: Hành động PHẢI  -> Vị trí mới: {info['state_tuple'][:2]}, Thưởng: {reward}")

        # Thử đâm vào ô vật cản để kiểm tra cơ chế phạt va chạm
        # Vị trí hiện tại là (4, 2), bên phải (4, 3) là vật cản
        next_obs, reward, term, trunc, info = env.step(config.ACTION_RIGHT)
        print(f"     * Bước 3: Đâm vào ô (4,3) là vật cản -> Xe bị giữ nguyên tại {info['state_tuple'][:2]}, Phạt va chạm: {reward} đ")

    # 3. Xuất đồ họa Môi trường kèm Bảng chú thích môi trường
    print("\n[3] XUẤT ĐỒ HỌA MÔ HÌNH MÔI TRƯỜNG KÈM BẢNG CHÚ THÍCH...")
    visualizer = GridVisualizer(
        grid_size=env.grid_size,
        depot_pos=env.depot_pos,
        customer_positions=env.customer_positions,
        obstacles=env.obstacles,
        campus_buildings=config.CAMPUS_BUILDINGS
    )

    save_img = os.path.join(config.FIGURES_DIR, "environment_layout_12x12.png")
    visualizer.plot_environment_layout(save_path=save_img, show=show_gui)

    print("\n" + "=" * 80)
    print("   HOÀN TẤT TRÌNH DIỄN MÔI TRƯỜNG ĐỘC LẬP (ĐÃ SẴN SÀNG NỘP CHO NHÓM)!")
    print(f"   Ảnh sơ đồ môi trường đã lưu tại: {save_img}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khởi chạy và kiểm thử môi trường giao hàng độc lập")
    parser.add_argument("--no-gui", action="store_true", help="Không mở cửa sổ đồ họa (chỉ lưu ảnh ra file)")
    args = parser.parse_args()

    run_environment_demo(show_gui=not args.no_gui)
