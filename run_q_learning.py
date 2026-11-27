
"""
=============================================================================
CHƯƠNG TRÌNH CHẠY THUẬT TOÁN Q-LEARNING
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Nhóm 8 - Trường Đại học Thủy Lợi
=============================================================================
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import config
from environment.delivery_env import DeliveryGridEnv
from algorithms.q_learning import QLearning
from visualization.visualizer import GridVisualizer


def create_output_directories():
    """Tạo các thư mục lưu kết quả."""
    output_dir = os.path.join(ROOT_DIR, "outputs")
    figures_dir = getattr(
        config,
        "FIGURES_DIR",
        os.path.join(output_dir, "figures")
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return output_dir, figures_dir


def plot_training_results(history, save_path):
    """Vẽ biểu đồ theo dõi quá trình huấn luyện Q-Learning."""

    episodes = np.arange(1, len(history["episode_rewards"]) + 1)
    rewards = np.array(history["episode_rewards"], dtype=float)
    success = np.array(history["success_per_episode"], dtype=float)
    epsilon = np.array(history["epsilon_history"], dtype=float)

    # Tính trung bình trượt trên 100 episodes
    window = min(100, len(rewards))
    moving_rewards = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )
    moving_episodes = episodes[window - 1:]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Biểu đồ phần thưởng
    axes[0].plot(
        episodes,
        rewards,
        alpha=0.2,
        label="Reward mỗi episode"
    )
    axes[0].plot(
        moving_episodes,
        moving_rewards,
        linewidth=2,
        label=f"Reward trung bình ({window} episodes)"
    )
    axes[0].set_title("QUÁ TRÌNH HỘI TỤ PHẦN THƯỞNG Q-LEARNING")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Biểu đồ tỷ lệ thành công
    moving_success = np.convolve(
        success,
        np.ones(window) / window,
        mode="valid"
    ) * 100

    axes[1].plot(
        moving_episodes,
        moving_success,
        linewidth=2,
        label=f"Tỷ lệ thành công ({window} episodes)"
    )
    axes[1].set_title("TỶ LỆ HOÀN THÀNH NHIỆM VỤ")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Biểu đồ giảm Epsilon
    axes[2].plot(episodes, epsilon, linewidth=2)
    axes[2].set_title("QUÁ TRÌNH GIẢM EPSILON")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[KẾT QUẢ] Đã lưu biểu đồ huấn luyện: {save_path}")


def save_training_history(history, save_path):
    """Lưu lịch sử huấn luyện dưới dạng JSON."""

    serializable_history = {}

    for key, value in history.items():
        if isinstance(value, np.ndarray):
            serializable_history[key] = value.tolist()
        elif isinstance(value, list):
            serializable_history[key] = [
                bool(item) if isinstance(item, (np.bool_, bool))
                else float(item) if isinstance(item, (np.floating, float))
                else int(item) if isinstance(item, (np.integer, int))
                else item
                for item in value
            ]
        else:
            serializable_history[key] = value

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(
            serializable_history,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"[KẾT QUẢ] Đã lưu lịch sử huấn luyện: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chạy thuật toán Q-Learning cho bài toán giao hàng."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Số episodes huấn luyện (mặc định: 10000)."
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Không tạo hoặc phát GIF mô phỏng."
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes phải lớn hơn 0.")

    output_dir, figures_dir = create_output_directories()

    print("\n" + "=" * 70)
    print("ỨNG DỤNG Q-LEARNING - TỐI ƯU HÓA LỘ TRÌNH GIAO HÀNG")
    print("=" * 70)

    # 1. Khởi tạo môi trường với khách hàng cố định
    env = DeliveryGridEnv(random_customers=False)

    print(f"Kích thước bản đồ: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"Số trạng thái: {env.num_states}")
    print(f"Số hành động: {env.action_space.n}")
    print(f"Vị trí kho: {env.depot_pos}")
    print(f"Các điểm giao hàng: {env.customer_positions}")

    # 2. Khởi tạo thuật toán Q-Learning
    agent = QLearning(
        env=env,
        alpha=0.1,
        gamma=config.GAMMA,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        num_episodes=args.episodes
    )

    # 3. Huấn luyện
    history = agent.train(verbose=True)

    # 4. Lưu bảng Q và lịch sử
    q_table_path = os.path.join(output_dir, "q_learning_table.npy")
    np.save(q_table_path, agent.Q)
    print(f"[KẾT QUẢ] Đã lưu bảng Q: {q_table_path}")

    history_path = os.path.join(
        output_dir,
        "q_learning_history.json"
    )
    save_training_history(history, history_path)

    # 5. Vẽ biểu đồ huấn luyện
    training_plot_path = os.path.join(
        figures_dir,
        "q_learning_training.png"
    )
    plot_training_results(history, training_plot_path)

    # 6. Trích xuất lộ trình
    route_result = agent.extract_optimal_route(max_steps=200)

    print("\n" + "=" * 70)
    print("KẾT QUẢ LỘ TRÌNH Q-LEARNING")
    print("=" * 70)
    print(f"Hoàn thành nhiệm vụ: {'CÓ' if route_result['is_completed'] else 'CHƯA'}")
    print(f"Tổng số bước: {route_result['total_steps']}")
    print(f"Tổng quãng đường: {route_result['total_distance']}")
    print(f"Tổng phần thưởng: {route_result['total_reward']:.2f}")
    print("Thứ tự giao hàng:")

    for customer in route_result["visited_customers_order"]:
        print(
            f"  {customer['customer_id']} - "
            f"Tọa độ {customer['position']} - "
            f"Bước {customer['step']}"
        )

    # 7. Khởi tạo bộ trực quan hóa
    visualizer = GridVisualizer(
        grid_size=config.GRID_SIZE,
        depot_pos=env.depot_pos,
        customer_positions=env.customer_positions,
        obstacles=config.OBSTACLES,
        campus_buildings=config.CAMPUS_BUILDINGS
    )

    # 8. Vẽ bản đồ lộ trình nếu đã hoàn thành
    if route_result["is_completed"]:
        route_plot_path = os.path.join(
            figures_dir,
            "q_learning_optimal_route.png"
        )
        visualizer.plot_optimal_route(
            route_result,
            save_path=route_plot_path,
            show=False
        )

        print(f"[KẾT QUẢ] Đã lưu bản đồ lộ trình: {route_plot_path}")

        # 9. Tạo GIF mô phỏng
        if not args.no_animation:
            gif_path = os.path.join(
                figures_dir,
                "q_learning_route.gif"
            )
            visualizer.animate_route_smooth(
                route_result,
                substeps=8,
                interval_ms=40,
                save_gif_path=gif_path,
                show=True,  
                algorithm_name="Q-LEARNING"

            )
    else:
        print(
            "\n[CẢNH BÁO] Chính sách hiện tại chưa hoàn thành nhiệm vụ. "
            "Không tạo bản đồ lộ trình tối ưu hoặc GIF."
        )
        print(
            "Có thể cần tăng số episodes hoặc điều chỉnh "
            "siêu tham số huấn luyện."
        )

    print("\n" + "=" * 70)
    print("ĐÃ KẾT THÚC CHƯƠNG TRÌNH Q-LEARNING")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    print("CHUONG TRINH BAT DAU")
    main()
    print("CHUONG TRINH KET THUC")