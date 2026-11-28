
"""
=============================================================================
CÀI ĐẶT THUẬT TOÁN Q-LEARNING (MODEL-FREE REINFORCEMENT LEARNING)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Nhóm 8 - Trường Đại học Thủy Lợi
=============================================================================
Căn cứ lý thuyết:
- Sử dụng thuật toán Q-Learning dựa trên phương trình Bellman.
- Cập nhật giá trị Q(s,a) thông qua trải nghiệm tương tác với môi trường.
- Sử dụng chiến lược Epsilon-Greedy để cân bằng khám phá và khai thác.
=============================================================================
"""

import sys
import os
import time
from typing import Dict, Any

import numpy as np

# Thêm thư mục gốc vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.delivery_env import DeliveryGridEnv


class QLearning:
    """
    Bộ giải thuật toán Q-Learning cho bài toán tối ưu hóa lộ trình giao hàng.
    """

    def __init__(
        self,
        env: DeliveryGridEnv,
        alpha: float = 0.1,
        gamma: float = config.GAMMA,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        num_episodes: int = 10000
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_episodes = num_episodes

        if not env.state_to_idx:
            raise ValueError(
                "Q-Learning dạng Q-table yêu cầu môi trường "
                "random_customers=False."
            )

        self.num_states = env.num_states
        self.num_actions = env.action_space.n

        # Khởi tạo bảng Q(s,a)
        self.Q = np.zeros(
            (self.num_states, self.num_actions),
            dtype=np.float64
        )

        self.history = {
            "episodes": 0,
            "episode_rewards": [],
            "episode_steps": [],
            "success_per_episode": [],
            "epsilon_history": [],
            "training_time_seconds": 0.0
        }

    def choose_action(self, state_idx: int) -> int:
        """
        Lựa chọn hành động theo chiến lược Epsilon-Greedy.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return int(np.argmax(self.Q[state_idx]))

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Huấn luyện Q-Learning thông qua tương tác với môi trường.
        """

        start_time = time.time()

        if verbose:
            print("=" * 70)
            print("BẮT ĐẦU HUẤN LUYỆN THUẬT TOÁN Q-LEARNING")
            print(f"Tổng số trạng thái: {self.num_states}")
            print(f"Số hành động: {self.num_actions}")
            print(f"Learning rate (alpha): {self.alpha}")
            print(f"Discount factor (gamma): {self.gamma}")
            print(f"Epsilon ban đầu: {self.epsilon}")
            print("=" * 70)

        for episode in range(self.num_episodes):

            # Khởi tạo trạng thái đầu mỗi episode
            observation, info = self.env.reset()
            state_idx = info["state_idx"]

            total_reward = 0.0
            steps = 0
            success = False

            for step in range(self.env.max_steps):

                # 1. Chọn hành động bằng Epsilon-Greedy
                action = self.choose_action(state_idx)

                # 2. Thực hiện hành động trong môi trường
                observation, reward, terminated, truncated, info = (
                    self.env.step(action)
                )

                next_state_idx = info["state_idx"]

                # 3. Cập nhật giá trị Q theo phương trình Bellman
                if terminated:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(
                        self.Q[next_state_idx]
                    )

                self.Q[state_idx, action] += self.alpha * (
                    target - self.Q[state_idx, action]
                )

                # 4. Chuyển sang trạng thái tiếp theo
                state_idx = next_state_idx
                total_reward += reward
                steps += 1

                if terminated:
                    success = True
                    break

                if truncated:
                    break

            # 5. Giảm epsilon sau mỗi episode
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            # 6. Lưu lịch sử huấn luyện
            self.history["episode_rewards"].append(total_reward)
            self.history["episode_steps"].append(steps)
            self.history["success_per_episode"].append(success)
            self.history["epsilon_history"].append(self.epsilon)

            if verbose and (
                (episode + 1) % 1000 == 0
                or episode == 0
                or episode == self.num_episodes - 1
            ):
                recent_rewards = self.history["episode_rewards"][-100:]
                recent_success = self.history["success_per_episode"][-100:]

                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_success) * 100

                print(
                    f"Episode {episode + 1:5d}/{self.num_episodes} | "
                    f"Reward TB: {avg_reward:8.2f} | "
                    f"Tỷ lệ thành công: {success_rate:6.2f}% | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        elapsed = time.time() - start_time

        self.history["episodes"] = self.num_episodes
        self.history["training_time_seconds"] = elapsed

        if verbose:
            print("=" * 70)
            print("HOÀN THÀNH HUẤN LUYỆN Q-LEARNING")
            print(f"Thời gian huấn luyện: {elapsed:.2f} giây")
            print("=" * 70)

        return self.history

    def extract_optimal_route(self, max_steps: int = 200) -> Dict[str, Any]:
        """
        Trích xuất lộ trình giao hàng từ bảng Q đã huấn luyện.
        Sử dụng chính sách Greedy: chọn hành động có Q(s,a) lớn nhất.
        """

        start_state = (
            self.env.depot_pos[0],
            self.env.depot_pos[1],
            0
        )

        curr_state = start_state

        route_coords = [self.env.depot_pos]
        states_sequence = [curr_state]
        actions_taken = []
        action_names = []
        rewards = []
        visited_customers = []

        total_reward = 0.0
        step_count = 0

        while (
            curr_state != self.env.terminal_state
            and step_count < max_steps
        ):
            state_idx = self.env.state_to_idx[curr_state]

            # Chọn hành động tốt nhất theo bảng Q
            action = int(np.argmax(self.Q[state_idx]))

            next_state, reward, is_terminated = self.env.get_transition(
                curr_state, action
            )

            total_reward += reward
            step_count += 1

            actions_taken.append(action)
            action_names.append(config.ACTION_NAMES[action])
            rewards.append(reward)

            # Xác định khách hàng vừa được giao hàng
            prev_mask = curr_state[2]
            next_mask = next_state[2]

            for k, customer in enumerate(self.env.customer_positions):
                if (
                    (next_mask & (1 << k))
                    and not (prev_mask & (1 << k))
                ):
                    visited_customers.append({
                        "customer_id": f"C{k + 1}",
                        "position": customer,
                        "step": step_count
                    })

            curr_state = next_state

            route_coords.append((curr_state[0], curr_state[1]))
            states_sequence.append(curr_state)

            if is_terminated:
                break

        is_completed = curr_state == self.env.terminal_state

        return {
            "is_completed": is_completed,
            "total_steps": step_count,
            "total_distance": step_count,
            "total_reward": total_reward,
            "route_coords": route_coords,
            "states_sequence": states_sequence,
            "actions_taken": actions_taken,
            "action_names": action_names,
            "rewards": rewards,
            "visited_customers_order": visited_customers
        }
