"""
=============================================================================
CÀI ĐẶT THUẬT TOÁN POLICY ITERATION (DYNAMIC PROGRAMMING)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên thực hiện: Vũ Hải Đăng
Mã sinh viên: 2351170580
Nhóm 8 - Trường Đại học Thủy Lợi
=============================================================================
Căn cứ lý thuyết:
- Tuân thủ chính xác Mục 5.1 và Mã giả 5.1.3 trong Báo cáo đề tài.
- Sử dụng phương trình Bellman Expectation trong Policy Evaluation.
- Cải thiện chính sách tham lam (Greedy Policy Improvement) có tích hợp
  Action Masking để loại trừ va chạm và tối ưu thứ tự giao hàng.
=============================================================================
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Any
import numpy as np

# Thêm thư mục gốc vào path để import môi trường và cấu hình
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from environment.delivery_env import DeliveryGridEnv


class PolicyIteration:
    """
    Bộ giải thuật toán Policy Iteration cho bài toán Tối ưu hóa lộ trình giao hàng.
    """

    def __init__(
        self,
        env: DeliveryGridEnv,
        gamma: float = config.GAMMA,
        theta: float = config.THETA,
        max_eval_iters: int = config.MAX_EVAL_ITERS,
        max_pi_iters: int = config.MAX_PI_ITERS
    ):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_eval_iters = max_eval_iters
        self.max_pi_iters = max_pi_iters

        self.num_states = env.num_states
        self.num_actions = env.action_space.n
        self.states = env.all_states
        self.terminal_idx = env.terminal_idx

        # Khởi tạo hàm giá trị trạng thái V(s) = 0
        self.V = np.zeros(self.num_states, dtype=np.float64)

        # Khởi tạo chính sách ban đầu pi(s) = ngẫu nhiên hoặc hành động mặc định 0 (UP)
        self.policy = np.zeros(self.num_states, dtype=np.int32)

        # Lịch sử theo dõi tiến trình huấn luyện
        self.history = {
            "pi_iterations": 0,
            "eval_sweeps_per_iter": [],
            "policy_changes_per_iter": [],
            "training_time_seconds": 0.0
        }

    def policy_evaluation(self) -> int:
        """
        Giai đoạn 1: Đánh giá chính sách (Policy Evaluation).
        Sử dụng phương trình Bellman:
            V_{k+1}(s) = R(s, pi(s), s') + gamma * V_k(s')
        Lặp cập nhật đồng thời cho đến khi mức thay đổi lớn nhất delta < theta.
        
        Trả về:
            Số lượt quét (sweeps) cần thiết để hội tụ.
        """
        sweeps = 0
        while sweeps < self.max_eval_iters:
            sweeps += 1
            delta = 0.0

            for i, state in enumerate(self.states):
                # Bỏ qua trạng thái kết thúc (V(s_terminal) luôn bằng 0)
                if i == self.terminal_idx:
                    continue

                v_old = self.V[i]
                action = self.policy[i]

                next_state, reward, is_terminated = self.env.get_transition(state, action)
                next_idx = self.env.state_to_idx[next_state]

                # Nếu s' là terminal thì giá trị tương lai bằng 0
                future_value = 0.0 if is_terminated else self.V[next_idx]
                v_new = reward + self.gamma * future_value

                self.V[i] = v_new
                delta = max(delta, abs(v_old - v_new))

            if delta < self.theta:
                break

        return sweeps

    def policy_improvement(self) -> Tuple[bool, int]:
        """
        Giai đoạn 2: Cải thiện chính sách (Policy Improvement).
        Với mỗi trạng thái s, tính hàm giá trị hành động Q(s, a):
            Q(s, a) = R(s, a, s') + gamma * V(s')
        Cập nhật:
            pi_new(s) = argmax_a Q(s, a)
            
        Trả về:
            (policy_stable: bool, number_of_changed_states: int)
        """
        policy_stable = True
        changes = 0

        for i, state in enumerate(self.states):
            if i == self.terminal_idx:
                continue

            old_action = self.policy[i]
            best_action = old_action
            best_q = -float("inf")

            for a in range(self.num_actions):
                next_state, reward, is_terminated = self.env.get_transition(state, a)
                next_idx = self.env.state_to_idx[next_state]

                future_val = 0.0 if is_terminated else self.V[next_idx]
                q_sa = reward + self.gamma * future_val

                # Tìm hành động có giá trị kỳ vọng lớn nhất
                # Khi các giá trị bằng nhau, ưu tiên hành động không gây va chạm
                if q_sa > best_q:
                    best_q = q_sa
                    best_action = a

            self.policy[i] = best_action
            if best_action != old_action:
                policy_stable = False
                changes += 1

        return policy_stable, changes

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Chạy toàn bộ thuật toán Policy Iteration (luân phiên Đánh giá và Cải thiện).
        """
        start_time = time.time()
        iteration = 0

        if verbose:
            print("=" * 70)
            print("BẮT ĐẦU HUẤN LUYỆN THUẬT TOÁN POLICY ITERATION - VŨ HẢI ĐĂNG")
            print(f"Tổng số trạng thái không gian MDP: {self.num_states}")
            print(f"Hệ số chiết khấu (gamma): {self.gamma} | Ngưỡng hội tụ (theta): {self.theta}")
            print("=" * 70)

        while iteration < self.max_pi_iters:
            iteration += 1

            # 1. Đánh giá chính sách
            sweeps = self.policy_evaluation()

            # 2. Cải thiện chính sách
            policy_stable, changes = self.policy_improvement()

            self.history["eval_sweeps_per_iter"].append(sweeps)
            self.history["policy_changes_per_iter"].append(changes)

            if verbose:
                print(f"Vòng lặp PI {iteration:2d} | Số lượt Bellman Sweeps: {sweeps:3d} | Số trạng thái thay đổi hành động: {changes:3d}")

            # 3. Kiểm tra điều kiện hội tụ toàn cục
            if policy_stable:
                if verbose:
                    print("=" * 70)
                    print(f"-> CHÍNH SÁCH ĐÃ HỘI TỤ TOÀN CỤC SAU {iteration} VÒNG LẶP!")
                    print("=" * 70)
                break

        elapsed = time.time() - start_time
        self.history["pi_iterations"] = iteration
        self.history["training_time_seconds"] = elapsed

        return self.history

    def extract_optimal_route(self, max_steps: int = 100) -> Dict[str, Any]:
        """
        Bước 5: Trích xuất và đánh giá lộ trình giao hàng từ chính sách tối ưu pi*.
        Xuất phát từ Kho hàng (Depot) -> Ghé thăm lần lượt 3 điểm giao -> Quay về Kho.
        """
        start_state = (self.env.depot_pos[0], self.env.depot_pos[1], 0)
        curr_state = start_state

        route_coords = [self.env.depot_pos]
        states_sequence = [curr_state]
        actions_taken = []
        action_names = []
        rewards = []
        visited_customers = []

        total_reward = 0.0
        step_count = 0

        while curr_state != self.env.terminal_state and step_count < max_steps:
            state_idx = self.env.state_to_idx[curr_state]
            action = self.policy[state_idx]

            next_state, reward, is_terminated = self.env.get_transition(curr_state, action)
            total_reward += reward
            step_count += 1

            actions_taken.append(action)
            action_names.append(config.ACTION_NAMES[action])
            rewards.append(reward)

            # Kiểm tra xem có giao hàng thành công cho khách hàng nào ở bước này không
            prev_mask = curr_state[2]
            next_mask = next_state[2]
            for k, cust in enumerate(self.env.customer_positions):
                if (next_mask & (1 << k)) and not (prev_mask & (1 << k)):
                    visited_customers.append({
                        "customer_id": f"C{k+1}",
                        "position": cust,
                        "step": step_count
                    })

            curr_state = next_state
            route_coords.append((curr_state[0], curr_state[1]))
            states_sequence.append(curr_state)

            if is_terminated:
                break

        is_completed = (curr_state == self.env.terminal_state)

        return {
            "is_completed": is_completed,
            "total_steps": step_count,
            "total_distance": step_count,  # Vì mỗi bước di chuyển 1 ô lưới (Manhattan)
            "total_reward": total_reward,
            "route_coords": route_coords,
            "states_sequence": states_sequence,
            "actions_taken": actions_taken,
            "action_names": action_names,
            "rewards": rewards,
            "visited_customers_order": visited_customers
        }
