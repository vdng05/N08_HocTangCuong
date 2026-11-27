"""On-policy first-visit Monte Carlo control cho môi trường giao hàng."""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from environment.delivery_env import DeliveryGridEnv

Position = Tuple[int, int]
CustomerLayout = Tuple[Position, ...]
EnvironmentState = Tuple[int, int, int]
# Khóa trạng thái gồm ID bố trí khách, hàng, cột và bitmask khách đã giao.
State = Tuple[int, int, int, int]
StateAction = Tuple[State, int]
# Mỗi bước trong episode lưu (trạng thái, hành động, phần thưởng).
EpisodeStep = Tuple[State, int, float]


class OnPolicyFirstVisitMC:
    """Huấn luyện chính sách ε-soft từ các episode tương tác với môi trường."""

    def __init__(
        self,
        env: DeliveryGridEnv,
        gamma: float = config.GAMMA,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Gamma phải nằm trong khoảng từ 0 đến 1.")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("Epsilon phải lớn hơn 0 và không vượt quá 1.")

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.num_actions = env.action_space.n

        # Chỉ lưu các trạng thái đã gặp vì có nhiều cách bố trí khách hàng.
        # Q lưu giá trị của từng hành động; policy lưu hành động greedy hiện tại.
        self.Q: Dict[State, np.ndarray] = {}
        self.policy: Dict[State, int] = {}
        # Lưu các return để tính trung bình mẫu Q(s, a) sau mỗi episode.
        self.returns: Dict[StateAction, List[float]] = {}
        # Ánh xạ bố trí tọa độ khách hàng sang ID ngắn dùng trong khóa trạng thái.
        self._layout_ids: Dict[CustomerLayout, int] = {}
        self.last_customer_layout: Optional[CustomerLayout] = None

        self.history: Dict[str, Any] = {
            "episodes": 0,
            "episode_rewards": [],
            "episode_lengths": [],
            "completed_episodes": 0,
            "truncated_episodes": 0,
            "training_time_seconds": 0.0,
        }

    def _make_state_key(
        self,
        state: EnvironmentState,
        customer_positions: List[Position],
    ) -> State:
        # Cùng vị trí xe và bitmask nhưng khác bố trí khách là hai trạng thái khác nhau.
        layout = tuple((int(row), int(col)) for row, col in customer_positions)
        layout_id = self._layout_ids.get(layout)
        if layout_id is None:
            layout_id = len(self._layout_ids)
            self._layout_ids[layout] = layout_id
        return layout_id, int(state[0]), int(state[1]), int(state[2])

    def _ensure_state(self, state: State) -> None:
        # Khởi tạo Q và hành động greedy lần đầu tiên bắt gặp một trạng thái.
        if state in self.Q:
            return

        self.Q[state] = self.rng.uniform(-1.0, 1.0, size=self.num_actions)
        self.policy[state] = int(self.rng.integers(self.num_actions))

    def _set_epsilon_soft_policy(
        self, state: State, best_action: Optional[int] = None
    ) -> None:
        if best_action is None:
            # Nếu nhiều hành động cùng có Q lớn nhất, chọn ngẫu nhiên để tránh thiên lệch.
            best_actions = np.flatnonzero(
                np.isclose(self.Q[state], np.max(self.Q[state]))
            )
            best_action = int(self.rng.choice(best_actions))

        # Chỉ lưu hành động tốt nhất; xác suất epsilon-soft được áp dụng khi lấy hành động.
        self.policy[state] = best_action

    def _sample_action(self, state: State) -> int:
        # Khám phá: chọn ngẫu nhiên; khai thác: chọn hành động greedy hiện tại.
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return self.policy[state]

    def _generate_episode(self, episode_index: int) -> Tuple[List[EpisodeStep], bool]:
        # Mỗi episode có seed riêng để tái lập được cả chuỗi bố trí ngẫu nhiên.
        episode_seed = None if self.seed is None else self.seed + episode_index
        _, info = self.env.reset(seed=episode_seed)
        customer_positions = [
            tuple(position) for position in info["customer_positions"]
        ]
        state = self._make_state_key(
            tuple(info["state_tuple"]),
            customer_positions,
        )
        self.last_customer_layout = tuple(customer_positions)

        # Chạy theo policy hiện tại đến khi giao xong hoặc chạm giới hạn bước.
        episode: List[EpisodeStep] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            self._ensure_state(state)
            action = self._sample_action(state)
            _, reward, terminated, truncated, info = self.env.step(action)
            episode.append((state, action, float(reward)))
            state = self._make_state_key(
                tuple(info["state_tuple"]),
                [tuple(position) for position in info["customer_positions"]],
            )

        return episode, terminated

    def train(
        self,
        num_episodes: int = 10_000,
        verbose: bool = False,
        log_interval: int = 1_000,
    ) -> Dict[str, Any]:
        """Tính return theo episode và cập nhật mỗi cặp lần đầu xuất hiện."""
        if num_episodes <= 0:
            raise ValueError("Số episode phải là số nguyên dương.")
        if log_interval <= 0:
            raise ValueError("Chu kỳ thông báo phải là số nguyên dương.")

        start_time = time.perf_counter()
        completed_episodes = 0
        truncated_episodes = 0

        # Mỗi episode cung cấp các mẫu return để cập nhật giá trị Q.
        for episode_index in range(num_episodes):
            episode, terminated = self._generate_episode(episode_index)
            completed_episodes += int(terminated)
            truncated_episodes += int(not terminated)

            episode_reward = sum(reward for _, _, reward in episode)
            self.history["episode_rewards"].append(episode_reward)
            self.history["episode_lengths"].append(len(episode))

            # Tính G_t lùi từ cuối episode: G_t = R_(t+1) + gamma * G_(t+1).
            returns_by_step = [0.0] * len(episode)
            discounted_return = 0.0
            for step_index in range(len(episode) - 1, -1, -1):
                discounted_return = (
                    self.gamma * discounted_return + episode[step_index][2]
                )
                returns_by_step[step_index] = discounted_return

            # First-visit: chỉ dùng lần xuất hiện đầu tiên của mỗi cặp (state, action).
            visited_state_actions = set()
            for (state, action, _), episode_return in zip(
                episode, returns_by_step
            ):
                state_action = (state, action)
                if state_action in visited_state_actions:
                    continue
                visited_state_actions.add(state_action)

                self.returns.setdefault(state_action, []).append(episode_return)
                # Ước lượng Q bằng trung bình các return đã quan sát cho cặp này.
                self.Q[state][action] = np.mean(self.returns[state_action])
                self._set_epsilon_soft_policy(state)

            if verbose and (
                (episode_index + 1) % log_interval == 0
                or episode_index + 1 == num_episodes
            ):
                window_start = max(0, episode_index + 1 - log_interval)
                recent_rewards = self.history["episode_rewards"][window_start:]
                print(
                    f"Episode {episode_index + 1}/{num_episodes} | "
                    f"Thưởng trung bình gần đây: {np.mean(recent_rewards):.2f} | "
                    f"Hoàn thành: {completed_episodes} | "
                    f"Chạm giới hạn bước: {truncated_episodes}"
                )

        # Cộng dồn thống kê để có thể gọi train nhiều lần trên cùng đối tượng.
        self.history["episodes"] += num_episodes
        self.history["completed_episodes"] += completed_episodes
        self.history["truncated_episodes"] += truncated_episodes
        self.history["completion_rate"] = (
            self.history["completed_episodes"] / self.history["episodes"]
        )
        self.history["training_time_seconds"] += time.perf_counter() - start_time
        return self.history

    def extract_greedy_route(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Đánh giá chính sách greedy trên bố trí khách hàng của episode cuối."""
        if max_steps is None:
            max_steps = self.env.max_steps
        if max_steps <= 0:
            raise ValueError("Số bước tối đa phải là số nguyên dương.")

        if self.last_customer_layout is None:
            _, info = self.env.reset(seed=self.seed)
        else:
            # Không random lại để giữ nguyên bố trí của episode huấn luyện cuối.
            _, info = self.env.reset(options={"random_customers": False})

        customer_positions = [
            tuple(position) for position in info["customer_positions"]
        ]
        state_key = self._make_state_key(
            tuple(info["state_tuple"]),
            customer_positions,
        )
        state: EnvironmentState = state_key[1:]
        route_coords = [(state[0], state[1])]
        states_sequence = [state]
        actions_taken: List[int] = []
        action_names: List[str] = []
        rewards: List[float] = []
        visited_customers: List[Dict[str, Any]] = []

        # Đánh giá greedy: luôn chọn hành động có Q cao nhất, không khám phá.
        for step_count in range(1, max_steps + 1):
            if state == self.env.terminal_state:
                break

            self._ensure_state(state_key)
            action = int(np.argmax(self.Q[state_key]))
            previous_mask = state[2]
            _, reward, terminated, truncated, info = self.env.step(action)
            next_state: EnvironmentState = tuple(info["state_tuple"])
            next_state_key = self._make_state_key(
                next_state,
                [tuple(position) for position in info["customer_positions"]],
            )

            actions_taken.append(action)
            action_names.append(config.ACTION_NAMES[action])
            rewards.append(float(reward))

            # So sánh bitmask trước/sau bước đi để ghi nhận khách vừa được giao.
            for customer_idx, customer_position in enumerate(customer_positions):
                if (next_state[2] & (1 << customer_idx)) and not (
                    previous_mask & (1 << customer_idx)
                ):
                    visited_customers.append(
                        {
                            "customer_id": f"C{customer_idx + 1}",
                            "position": customer_position,
                            "step": step_count,
                        }
                    )

            state = next_state
            route_coords.append((state[0], state[1]))
            states_sequence.append(state)
            state_key = next_state_key
            if terminated or truncated:
                break

        return {
            "is_completed": state == self.env.terminal_state,
            "total_steps": len(actions_taken),
            "total_reward": float(sum(rewards)),
            "customer_positions": customer_positions,
            "route_coords": route_coords,
            "states_sequence": states_sequence,
            "actions_taken": actions_taken,
            "action_names": action_names,
            "rewards": rewards,
            "visited_customers_order": visited_customers,
        }
