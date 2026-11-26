"""
=============================================================================
MÔI TRƯỜNG MÔ PHỎNG GIAO HÀNG TRÊN LƯỚI 12x12 (GYMNASIUM COMPLIANT)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên thực hiện: Vương Tiến Dũng (Môi trường) & Vũ Hải Đăng
Nhóm 8 - Trường Đại học Thủy Lợi
=============================================================================
"""

import sys
import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Thêm thư mục gốc vào path để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DeliveryGridEnv(gym.Env):
    """
    Môi trường Lưới 12x12 mô phỏng xe giao hàng trong khuôn viên trường học / đô thị.
    Tuân thủ đầy đủ chuẩn giao diện Gymnasium (OpenAI Gym mới nhất).
    
    Không gian trạng thái S:
        Tuple (row, col, visited_mask)
        - row, col: Tọa độ hiện tại của xe trên lưới 12x12 (0 <= row, col < 12)
        - visited_mask: Chuỗi nhị phân (0..7) đánh dấu trạng thái 3 điểm giao:
            + Bit 0 (1): Điểm C1 (2, 9) đã được phục vụ
            + Bit 1 (2): Điểm C2 (7, 4) đã được phục vụ
            + Bit 2 (4): Điểm C3 (7, 9) đã được phục vụ
            
    Không gian hành động A:
        Discrete(4):
        - 0: LÊN (UP)
        - 1: XUỐNG (DOWN)
        - 2: TRÁI (LEFT)
        - 3: PHẢI (RIGHT)
        
    Hàm thưởng R:
        - Di chuyển hợp lệ thông thường: -1
        - Va chạm vật cản hoặc tường biên: -10 (giữ nguyên vị trí)
        - Ghé điểm giao hàng mới: +20
        - Hoàn thành giao hết 3 điểm và quay về kho an toàn: +100 (Terminal)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        random_customers: bool = False,
        min_customers: int = 3,
        max_customers: int = 5
    ):
        super().__init__()

        self.grid_size = config.GRID_SIZE
        self.depot_pos = config.DEPOT_POS
        self.random_customers = random_customers
        self.min_customers = max(3, min_customers)  # Đảm bảo số lượng điểm giao luôn > 2
        self.max_customers = max(self.min_customers, max_customers)

        self.customer_positions = list(config.CUSTOMER_POSITIONS)
        self.num_customers = len(self.customer_positions)
        self.target_mask = (1 << self.num_customers) - 1
        self.obstacles = set(config.OBSTACLES)
        self.render_mode = render_mode

        # Không gian hành động: 4 hướng
        self.action_space = spaces.Discrete(4)

        # Không gian quan sát theo chuẩn Gym:
        # Gồm vị trí xe [row, col] và bitmask
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.int32),
                dtype=np.int32
            ),
            "visited_mask": spaces.Discrete(1 << self.max_customers if self.random_customers else (1 << self.num_customers))
        })

        # Xây dựng danh sách toàn bộ các ô hợp lệ (chỉ trên ô không có vật cản)
        self.valid_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.obstacles
        ]

        if not self.random_customers:
            self.all_states = [
                (r, c, mask)
                for (r, c) in self.valid_cells
                for mask in range(1 << self.num_customers)
            ]
            self.state_to_idx = {s: idx for idx, s in enumerate(self.all_states)}
            self.idx_to_state = {idx: s for idx, s in enumerate(self.all_states)}
            self.num_states = len(self.all_states)
            self.terminal_state = (self.depot_pos[0], self.depot_pos[1], self.target_mask)
            self.terminal_idx = self.state_to_idx[self.terminal_state]
        else:
            self.all_states = []
            self.state_to_idx = {}
            self.idx_to_state = {}
            self.num_states = len(self.valid_cells) * (1 << self.num_customers)
            self.terminal_state = (self.depot_pos[0], self.depot_pos[1], self.target_mask)
            self.terminal_idx = None

        # Biến theo dõi trạng thái hiện tại của phiên chạy
        self.current_state: Tuple[int, int, int] = (self.depot_pos[0], self.depot_pos[1], 0)
        self.step_count = 0
        self.max_steps = 200

    def is_valid_cell(self, r: int, c: int) -> bool:
        """Kiểm tra ô (r, c) có nằm trong biên và không phải vật cản hay không."""
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size and (r, c) not in self.obstacles

    def get_transition(self, state: Tuple[int, int, int], action: int) -> Tuple[Tuple[int, int, int], float, bool]:
        """
        Hàm mô hình MDP chuyển trạng thái P(s' | s, a) và tính phần thưởng R(s, a, s').
        Được thiết kế xác định (Deterministic), phục vụ thuật toán Policy Iteration (Dynamic Programming)
        và làm lõi tính toán cho môi trường mô phỏng.
        
        Trả về:
            (next_state, reward, is_terminated)
        """
        r, c, mask = state

        # Nếu đã ở trạng thái kết thúc, dừng lại
        if state == self.terminal_state:
            return state, 0.0, True

        dr, dc = config.ACTIONS[action]
        nr, nc = r + dr, c + dc

        # 1. Trường hợp va chạm vật cản hoặc vượt biên lưới
        if not self.is_valid_cell(nr, nc):
            # Xe bị giữ nguyên vị trí cũ, bị phạt nặng
            return (r, c, mask), config.REWARD_COLLISION, False

        # 2. Di chuyển hợp lệ sang ô trống (nr, nc)
        nmask = mask
        reward = config.REWARD_STEP  # Chi phí -1 mỗi bước

        # Kiểm tra xem ô đích có phải là điểm giao hàng chưa ghé hay không
        for k, cust_pos in enumerate(self.customer_positions):
            if (nr, nc) == cust_pos:
                if not (mask & (1 << k)):
                    # Lần đầu ghé điểm này: Thưởng lớn và đánh dấu bit
                    nmask = mask | (1 << k)
                    reward += config.REWARD_DELIVERY
                break

        # Kiểm tra điều kiện hoàn thành nhiệm vụ và quay về kho
        if (nr, nc) == self.depot_pos and nmask == self.target_mask:
            reward += config.REWARD_TERMINAL
            return (nr, nc, nmask), reward, True

        return (nr, nc, nmask), reward, False

    def _randomize_customers(self):
        """Sinh ngẫu nhiên số lượng (> 2) và vị trí các điểm giao hàng trên ô hợp lệ."""
        num_c = int(self.np_random.integers(self.min_customers, self.max_customers + 1))
        candidates = [cell for cell in self.valid_cells if cell != self.depot_pos]
        chosen_indices = self.np_random.choice(len(candidates), size=num_c, replace=False)
        self.customer_positions = [candidates[int(i)] for i in chosen_indices]
        self.num_customers = len(self.customer_positions)
        self.target_mask = (1 << self.num_customers) - 1
        self.terminal_state = (self.depot_pos[0], self.depot_pos[1], self.target_mask)
        self.num_states = len(self.valid_cells) * (1 << self.num_customers)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Khởi tạo lại môi trường về trạng thái ban đầu tại Kho hàng."""
        super().reset(seed=seed)

        use_random = self.random_customers
        if options and "random_customers" in options:
            use_random = bool(options["random_customers"])

        if use_random:
            self._randomize_customers()

        self.current_state = (self.depot_pos[0], self.depot_pos[1], 0)
        self.step_count = 0

        obs = self._get_obs()
        info = {
            "state_tuple": self.current_state,
            "state_idx": self.state_to_idx.get(self.current_state),
            "remaining_customers": self.num_customers,
            "customer_positions": list(self.customer_positions)
        }
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Thực hiện một hành động của tác tử trong môi trường."""
        assert self.action_space.contains(action), f"Hành động không hợp lệ: {action}"

        self.step_count += 1
        next_state, reward, terminated = self.get_transition(self.current_state, action)
        self.current_state = next_state

        truncated = self.step_count >= self.max_steps and not terminated

        obs = self._get_obs()
        delivered_count = bin(self.current_state[2]).count("1")
        info = {
            "state_tuple": self.current_state,
            "state_idx": self.state_to_idx.get(self.current_state),
            "delivered_count": delivered_count,
            "remaining_customers": self.num_customers - delivered_count,
            "step_count": self.step_count,
            "customer_positions": list(self.customer_positions)
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        """Chuyển đổi trạng thái hiện tại sang Dictionary quan sát chuẩn Gymnasium."""
        return {
            "agent_pos": np.array([self.current_state[0], self.current_state[1]], dtype=np.int32),
            "visited_mask": self.current_state[2]
        }

    def render(self) -> Optional[str]:
        """Hiển thị giao diện đồ họa ASCII trên màn hình dòng lệnh."""
        r_curr, c_curr, mask = self.current_state
        delivered_count = bin(mask).count("1")

        lines = []
        header = f"=== BẢN ĐỒ GIAO HÀNG (12x12) | Bước: {self.step_count} | Đã giao: {delivered_count}/{self.num_customers} ==="
        lines.append(header)

        col_header = "   " + " ".join(f"{c:2d}" for c in range(self.grid_size))
        lines.append(col_header)

        for r in range(self.grid_size):
            row_str = f"{r:2d} "
            for c in range(self.grid_size):
                if (r, c) == (r_curr, c_curr):
                    row_str += " R "  # Xe giao hàng (Robot)
                elif (r, c) == self.depot_pos:
                    row_str += " D "  # Kho hàng (Depot)
                elif (r, c) in self.obstacles:
                    row_str += " ■ "  # Ô vật cản (Obstacle)
                elif (r, c) in self.customer_positions:
                    k = self.customer_positions.index((r, c))
                    if mask & (1 << k):
                        row_str += " ✓ "  # Điểm đã giao thành công
                    else:
                        row_str += f"C{k+1} "  # Điểm khách hàng chưa giao
                else:
                    row_str += " . "  # Ô đường đi hợp lệ
            lines.append(row_str)

        output = "\n".join(lines)
        if self.render_mode in ("human", "ansi"):
            print(output)
        return output
