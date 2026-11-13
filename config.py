"""
=============================================================================
CẤU HÌNH BÀI TOÁN GIAO HÀNG LƯỚI 12x12 (CAMPUS DELIVERY GRIDWORLD)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên thực hiện: Vũ Hải Đăng (Thuật toán) & Vương Tiến Dũng (Môi trường)
Nhóm 8 - Trường Đại học Thủy Lợi
=============================================================================
"""

# Kích thước lưới
GRID_SIZE = 12  # Lưới 12 x 12 (hàng 0..11, cột 0..11)

# Tọa độ Kho hàng cố định (Depot) theo yêu cầu của nhóm
DEPOT_POS = (3, 1)

# Tọa độ các điểm giao hàng cố định (Customer Delivery Points)
# C1: (2, 9), C2: (7, 4), C3: (7, 9)
CUSTOMER_POSITIONS = [
    (2, 9),  # C1
    (7, 4),  # C2
    (7, 9)   # C3
]

NUM_CUSTOMERS = len(CUSTOMER_POSITIONS)
TARGET_MASK = (1 << NUM_CUSTOMERS) - 1  # 0b111 = 7 (khi cả 3 điểm đều đã được phục vụ)

# Danh sách tọa độ các ô vật cản (Obstacles - Các tòa nhà / KTX / Sân bóng)
# Dựa trên layout mặt bằng khuôn viên và ảnh chụp phác họa trong nhóm:
CAMPUS_BUILDINGS = {
    "KTX Khu Tây": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "Nhà A1 (Hiệu Bộ)": [(0, 3), (0, 4), (2, 3), (2, 4)],
    "Dãy C3 (Phòng học)": [(4, 3), (5, 3), (7, 3), (8, 3), (9, 3)],
    "KTX Khu Nam": [(5, 0), (5, 1), (6, 0), (6, 1), (8, 1)],
    "Thư Viện Trung Tâm": [(1, 6), (1, 7), (2, 6), (2, 7)],
    "Giảng Đường C1": [(4, 6), (5, 6), (5, 7)],
    "Sân Vận Động": [(9, 6), (9, 7), (10, 6), (10, 7)],
    "Xưởng Kỹ Thuật": [(4, 10), (4, 11), (5, 10), (5, 11)],
    "Khu Thực Hành Điện": [(0, 9), (0, 10)]
}

OBSTACLES = set()
for building_cells in CAMPUS_BUILDINGS.values():
    OBSTACLES.update(building_cells)

# Tên chi tiết cho các điểm giao hàng
CUSTOMER_LABELS = {
    0: "Khách hàng C1 (Khu Kỹ Thuật)",
    1: "Khách hàng C2 (Giảng Đường C1)",
    2: "Khách hàng C3 (Ký Túc Xá)"
}

# Không gian hành động: 4 hướng di chuyển
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTIONS = {
    ACTION_UP: (-1, 0),     # Lên (giảm hàng)
    ACTION_DOWN: (1, 0),    # Xuống (tăng hàng)
    ACTION_LEFT: (0, -1),   # Trái (giảm cột)
    ACTION_RIGHT: (0, 1)    # Phải (tăng cột)
}

ACTION_NAMES = {
    ACTION_UP: "LÊN (UP)",
    ACTION_DOWN: "XUỐNG (DOWN)",
    ACTION_LEFT: "TRÁI (LEFT)",
    ACTION_RIGHT: "PHẢI (RIGHT)"
}

ACTION_SYMBOLS = {
    ACTION_UP: "↑",
    ACTION_DOWN: "↓",
    ACTION_LEFT: "←",
    ACTION_RIGHT: "→"
}

# Cấu hình Phần thưởng (Reward Shaping cố định theo Báo cáo Đề tài Mục 3.2.4)
# TUÂN THỦ CHÍNH XÁC THEO TÀI LIỆU DOCS NHÓM TRƯỞNG HOA ĐINH YÊU CẦU:
REWARD_STEP = -1.0           # Chi phí năng lượng / thời gian mỗi bước di chuyển
REWARD_COLLISION = -10.0     # Phạt va chạm vật cản hoặc vượt biên lưới (xe giữ nguyên vị trí)
REWARD_DELIVERY = 50.0       # Thưởng khi giao hàng thành công lần đầu tại mỗi điểm (R = +50)
REWARD_TERMINAL = 100.0      # Thưởng lớn khi hoàn tất toàn bộ điểm và quay về kho an toàn (R = +100)

# Siêu tham số cố định cho Thuật toán Policy Iteration (Mục 5.1 Báo cáo)
GAMMA = 0.99                 # Hệ số chiết khấu phần thưởng tương lai (cố định)
THETA = 1e-4                 # Ngưỡng hội tụ của giai đoạn Policy Evaluation (cố định)
MAX_EVAL_ITERS = 1000        # Số vòng lặp tối đa cho mỗi lần Policy Evaluation
MAX_PI_ITERS = 100           # Số lần cải thiện chính sách tối đa

# =============================================================================
# CẤU HÌNH CÁC THƯ MỤC ĐẦU RA (OUTPUT DIRECTORIES)
# Gom và phân loại rạch ròi các kết quả xuất ra:
# - outputs/figures/    : Các hình ảnh biểu đồ, sơ đồ lộ trình, bản đồ nhiệt
# - outputs/animations/ : Các file hoạt ảnh mô phỏng xe chạy (.gif)
# - outputs/data/       : Các file dữ liệu nhật ký bước đi, số liệu hội tụ (.csv)
# =============================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
ANIMATIONS_DIR = os.path.join(OUTPUTS_DIR, "animations")
DATA_DIR = os.path.join(OUTPUTS_DIR, "data")

# Tự động khởi tạo cấu trúc thư mục nếu chưa có
for _dir in [OUTPUTS_DIR, FIGURES_DIR, ANIMATIONS_DIR, DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)
