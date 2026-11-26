# BÁO CÁO KỸ THUẬT VÀ HƯỚNG DẪN THỰC THI DỰ ÁN (BẢN CHUẨN THEO YÊU CẦU NHÓM)
## ĐỀ TÀI: ỨNG DỤNG HỌC TĂNG CƯỜNG TRONG TỐI ƯU HÓA LỘ TRÌNH GIAO HÀNG TRÊN LƯỚI 12x12
**Môn học:** Học tăng cường và Ứng dụng  
**Giảng viên phụ trách:** TS. Lê Nguyễn Tuấn Thành  
**Nhóm thực hiện:** Nhóm 8 - Khoa Công nghệ Thông tin - Trường Đại học Thủy Lợi  
**Thành viên phụ trách:**
- **Vương Tiến Dũng** (MSV: 2351170586) & **Vũ Hải Đăng** (MSV: 2351170580) - Phụ trách Xây dựng Môi trường Mô phỏng Lưới 12x12 (Environment)
- **Vũ Hải Đăng** - Phụ trách Module Thuật toán Policy Iteration (Tách độc lập)

---

## 1. CÁC ĐIỀU CHỈNH

Căn cứ vào trao đổi trong nhóm lúc 23:22 - 23:38 ngày 13/11:
1. **Tách biệt Môi trường mô phỏng (Environment) và Thuật toán (Policy Iteration):**
   - Phần Môi trường được đóng gói độc lập trong file `run_env.py`. Khi chạy, màn hình hoàn toàn không dính líu thông tin của thuật toán Policy Iteration.
   - Thuật toán Policy Iteration được cất riêng trong file `run_policy_iteration.py` (hoặc module `algorithms/`), chỉ hiển thị khi có lệnh gọi chạy thuật toán.

2. **BẢNG CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG:**
   - Cột bên phải của đồ họa môi trường giờ đây là **Bảng chú thích môi trường** chi tiết:
     - Kích thước lưới: Lưới $12 \times 12$ (144 ô vuông).
     - Quy ước ô: Ô đen/xám là vật cản (35 ô công trình khuôn viên), ô trắng là đường đi hợp lệ (109 ô).
     - Kho hàng cố định (Depot): Tọa độ $(3, 1)$ [Ký hiệu `D` màu xanh dương].
     - 3 Điểm giao hàng cố định: $C_1(2, 9), C_2(7, 4), C_3(7, 9)$ [Ký hiệu ngôi sao cam].
     - Không gian MDP: $S = (x, y, \text{visited\_mask})$ với 872 trạng thái, Hành động $A = \{0: \text{Lên}, 1: \text{Xuống}, 2: \text{Trái}, 3: \text{Phải}\}$.
     - Hàm phần thưởng cố định theo Docs.

3. **Các siêu tham số và hàm phần thưởng**
   - Không cho phép người dùng tự điều chỉnh lung tung qua thanh trượt.
   - Các giá trị được cố định chính xác theo ảnh chụp Docs:
     * Chi phí bước đi: $R = -1$
     * Phạt va chạm vật cản / biên: $R = -10$
     * Thưởng giao hàng thành công: **$R = +50$** (đã chỉnh đúng từ $+20$ thành $+50$)
     * Thưởng hoàn thành về Kho: $R = +100$
     * $\gamma = 0.99, \theta = 10^{-4}$

---

## 2. CẤU TRÚC PROJECT VÀ SỰ PHÂN CHIA ĐỘC LẬP

```text
delivery_grid_rl/
│
├── config.py                       # Cấu hình CỐ ĐỊNH lưới 12x12, kho, điểm giao, vật cản, reward theo Docs
├── requirements.txt                # gymnasium, numpy, matplotlib, pandas, seaborn, streamlit
│
├── run_env.py                      # [FILE NỘP SÁNG MAI CHO NHÓM] Chạy & trực quan hóa Môi trường độc lập
├── run_policy_iteration.py         # [FILE THUẬT TOÁN ĐỂ RIÊNG] Huấn luyện & đánh giá Policy Iteration
├── main.py                         # File thực thi tổng hợp
├── convergence_analysis.py         # Phân tích chuyên sâu (Bản đồ nhiệt V(s) & Biểu đồ hội tụ)
├── app_streamlit.py                # Web Dashboard tương tác (đã tách 2 chế độ độc lập và cố định tham số)
├── HUONG_DAN_CHI_TIET.md           # Báo cáo kỹ thuật chi tiết
│
├── environment/                    # Package Môi trường Gymnasium (Vương Tiến Dũng & Vũ Hải Đăng)
│   ├── __init__.py
│   └── delivery_env.py             # Class DeliveryGridEnv chuẩn Gymnasium, 872 trạng thái MDP
│
├── algorithms/                     # Package Thuật toán (Vũ Hải Đăng - để riêng độc lập)
│   ├── __init__.py
│   └── policy_iteration.py        # Thuật toán Policy Iteration thuần túy
│
├── visualization/                  # Package Đồ họa trực quan
│   ├── __init__.py
│   └── visualizer.py               # GridVisualizer hỗ trợ vẽ Môi trường & vẽ Lộ trình
│
└── outputs/                        # Thư mục chứa các kết quả đầu ra phân loại rạch ròi
    ├── figures/                    # Ảnh sơ đồ môi trường và ảnh đồ thị
    │   ├── environment_layout_12x12.png    # [Ảnh nộp cho nhóm] Bản đồ môi trường + Bảng chú thích
    │   ├── optimal_route_12x12.png         # Sơ đồ lộ trình khi chạy thuật toán
    │   ├── value_function_heatmaps.png     # 4 Bản đồ nhiệt hàm giá trị V(s)
    │   └── convergence_curves.png          # Đồ thị hội tụ PI
    ├── animations/                 # Hoạt ảnh xe chạy mượt mà (.gif)
    │   └── delivery_animation.gif
    └── data/                       # Dữ liệu bảng số liệu (.csv)
        ├── route_steps_log.csv
        ├── algorithm_benchmark_summary.csv
        └── policy_iteration_convergence_metrics.csv
```

---

## 3. HƯỚNG DẪN TRÌNH BÀY

### 3.1. Trình bày Môi trường mô phỏng độc lập:
Chạy lệnh PowerShell:
```bash
python run_env.py
```
* **Output:**
  * Terminal in đầy đủ thông số môi trường, test thử gọi hàm `reset()`, `step(DOWN)`, `step(RIGHT)`, và test cơ chế đâm vật cản bị giữ nguyên vị trí kèm phạt va chạm $-10$.
  * Cửa sổ đồ họa mở ra: Bên trái là **Bản đồ Lưới 12x12** (có Kho `D`, 3 Khách hàng, 35 ô vật cản tòa nhà), bên phải là **BẢNG CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG** chi tiết.
  * Tự động lưu file ảnh tại: `outputs/figures/environment_layout_12x12.png`.

### 3.2. Mở Web Streamlit để demo trực tiếp:
```bash
streamlit run demo_app.py
```
* **Hiện tượng:**
  * Thanh bên (Sidebar) hiển thị toàn bộ thông số cố định theo Docs.
  * Chế độ 1 mặc định: Hiển thị Bản đồ môi trường và Bảng chú thích môi trường, kèm 4 nút bấm điều khiển xe di chuyển thủ công để test môi trường.
  * Khi muốn xem thuật toán thì chuyển sang Chế độ 2.

### 3.3. Khi cần chạy và kiểm thử Thuật toán Policy Iteration:
```bash
python run_policy_iteration.py
```
* Chạy huấn luyện PI, trích xuất lộ trình 26 bước tối ưu (Tổng reward: $+224.0$ điểm theo hàm thưởng mới: $R_{del} = +50$).
