# BÁO CÁO KỸ THUẬT VÀ HƯỚNG DẪN THỰC THI DỰ ÁN
## ĐỀ TÀI: ỨNG DỤNG HỌC TĂNG CƯỜNG TRONG TỐI ƯU HÓA LỘ TRÌNH GIAO HÀNG TRÊN LƯỚI 12x12
**Môn học:** Học tăng cường và Ứng dụng  
**Giảng viên phụ trách:** TS. Lê Nguyễn Tuấn Thành  
**Nhóm thực hiện:** Nhóm 8 - Khoa Công nghệ Thông tin - Trường Đại học Thủy Lợi  
**Thành viên triển khai module:**
- **Vũ Hải Đăng** (MSV: 2351170580) - Phụ trách Thuật toán Policy Iteration, Phân tích Hội tụ & Visualizer
- **Vương Tiến Dũng** (MSV: 2351170586) - Phụ trách Xây dựng Môi trường Mô phỏng Lưới 12x12 (Environment)

---

## 1. CẤU TRÚC THƯ MỤC PROJECT VÀ PHÂN CHIA OUTPUT RẠCH RÒI

Toàn bộ mã nguồn và các file kết quả đầu ra (output) đã được tổ chức phân tầng module hóa chuẩn mực:

```text
delivery_grid_rl/
│
├── config.py                       # Cấu hình lưới 12x12, kho, điểm giao, vật cản, tên các tòa nhà, đường dẫn output
├── requirements.txt                # Danh sách thư viện: gymnasium, numpy, matplotlib, pandas, seaborn, streamlit
├── main.py                         # File thực thi chính (Huấn luyện, trích xuất lộ trình, xuất đồ họa Dual-Panel)
├── convergence_analysis.py         # File phân tích chuyên sâu (Xuất Heatmap V(s) 4 giai đoạn & Đồ thị hội tụ)
├── app_streamlit.py                # Ứng dụng Web tương tác Streamlit chuyên nghiệp
├── HUONG_DAN_CHI_TIET.md           # Tài liệu hướng dẫn sử dụng, giải thích toán và bảo vệ đề tài
│
├── environment/                    # PACKAGE MÔI TRƯỜNG (VƯƠNG TIẾN DŨNG & VŨ HẢI ĐĂNG)
│   ├── __init__.py
│   └── delivery_env.py             # Class DeliveryGridEnv chuẩn Gymnasium, 872 trạng thái MDP
│
├── algorithms/                     # PACKAGE THUẬT TOÁN (VŨ HẢI ĐĂNG)
│   ├── __init__.py
│   └── policy_iteration.py        # Thuật toán Policy Iteration với Bellman Evaluation & Improvement
│
├── visualization/                  # PACKAGE ĐỒ HỌA CAO CẤP
│   ├── __init__.py
│   └── visualizer.py               # Lớp GridVisualizer: Dual-Panel HUD, xe đa giác, chuyển động mượt mà
│
└── outputs/                        # THƯ MỤC CHỨA TẤT CẢ KẾT QUẢ ĐẦU RA (ĐƯỢC PHÂN LOẠI RẠCH RÒI)
    │
    ├── figures/                    # [1] CÁC HÌNH ẢNH BIỂU ĐỒ & SƠ ĐỒ ĐỘ NÉT CAO (DÙNG CHO BÁO CÁO WORD)
    │   ├── optimal_route_12x12.png         # Sơ đồ lộ trình tối ưu 2 khung hình (Bản đồ khuôn viên + Telemetry HUD)
    │   ├── value_function_heatmaps.png     # 4 Bản đồ nhiệt hàm giá trị trạng thái V(s) qua 4 giai đoạn
    │   └── convergence_curves.png          # Đồ thị số lượt quét Bellman và tốc độ ổn định chính sách qua 19 vòng lặp
    │
    ├── animations/                 # [2] CÁC FILE HOẠT ẢNH CHUYỂN ĐỘNG (DÙNG CHO SLIDE POWERPOINT)
    │   └── delivery_animation.gif          # File GIF mô phỏng xe chạy lướt êm ái trên lưới, đèn pha xoay theo khúc cua
    │
    └── data/                       # [3] CÁC FILE DỮ LIỆU & BẢNG SỐ LIỆU THỰC NGHIỆM (.CSV)
        ├── route_steps_log.csv             # Bảng nhật ký chi tiết 26 bước di chuyển (tọa độ, hành động, reward, mask)
        ├── algorithm_benchmark_summary.csv # Bảng tổng hợp các chỉ số đánh giá thuật toán phục vụ Báo cáo
        └── policy_iteration_convergence_metrics.csv # Bảng theo dõi chi tiết độ hội tụ qua từng vòng lặp
```

---

## 2. HƯỚNG DẪN CÀI ĐẶT VÀ CÁC CÁCH CHẠY CHƯƠNG TRÌNH

### 2.1. Cài đặt thư viện
Tại thư mục `delivery_grid_rl/`, chạy lệnh:
```bash
pip install -r requirements.txt
```

### 2.2. Cách 1: Chạy chương trình chính (Xuất ảnh độ nét cao & Bảng dữ liệu)
```bash
python delivery_grid_rl/main.py
```
*Kết quả:* Huấn luyện Policy Iteration xong trong 2.5s, tự động xuất:
- Ảnh lộ trình: `outputs/figures/optimal_route_12x12.png`
- Bảng nhật ký: `outputs/data/route_steps_log.csv`
- Bảng tóm tắt: `outputs/data/algorithm_benchmark_summary.csv`

### 2.3. Cách 2: Chạy xem Hoạt ảnh xe chuyển động MƯỢT MÀ trên cửa sổ đồ họa
```bash
python delivery_grid_rl/main.py --animate
```
*Cửa sổ Matplotlib: Xe giao hàng ái qua từng ô, quay theo khúc cua, các ô đã đi đổi màu xanh, bảng Telemetry bên phải nhảy thông số thời gian thực.*

### 2.4. Cách 3: Xuất file GIF hoạt ảnh phục vụ Slide PowerPoint
```bash
python delivery_grid_rl/main.py --save-gif
```
*Tạo file GIF tại `outputs/animations/delivery_animation.gif`.*

### 2.5. Cách 4: Chạy Phân tích Chuyên sâu (Heatmaps & Đồ thị hội tụ)
```bash
python delivery_grid_rl/convergence_analysis.py
```
*Tạo 2 file ảnh phân tích sâu*
- `outputs/figures/value_function_heatmaps.png`
- `outputs/figures/convergence_curves.png`
- `outputs/data/policy_iteration_convergence_metrics.csv`

### 2.6. Cách 5: Khởi chạy Ứng dụng Web Tương tác Streamlit
```bash
streamlit run delivery_grid_rl/app_streamlit.py
```
*Mở giao diện web trên trình duyệt: Kéo slider để tua từng bước xe chạy, xem Heatmap, tra cứu dữ liệu và tải báo cáo CSV.*

---

## 3. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH HỌC THUẬT

### 3.1. Bảng số liệu tổng hợp
- **Kích thước không gian trạng thái:** $872$ trạng thái rời rạc ($109$ ô đường đi hợp lệ $\times 8$ trạng thái bitmask giao hàng).
- **Số vòng lặp hội tụ:** $19$ vòng lặp Policy Iteration.
- **Thời gian tính toán:** $\approx 2.55$ giây trên CPU.
- **Tổng quãng đường di chuyển:** **26 bước** (Đạt mức tối ưu tuyệt đối theo khoảng cách Manhattan trong mê cung có 35 ô vật cản).
- **Tổng phần thưởng (Cumulative Reward):** **$+134.00$ điểm** ($-26$ chi phí bước $+ 60$ thưởng 3 điểm giao $+ 100$ thưởng về kho an toàn).
- **Thứ tự giao hàng tối ưu:** Kho $(3, 1) \rightarrow C_2 (7, 4) \rightarrow C_3 (7, 9) \rightarrow C_1 (2, 9) \rightarrow$ Kho $(3, 1)$.

### 3.2. Ý nghĩa của Bản đồ nhiệt Hàm giá trị $V(s)$ (Value Function Heatmaps)
Khi mở ảnh `outputs/figures/value_function_heatmaps.png`:
1. **Giai đoạn 1 (Chưa giao khách nào):** Vùng xung quanh điểm $C_2 (7, 4)$ có màu vàng sáng (giá trị $V(s)$ cao nhất $\approx 134$). Các ô càng xa điểm $C_2$ có màu tím tối dần. Điều này chứng minh tác tử đã học được rằng mục tiêu tối ưu đầu tiên là phải đi tới $C_2$.
2. **Giai đoạn 2 (Đã giao $C_2$):** Vùng sáng cao nhất lập tức chuyển dịch sang điểm $C_3 (7, 9)$. Tác tử đi ngang qua hành lang phía dưới để tiếp cận $C_3$.
3. **Giai đoạn 3 (Đã giao $C_2$ & $C_3$):** Vùng sáng cao nhất chuyển lên phía trên về điểm $C_1 (2, 9)$.
4. **Giai đoạn 4 (Đã giao đủ cả 3):** Toàn bộ bản đồ dồn gradient ánh sáng về lại Kho hàng Depot $(3, 1)$ với phần thưởng về đích lớn nhất $+100$.

---