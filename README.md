# ỨNG DỤNG HỌC TĂNG CƯỜNG TRONG TỐI ƯU HÓA LỘ TRÌNH GIAO HÀNG
**Môn học:** Học tăng cường và Ứng dụng - Nhóm 8 - ĐH Thủy Lợi  
**Phụ trách Môi trường mô phỏng:** Vương Tiến Dũng & Vũ Hải Đăng  
**Phụ trách Thuật toán Policy Iteration:** Vũ Hải Đăng (Module độc lập)

---

## 📌 CÁC FILE CHẠY CHÍNH

### 1. File chạy Môi trường độc lập (Phần việc nộp cho nhóm):
```bash
python run_env.py
```
* **Chức năng:** Khởi tạo môi trường Lưới 12x12, kiểm thử các hàm Gymnasium (`reset`, `step`), mở cửa sổ đồ họa hiển thị **Bản đồ Lưới 12x12** và **BẢNG CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG** chi tiết.
* **File ảnh xuất ra:** `outputs/figures/environment_layout_12x12.png`.

### 2. Giao diện Web tương tác Streamlit (Cố định tham số theo Docs):
```bash
streamlit run app_streamlit.py
```
* **Chức năng:** Giao diện Web có 2 chế độ:
  - **Chế độ 1:** Khảo sát môi trường độc lập & 4 nút điều khiển xe di chuyển thủ công.
  - **Chế độ 2:** Huấn luyện thuật toán Policy Iteration & Phân tích lộ trình.
  - Các tham số hàm thưởng và mô hình được **cố định chính xác theo tài liệu Docs**, không có thanh trượt tùy chỉnh lung tung.

### 3. File chạy Thuật toán Policy Iteration (Để riêng của Đăng):
```bash
python run_policy_iteration.py
```
* **Chức năng:** Huấn luyện thuật toán Policy Iteration, trích xuất lộ trình 26 bước tối ưu (Reward: +224.0 điểm).
