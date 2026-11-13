"""
=============================================================================
GIAO DIỆN MÔ PHỎNG VÀ TRỰC QUAN HÓA MÔI TRƯỜNG LƯỚI 12x12 (STREAMLIT)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên: Vương Tiến Dũng (MSV: 2351170586) & Vũ Hải Đăng (MSV: 2351170580)
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
ĐÃ ĐƯỢC CHỈNH SỬA THEO GÓP Ý CỦA NHÓM TRƯỞNG HOA ĐINH:
1. Môi trường độc lập hoàn toàn, không mặc định gắn thuật toán Policy Iteration.
2. Các tham số mô hình và hàm phần thưởng là CỐ ĐỊNH theo đúng Docs (không để thanh trượt chỉnh lung tung).
3. Khung bên phải mặc định hiển thị BẢNG CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG (Kích thước, ô đen vật cản, ô trắng đường đi, tọa độ kho và 3 điểm giao).
4. Tách biệt phần Huấn luyện thuật toán sang chế độ riêng ("Khi nào gọi thuật toán thì mới ra thông tin đấy").
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from environment.delivery_env import DeliveryGridEnv
from visualization.visualizer import GridVisualizer

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Mô Phỏng Môi Trường Lưới 12x12 - Nhóm 8",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗺️ HỆ THỐNG MÔ PHỎNG MÔI TRƯỜNG GIAO HÀNG TRÊN LƯỚI 12x12")
st.markdown("""
**Học phần:** Học tăng cường và Ứng dụng | **Giảng viên hướng dẫn:** TS. Lê Nguyễn Tuấn Thành  
**Nhóm thực hiện:** Nhóm 8 | **Phụ trách Môi trường:** Vương Tiến Dũng & Vũ Hải Đăng
""")

# =============================================================================
# THANH BÊN (SIDEBAR): HIỂN THỊ CẤU HÌNH THÔNG SỐ CỐ ĐỊNH THEO TÀI LIỆU DOCS
# (Đã bỏ thanh trượt chỉnh tham số theo đúng yêu cầu của Hoa Đinh)
# =============================================================================
st.sidebar.header("📋 THÔNG SỐ CỐ ĐỊNH (THEO DOCS)")
st.sidebar.info("""
**Quy định bài toán:** Các thông số dưới đây được **cố định chuẩn xác** theo tài liệu Mô hình hóa MDP của nhóm, không cho phép tùy biến.
""")

st.sidebar.markdown(f"""
* **Kích thước bản đồ:** `{config.GRID_SIZE} x {config.GRID_SIZE}` (144 ô)
* **Kho hàng cố định (Depot):** `{config.DEPOT_POS}`
* **Điểm giao cố định:**
  - C1: `{config.CUSTOMER_POSITIONS[0]}`
  - C2: `{config.CUSTOMER_POSITIONS[1]}`
  - C3: `{config.CUSTOMER_POSITIONS[2]}`
* **Số ô vật cản (ô đen):** `{len(config.OBSTACLES)} ô`
* **Số ô đường đi (ô trắng):** `{config.GRID_SIZE*config.GRID_SIZE - len(config.OBSTACLES)} ô`
""")

st.sidebar.subheader("🎯 Hàm Phần Thưởng Cố Định")
st.sidebar.markdown(f"""
- Di chuyển mỗi bước: `R = {config.REWARD_STEP}`
- Đâm vật cản / vượt biên: `R = {config.REWARD_COLLISION}`
- Giao hàng thành công: `R = +{config.REWARD_DELIVERY}`
- Hoàn thành về Kho Depot: `R = +{config.REWARD_TERMINAL}`
""")

st.sidebar.subheader("⚙️ Siêu Tham Số RL Cố Định")
st.sidebar.markdown(f"""
- Hệ số chiết khấu: `γ = {config.GAMMA}`
- Ngưỡng hội tụ Bellman: `θ = {config.THETA}`
""")

# Lựa chọn Chế độ hiển thị
st.sidebar.markdown("---")
st.sidebar.subheader("🔘 CHẾ ĐỘ HIỂN THỊ")
mode = st.sidebar.radio(
    "Chọn nội dung hiển thị:",
    ["1. Môi Trường Mô Phỏng Độc Lập (Báo Cáo Nhóm)", "2. Kiểm Thử Thuật Toán Policy Iteration"],
    index=0
)

# Khởi tạo Visualizer
visualizer = GridVisualizer(
    grid_size=config.GRID_SIZE,
    depot_pos=config.DEPOT_POS,
    customer_positions=config.CUSTOMER_POSITIONS,
    obstacles=config.OBSTACLES,
    campus_buildings=config.CAMPUS_BUILDINGS
)

# =============================================================================
# CHẾ ĐỘ 1: MÔI TRƯỜNG MÔ PHỎNG ĐỘC LẬP (NỘI DUNG CHÍNH ĐỂ TRÌNH BÀY)
# =============================================================================
if mode.startswith("1."):
    st.header("1. KHẢO SÁT & TRỰC QUAN HÓA MÔ HÌNH MÔI TRƯỜNG ĐỘC LẬP")
    st.success("Môi trường được thiết kế độc lập theo chuẩn Gymnasium (`gym.Env`), sẵn sàng bàn giao cho các thành viên huấn luyện các thuật toán khác nhau (Policy Iteration, Monte Carlo, Q-Learning).")

    # 1. Khởi tạo State trong Session State trước khi tương tác
    if "manual_env" not in st.session_state:
        st.session_state.manual_env = DeliveryGridEnv()
        st.session_state.manual_obs, st.session_state.manual_info = st.session_state.manual_env.reset()
        st.session_state.manual_history = []
        st.session_state.visited_cells = {config.DEPOT_POS}
        st.session_state.last_action_dir = (1.0, 0.0)  # Hướng xe ban đầu: Xuống
        st.session_state.last_event_msg = "Xe đang sẵn sàng xuất phát tại Kho hàng (3, 1)"
        st.session_state.total_manual_reward = 0.0

    menv = st.session_state.manual_env

    # 2. Xử lý điều khiển tương tác (Bàn phím ảo điều khiển xe)
    st.subheader("🎮 Bảng Điều Khiển Di Chuyển Xe Trực Tiếp")
    st.caption("Bấm các nút bên dưới để điều khiển xe di chuyển qua các ô trên bản đồ và kiểm tra phản hồi của môi trường:")

    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
    action_taken = None
    act_name = ""
    dir_vector = (0.0, 0.0)

    with col_btn1:
        if st.button("⬅️ Đi TRÁI (LEFT)", use_container_width=True, key="btn_left"):
            action_taken = config.ACTION_LEFT
            act_name = "TRÁI"
            dir_vector = (0.0, -1.0)
    with col_btn2:
        if st.button("⬆️ Đi LÊN (UP)", use_container_width=True, key="btn_up"):
            action_taken = config.ACTION_UP
            act_name = "LÊN"
            dir_vector = (-1.0, 0.0)
    with col_btn3:
        if st.button("⬇️ Đi XUỐNG (DOWN)", use_container_width=True, key="btn_down"):
            action_taken = config.ACTION_DOWN
            act_name = "XUỐNG"
            dir_vector = (1.0, 0.0)
    with col_btn4:
        if st.button("➡️ Đi PHẢI (RIGHT)", use_container_width=True, key="btn_right"):
            action_taken = config.ACTION_RIGHT
            act_name = "PHẢI"
            dir_vector = (0.0, 1.0)
    with col_btn5:
        if st.button("🔄 ĐẶT LẠI VỀ KHO (RESET)", use_container_width=True, type="secondary", key="btn_reset"):
            menv.reset()
            st.session_state.manual_history = []
            st.session_state.visited_cells = {config.DEPOT_POS}
            st.session_state.last_action_dir = (1.0, 0.0)
            st.session_state.last_event_msg = "Xe đã được đặt lại về Kho hàng Depot (3, 1)"
            st.session_state.total_manual_reward = 0.0
            st.rerun()

    # Thực thi bước di chuyển nếu người dùng bấm nút
    if action_taken is not None:
        prev_pos = menv.current_state[:2]
        prev_mask = menv.current_state[2]
        o, r, term, trunc, inf = menv.step(action_taken)
        curr_pos = inf["state_tuple"][:2]
        curr_mask = inf["state_tuple"][2]

        st.session_state.visited_cells.add(curr_pos)
        st.session_state.last_action_dir = dir_vector
        st.session_state.total_manual_reward += r

        # Xác định thông báo sự kiện
        if r == config.REWARD_COLLISION:
            st.session_state.last_event_msg = f"⚠️ Va chạm vật cản / biên tại {prev_pos}! Xe bị giữ nguyên vị trí, phạt {r:.0f}đ"
        elif curr_mask == config.TARGET_MASK and curr_pos == config.DEPOT_POS:
            st.session_state.last_event_msg = f"🎉 HOÀN THÀNH NHIỆM VỤ! Xe đã giao hết 3 điểm và về Kho thành công (+{config.REWARD_TERMINAL:.0f}đ)!"
        elif curr_mask > prev_mask:
            delivered_idx = [k for k in range(config.NUM_CUSTOMERS) if (curr_mask & (1 << k)) and not (prev_mask & (1 << k))]
            c_name = f"C{delivered_idx[0]+1}" if delivered_idx else "khách hàng"
            st.session_state.last_event_msg = f"★ GIAO HÀNG THÀNH CÔNG CHO {c_name} tại {curr_pos}! Thưởng +{config.REWARD_DELIVERY:.0f}đ"
        else:
            st.session_state.last_event_msg = f"Xe di chuyển {act_name} tới {curr_pos} (Chi phí bước: {r:.0f}đ)"

        st.session_state.manual_history.append((act_name, curr_pos, r))
        st.rerun()

    # 3. Lấy trạng thái hiện tại của xe để vẽ lên bản đồ thời gian thực
    curr_r, curr_c, curr_mask = menv.current_state
    delivered_count = bin(curr_mask).count("1")
    remaining_count = config.NUM_CUSTOMERS - delivered_count
    dr_dir, dc_dir = st.session_state.last_action_dir

    # Hiển thị thông báo trạng thái
    if "Va chạm" in st.session_state.last_event_msg:
        st.warning(st.session_state.last_event_msg)
    elif "HOÀN THÀNH" in st.session_state.last_event_msg or "GIAO HÀNG THÀNH CÔNG" in st.session_state.last_event_msg:
        st.success(st.session_state.last_event_msg)
    else:
        st.info(st.session_state.last_event_msg)

    # Hiển thị các metric thời gian thực của xe
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vị trí xe hiện tại", f"({curr_r}, {curr_c})", f"Đã đi {menv.step_count} bước")
    c2.metric("Tiến độ giao hàng", f"{delivered_count} / {config.NUM_CUSTOMERS} điểm", f"Còn lại: {remaining_count}")
    c3.metric("Tổng phần thưởng", f"{st.session_state.total_manual_reward:+.1f} điểm", "Theo quy tắc Docs")
    c4.metric("Trạng thái bitmask", f"{bin(curr_mask)}", "S = (x, y, mask)")

    # 4. Vẽ Bản đồ môi trường cập nhật ĐỘNG theo từng bước di chuyển của xe
    fig = plt.figure(figsize=(15, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.15)
    ax_map = fig.add_subplot(gs[0])
    ax_hud = fig.add_subplot(gs[1])

    # Vẽ bản đồ khuôn viên có cập nhật các ô đã đi và mask giao hàng
    visualizer.draw_campus_map(ax_map, visited_cells=st.session_state.visited_cells, mask=curr_mask)

    # Vẽ xe giao hàng tại vị trí THỰC TẾ với hướng nhìn chuẩn xác
    visualizer.draw_vehicle(ax_map, float(curr_c), float(curr_r), float(dc_dir), float(dr_dir))

    ax_map.set_title(
        f"BẢN ĐỒ MÔ HÌNH HÓA MÔI TRƯỜNG LƯỚI 12x12 (CẬP NHẬT THỜI GIAN THỰC)\n"
        f"Vị trí xe: ({curr_r}, {curr_c}) | Đã giao: {delivered_count}/3 khách | Bước: {menv.step_count}",
        fontsize=11.5, fontweight="bold", pad=12, color="#0f172a"
    )

    # Vẽ BẢNG CHÚ THÍCH MÔI TRƯỜNG bên phải
    visualizer.draw_environment_legend_hud(ax_hud)

    st.pyplot(fig)
    plt.close(fig)

    # Lịch sử di chuyển
    if st.session_state.manual_history:
        with st.expander("📜 Xem nhật ký các bước di chuyển vừa thực hiện"):
            st.write(" $\\rightarrow$ ".join([f"**{h[0]}** tới `{h[1]}` (R: {h[2]:+.0f})" for h in st.session_state.manual_history]))

# =============================================================================
# CHẾ ĐỘ 2: KIỂM THỬ THUẬT TOÁN POLICY ITERATION (ĐƯỢC TÁCH RIÊNG)
# =============================================================================
else:
    st.header("2. KẾT QUẢ THỰC NGHIỆM THUẬT TOÁN POLICY ITERATION")
    st.info("Phần này được tách độc lập: Chỉ khi nào chủ động gọi chạy thuật toán thì mới tính toán chính sách tối ưu, hiển thị Telemetry và biểu đồ hội tụ.")

    from algorithms.policy_iteration import PolicyIteration

    @st.cache_resource
    def run_pi():
        env_instance = DeliveryGridEnv()
        solver_instance = PolicyIteration(env_instance, gamma=config.GAMMA, theta=config.THETA)
        hist = solver_instance.train(verbose=False)
        route = solver_instance.extract_optimal_route(max_steps=100)
        return env_instance, solver_instance, hist, route

    env_pi, solver_pi, hist_pi, route_pi = run_pi()

    # Metric kết quả
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng Quãng Đường Tối Ưu", f"{route_pi['total_steps']} bước", "Tối ưu tuyệt đối")
    m2.metric("Tổng Phần Thưởng", f"{route_pi['total_reward']:.1f} điểm", "Đúng hàm thưởng mới")
    m3.metric("Số Vòng Lặp Hội Tụ", f"{hist_pi['pi_iterations']} vòng", "Ổn định chính sách")
    m4.metric("Thời Gian Huấn Luyện", f"{hist_pi['training_time_seconds']:.3f} s", "Tức thì trên CPU")

    st.markdown("---")

    tab_pi1, tab_pi2, tab_pi3 = st.tabs([
        "🎬 Hoạt Ảnh Mô Phỏng Lộ Trình",
        "📈 Phân Tích Độ Hội Tụ & Bản Đồ Nhiệt V(s)",
        "📋 Nhật Ký Chi Tiết Từng Bước Đi"
    ])

    with tab_pi1:
        st.subheader("Lộ Trình Tối Ưu và Bảng Telemetry Thời Gian Thực")
        step_val = st.slider("Kéo xem vị trí xe:", 0, route_pi["total_steps"], route_pi["total_steps"], 1)

        fig_pi = plt.figure(figsize=(15, 8.5))
        gs_pi = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.15)
        ax_m = fig_pi.add_subplot(gs_pi[0])
        ax_t = fig_pi.add_subplot(gs_pi[1])

        # Vẽ bản đồ có lộ trình
        coords = route_pi["route_coords"]
        states_s = route_pi["states_sequence"]
        visited = set(coords[:step_val + 1])
        c_pos = coords[step_val]
        c_mask = states_s[step_val][2]

        visualizer.draw_campus_map(ax_m, visited_cells=visited, mask=c_mask)

        # Hướng xe
        if step_val < len(coords) - 1:
            dr = coords[step_val + 1][0] - c_pos[0]
            dc = coords[step_val + 1][1] - c_pos[1]
        elif step_val > 0:
            dr = c_pos[0] - coords[step_val - 1][0]
            dc = c_pos[1] - coords[step_val - 1][1]
        else:
            dr, dc = 1, 0

        visualizer.draw_vehicle(ax_m, float(c_pos[1]), float(c_pos[0]), float(dc), float(dr))

        act_text = route_pi["action_names"][step_val - 1] if step_val > 0 else "XUẤT PHÁT"
        cum_rew = sum(route_pi["rewards"][:step_val])
        evt_txt = "Xe đang di chuyển theo lộ trình tối ưu..."
        if c_mask == 7 and c_pos == config.DEPOT_POS:
            evt_txt = "★ HOÀN TẤT VÀ VỀ KHO THÀNH CÔNG (+100đ)!"
        elif step_val > 0:
            for k, cp in enumerate(config.CUSTOMER_POSITIONS):
                if c_pos == cp and (c_mask & (1 << k)):
                    evt_txt = f"★ GIAO THÀNH CÔNG C{k+1} (+50đ)!"

        visualizer.draw_telemetry_hud(
            ax_t, step_val, route_pi["total_steps"], c_pos, act_text, c_mask, cum_rew, evt_txt
        )

        st.pyplot(fig_pi)
        plt.close(fig_pi)

        # Hiển thị file gif nếu có
        gif_file = os.path.join(config.ANIMATIONS_DIR, "delivery_animation.gif")
        if os.path.exists(gif_file):
            with st.expander("🎥 Xem Hoạt Ảnh Chuyển Động GIF Mượt Mà"):
                st.image(gif_file, use_container_width=True)

    with tab_pi2:
        st.subheader("Bản Đồ Nhiệt Hàm Giá Trị Trạng Thái V(s) Qua 4 Giai Đoạn")
        h1, h2 = st.columns(2)
        stage_masks = [
            (0, "Giai Đoạn 1: Chưa giao khách nào -> Hướng tới C2 (7, 4)"),
            (2, "Giai Đoạn 2: Đã giao C2 -> Hướng tới C3 (7, 9)"),
            (6, "Giai Đoạn 3: Đã giao C2 & C3 -> Hướng tới C1 (2, 9)"),
            (7, "Giai Đoạn 4: Đã giao đủ 3 điểm -> Quay về Kho (3, 1)")
        ]
        for idx, (m, stitle) in enumerate(stage_masks):
            grid_v = np.full((config.GRID_SIZE, config.GRID_SIZE), np.nan)
            for r in range(config.GRID_SIZE):
                for c in range(config.GRID_SIZE):
                    s = (r, c, m)
                    if s in env_pi.state_to_idx:
                        grid_v[r, c] = solver_pi.V[env_pi.state_to_idx[s]]

            fig_h, ax_h = plt.subplots(figsize=(6, 5))
            sns.heatmap(grid_v, ax=ax_h, cmap="viridis" if m != 7 else "plasma", cbar=True)
            ax_h.set_title(stitle, fontsize=9.5, fontweight="bold")
            
            target_col = h1 if idx % 2 == 0 else h2
            target_col.pyplot(fig_h)
            plt.close(fig_h)

    with tab_pi3:
        st.subheader("Bảng Nhật Ký 26 Bước Di Chuyển")
        steps_d = []
        for step_i in range(len(route_pi["actions_taken"])):
            steps_d.append({
                "Bước": step_i + 1,
                "Tọa độ hiện tại": f"({route_pi['route_coords'][step_i][0]}, {route_pi['route_coords'][step_i][1]})",
                "Hành động": route_pi["action_names"][step_i],
                "Tọa độ kế tiếp": f"({route_pi['route_coords'][step_i+1][0]}, {route_pi['route_coords'][step_i+1][1]})",
                "Phần thưởng": route_pi["rewards"][step_i],
                "Đã giao (Bitmask)": bin(route_pi["states_sequence"][step_i][2])
            })
        st.dataframe(pd.DataFrame(steps_d), use_container_width=True)
