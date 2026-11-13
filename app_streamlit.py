"""
=============================================================================
GIAO DIỆN TƯƠNG TÁC WEB STREAMLIT (INTERACTIVE RL DASHBOARD)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên: Vũ Hải Đăng (MSV: 2351170580) & Vương Tiến Dũng (MSV: 2351170586)
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
Cách chạy:
    streamlit run delivery_grid_rl/app_streamlit.py
=============================================================================
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import config
from environment.delivery_env import DeliveryGridEnv
from algorithms.policy_iteration import PolicyIteration
from visualization.visualizer import GridVisualizer

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Tối Ưu Lộ Trình Giao Hàng - Nhóm 8",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚚 HỆ THỐNG MÔ PHỎNG VÀ TỐI ƯU HÓA LỘ TRÌNH GIAO HÀNG TỰ HÀNH")
st.markdown("""
**Học phần:** Học tăng cường và Ứng dụng | **Giảng viên hướng dẫn:** TS. Lê Nguyễn Tuấn Thành  
**Nhóm thực hiện:** Nhóm 8 | **Thành viên:** Vũ Hải Đăng (Thuật toán Policy Iteration) & Vương Tiến Dũng (Môi trường)
""")

# Thanh bên điều khiển (Sidebar Controls)
st.sidebar.header("⚙️ CẤU HÌNH SIÊU THAM SỐ")

gamma_param = st.sidebar.slider("Hệ số chiết khấu (Gamma)", min_value=0.80, max_value=0.999, value=0.99, step=0.01)
theta_param = st.sidebar.select_slider("Ngưỡng hội tụ (Theta)", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, format_func=lambda x: f"{x:.0e}")

st.sidebar.subheader("🎯 Cấu hình Phần thưởng (Reward)")
reward_step_param = st.sidebar.number_input("Chi phí bước đi (R_step)", value=-1.0, step=0.5)
reward_collision_param = st.sidebar.number_input("Phạt va chạm (R_collision)", value=-10.0, step=1.0)
reward_delivery_param = st.sidebar.number_input("Thưởng giao hàng (R_delivery)", value=20.0, step=5.0)
reward_terminal_param = st.sidebar.number_input("Thưởng về kho (R_terminal)", value=100.0, step=10.0)

st.sidebar.markdown("---")
train_btn = st.sidebar.button("🚀 BẮT ĐẦU HUẤN LUYỆN POLICY ITERATION", type="primary")

# Caching mô hình huấn luyện
@st.cache_resource
def get_trained_model(gamma, theta, r_step, r_col, r_del, r_term):
    # Cập nhật tạm thời config
    config.GAMMA = gamma
    config.THETA = theta
    config.REWARD_STEP = r_step
    config.REWARD_COLLISION = r_col
    config.REWARD_DELIVERY = r_del
    config.REWARD_TERMINAL = r_term

    env = DeliveryGridEnv()
    solver = PolicyIteration(env, gamma=gamma, theta=theta)
    history = solver.train(verbose=False)
    route_result = solver.extract_optimal_route(max_steps=100)
    return env, solver, history, route_result

env, solver, history, route_result = get_trained_model(
    gamma_param, theta_param, reward_step_param, reward_collision_param, reward_delivery_param, reward_terminal_param
)

# Hiển thị số liệu tổng quan (Metric Cards)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tổng Quãng Đường", f"{route_result['total_steps']} bước", delta="Tối ưu tuyệt đối", delta_color="normal")
col2.metric("Tổng Phần Thưởng", f"{route_result['total_reward']:.1f} đ", delta="+160 thưởng", delta_color="normal")
col3.metric("Số Vòng Lặp Hội Tụ", f"{history['pi_iterations']} vòng", delta="Ổn định chính sách")
col4.metric("Thời Gian Huấn Luyện", f"{history['training_time_seconds']:.3f} s", delta="Tức thì")
col5.metric("Số Trạng Thái MDP", f"{env.num_states} states", delta="Hữu hạn")

st.markdown("---")

# Các Tab chức năng chính
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Mô Phỏng Di Chuyển Trực Quan",
    "📈 Phân Tích Hội Tụ & Bản Đồ Nhiệt V(s)",
    "📋 Nhật Ký 26 Bước Lộ Trình",
    "📖 Hướng Dẫn & Lý Thuyết MDP"
])

# TAB 1: MÔ PHỎNG DI CHUYỂN
with tab1:
    st.subheader("Bản Đồ Khuôn Viên Trường Đại Học Thủy Lợi (Lưới 12x12)")
    
    step_slider = st.slider(
        "Kéo thanh trượt để xem xe di chuyển từng bước:",
        min_value=0, max_value=route_result["total_steps"], value=route_result["total_steps"], step=1
    )

    # Lấy thông tin bước hiện tại
    route_coords = route_result["route_coords"]
    states_seq = route_result["states_sequence"]
    
    current_pos = route_coords[step_slider]
    current_state = states_seq[step_slider]
    current_mask = current_state[2]
    
    visited_cells = set(route_coords[:step_slider + 1])
    
    # Xác định hướng xe
    if step_slider < len(route_coords) - 1:
        dr = route_coords[step_slider + 1][0] - current_pos[0]
        dc = route_coords[step_slider + 1][1] - current_pos[1]
    elif step_slider > 0:
        dr = current_pos[0] - route_coords[step_slider - 1][0]
        dc = current_pos[1] - route_coords[step_slider - 1][1]
    else:
        dr, dc = 1, 0

    visualizer = GridVisualizer(
        grid_size=env.grid_size,
        depot_pos=env.depot_pos,
        customer_positions=env.customer_positions,
        obstacles=env.obstacles,
        campus_buildings=config.CAMPUS_BUILDINGS
    )

    # Vẽ Figure 2 khung hình
    fig, (ax_map, ax_hud) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.7, 1.0]})
    visualizer.draw_campus_map(ax_map, visited_cells=visited_cells, mask=current_mask)
    visualizer.draw_vehicle(ax_map, current_pos[1], current_pos[0], dc, dr)

    # Hành động hiện tại
    act_str = route_result["action_names"][step_slider - 1] if step_slider > 0 else "XUẤT PHÁT TẠI KHO"
    curr_reward = sum(route_result["rewards"][:step_slider])

    evt = "Xe đang di chuyển giao hàng..."
    if current_mask == 7 and current_pos == env.depot_pos:
        evt = "★ HOÀN TẤT VÀ VỀ KHO THÀNH CÔNG (+100đ)!"
    elif step_slider > 0:
        for k, cpos in enumerate(env.customer_positions):
            if current_pos == cpos and (current_mask & (1 << k)):
                evt = f"★ GIAO THÀNH CÔNG ĐIỂM C{k+1} (+20đ)!"

    visualizer.draw_telemetry_hud(
        ax_hud, step_slider, route_result["total_steps"], current_pos, act_str, current_mask, curr_reward, evt
    )

    st.pyplot(fig)
    plt.close(fig)

    # Hiển thị file GIF hoạt ảnh mượt mà nếu có sẵn trong outputs/animations/
    gif_file = os.path.join(config.ANIMATIONS_DIR, "delivery_animation.gif")
    if os.path.exists(gif_file):
        with st.expander("🎥 Xem Hoạt Ảnh Chuyển Động GIF Mượt Mà (Full Animation)"):
            st.image(gif_file, caption="Hoạt ảnh mô phỏng xe giao hàng chuyển động mượt mà trên lưới 12x12", use_container_width=True)

# TAB 2: PHÂN TÍCH HỘI TỤ & HEATMAPS
with tab2:
    st.subheader("Bản Đồ Nhiệt Hàm Giá Trị Trạng Thái V(s) Qua 4 Giai Đoạn")
    st.info("Bản đồ nhiệt thể hiện rõ gradient giá trị kỳ vọng (Expected Return) của phương trình Bellman: Tác tử luôn di chuyển theo hướng có giá trị V(s) cao nhất để đến khách hàng tiếp theo!")

    col_h1, col_h2 = st.columns(2)
    stage_masks = [
        (0, "Giai Đoạn 1: Xuất phát tại Kho (Chưa giao khách nào)\nMục tiêu: Điểm C2 (7, 4)"),
        (2, "Giai Đoạn 2: Đã giao C2 (7, 4)\nMục tiêu: Điểm C3 (7, 9)"),
        (6, "Giai Đoạn 3: Đã giao C2 & C3\nMục tiêu: Điểm C1 (2, 9)"),
        (7, "Giai Đoạn 4: Đã giao xong tất cả 3 điểm\nMục tiêu: Về Kho Depot (3, 1)")
    ]

    for idx, (m, stage_title) in enumerate(stage_masks):
        grid_v = np.full((config.GRID_SIZE, config.GRID_SIZE), np.nan)
        for r in range(config.GRID_SIZE):
            for c in range(config.GRID_SIZE):
                s = (r, c, m)
                if s in env.state_to_idx:
                    grid_v[r, c] = solver.V[env.state_to_idx[s]]

        fig_h, ax_h = plt.subplots(figsize=(6, 5))
        sns.heatmap(grid_v, ax=ax_h, cmap="viridis" if m != 7 else "plasma", cbar=True)
        ax_h.set_title(stage_title, fontsize=10, fontweight="bold")
        
        target_col = col_h1 if idx % 2 == 0 else col_h2
        target_col.pyplot(fig_h)
        plt.close(fig_h)

    st.subheader("Đồ Thị Hội Tụ Thuật Toán Policy Iteration")
    c1, c2 = st.columns(2)
    
    iters = list(range(1, history["pi_iterations"] + 1))
    fig_c1, ax_c1 = plt.subplots(figsize=(6, 3.5))
    ax_c1.plot(iters, history["eval_sweeps_per_iter"], marker="o", color="#2563eb")
    ax_c1.set_title("Số Lượt Quét Bellman (Sweeps)", fontsize=10, fontweight="bold")
    ax_c1.set_xlabel("Vòng lặp PI")
    ax_c1.grid(True, linestyle="--", alpha=0.6)
    c1.pyplot(fig_c1)
    plt.close(fig_c1)

    fig_c2, ax_c2 = plt.subplots(figsize=(6, 3.5))
    ax_c2.plot(iters, history["policy_changes_per_iter"], marker="s", color="#dc2626")
    ax_c2.set_title("Số Trạng Thái Thay Đổi Hành Động", fontsize=10, fontweight="bold")
    ax_c2.set_xlabel("Vòng lặp PI")
    ax_c2.grid(True, linestyle="--", alpha=0.6)
    c2.pyplot(fig_c2)
    plt.close(fig_c2)

# TAB 3: NHẬT KÝ CHI TIẾT
with tab3:
    st.subheader("Bảng Nhật Ký Chi Tiết 26 Bước Di Chuyển Của Xe")
    steps_data = []
    for step_i in range(len(route_result["actions_taken"])):
        curr_p = route_result["route_coords"][step_i]
        next_p = route_result["route_coords"][step_i + 1]
        act_name = route_result["action_names"][step_i]
        rew = route_result["rewards"][step_i]
        mask_val = route_result["states_sequence"][step_i][2]
        steps_data.append({
            "Bước": step_i + 1,
            "Tọa độ hiện tại": f"({curr_p[0]}, {curr_p[1]})",
            "Hành động": act_name,
            "Tọa độ tiếp theo": f"({next_p[0]}, {next_p[1]})",
            "Phần thưởng": rew,
            "Đã giao (Bitmask)": f"{bin(mask_val)}"
        })
    df_steps = pd.DataFrame(steps_data)
    st.dataframe(df_steps, use_container_width=True)

    csv_bytes = df_steps.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Tải xuống file CSV nhật ký lộ trình",
        data=csv_bytes,
        file_name="nhat_ky_lo_trinh_26_buoc.csv",
        mime="text/csv"
    )

# TAB 4: HƯỚNG DẪN & LÝ THUYẾT
with tab4:
    st.markdown("""
    ### Mô Hình Toán Học Markov Decision Process (MDP)
    - **Không gian trạng thái $S$:** $S = (r, c, \\text{visited\\_mask})$ gồm 872 trạng thái hợp lệ.
    - **Không gian hành động $A$:** 4 hướng (Lên, Xuống, Trái, Phải).
    - **Hàm phần thưởng $R$:** Phạt bước đi $-1$, phạt va chạm $-10$, thưởng giao hàng $+20$, thưởng hoàn thành $+100$.
    - **Phương trình Bellman cho Policy Iteration:**
      $$V_{k+1}(s) = R(s, \\pi(s), s') + \\gamma V_k(s')$$
      $$\\pi_{new}(s) = \\arg\\max_{a \\in A} \\left[ R(s, a, s') + \\gamma V(s') \\right]$$
    """)
