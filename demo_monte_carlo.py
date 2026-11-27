"""Giao diện Streamlit riêng để huấn luyện và trực quan hóa Monte Carlo."""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config
from algorithms.monte_carlo import OnPolicyFirstVisitMC
from environment.delivery_env import DeliveryGridEnv
from visualization.visualizer import GridVisualizer


st.set_page_config(
    page_title="On-policy First-visit Monte Carlo - Nhóm 8",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ ON-POLICY FIRST-VISIT MONTE CARLO")
st.markdown(
    "Mỗi episode sử dụng bố trí ngẫu nhiên **3–5 điểm giao hàng**. "
    "Ứng dụng này chạy độc lập với `demo_app.py`."
)

with st.sidebar:
    st.header("⚙️ Tham số huấn luyện")
    mc_episodes = st.number_input(
        "Số episode",
        min_value=1,
        max_value=100_000,
        value=10_000,
        step=1_000,
        key="mc_episode_input",
    )
    mc_epsilon = st.number_input(
        "Epsilon (xác suất khám phá)",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.05,
        format="%.2f",
        key="mc_epsilon_input",
    )
    mc_seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        key="mc_seed_input",
    )

    train_mc = st.button(
        "🚀 Huấn luyện Monte Carlo",
        type="primary",
        use_container_width=True,
    )

if train_mc:
    with st.spinner(
        f"Đang huấn luyện {int(mc_episodes):,} episode với điểm giao ngẫu nhiên..."
    ):
        env_mc = DeliveryGridEnv(
            random_customers=True,
            min_customers=3,
            max_customers=5,
        )
        solver_mc = OnPolicyFirstVisitMC(
            env=env_mc,
            gamma=config.GAMMA,
            epsilon=float(mc_epsilon),
            seed=int(mc_seed),
        )
        history_mc = solver_mc.train(
            num_episodes=int(mc_episodes),
            verbose=False,
        )
        route_mc = solver_mc.extract_greedy_route()

        st.session_state.mc_results = {
            "history": history_mc,
            "route": route_mc,
            "episodes": int(mc_episodes),
            "epsilon": float(mc_epsilon),
            "seed": int(mc_seed),
        }

if "mc_results" not in st.session_state:
    st.info("Chọn tham số ở thanh bên rồi bấm **Huấn luyện Monte Carlo**.")
else:
    mc_result = st.session_state.mc_results
    history_mc = mc_result["history"]
    route_mc = mc_result["route"]

    st.caption(
        f"Kết quả gần nhất: {mc_result['episodes']:,} episode | "
        f"ε = {mc_result['epsilon']:.2f} | seed = {mc_result['seed']}"
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Tỷ lệ episode hoàn thành",
        f"{history_mc['completion_rate']:.2%}",
        f"{history_mc['completed_episodes']} episode",
    )
    metric_cols[1].metric(
        "Reward trung bình gần nhất",
        f"{np.mean(history_mc['episode_rewards'][-1000:]):.1f}",
        "Tối đa 1000 episode cuối",
    )
    metric_cols[2].metric(
        "Số bước greedy route",
        f"{route_mc['total_steps']} bước",
        "Trên bố trí episode cuối",
    )
    metric_cols[3].metric(
        "Thời gian huấn luyện",
        f"{history_mc['training_time_seconds']:.2f} giây",
        f"{history_mc['truncated_episodes']} episode chạm giới hạn bước",
    )

    if not route_mc["is_completed"]:
        st.warning(
            "Greedy route chưa giao đủ khách và quay về kho trong giới hạn bước. "
            "Tỷ lệ hoàn thành khi huấn luyện không đồng nghĩa greedy route cuối đã tối ưu."
        )

    tab_route, tab_learning, tab_log = st.tabs(
        ["🗺️ Lộ trình greedy", "📈 Quá trình học", "📋 Nhật ký lộ trình"]
    )

    with tab_route:
        st.subheader("Lộ trình trên bố trí khách hàng của episode cuối")
        st.write(
            "**Tọa độ khách hàng:** "
            + ", ".join(
                f"C{index + 1} {position}"
                for index, position in enumerate(route_mc["customer_positions"])
            )
        )

        playback_col, speed_col = st.columns([1, 2])
        with playback_col:
            play_route = st.button(
                "▶️ Chạy lộ trình tự động",
                key="mc_play_route",
            )
        with speed_col:
            playback_delay = st.slider(
                "Thời gian giữa các bước (giây)",
                min_value=0.05,
                max_value=0.5,
                value=0.15,
                step=0.05,
                key="mc_playback_delay",
            )

        step_mc = st.slider(
            "Hoặc kéo để xem từng bước",
            min_value=0,
            max_value=route_mc["total_steps"],
            value=route_mc["total_steps"],
            step=1,
            key="mc_route_step",
        )
        coords_mc = route_mc["route_coords"]
        states_mc = route_mc["states_sequence"]

        visualizer_mc = GridVisualizer(
            grid_size=config.GRID_SIZE,
            depot_pos=config.DEPOT_POS,
            customer_positions=list(route_mc["customer_positions"]),
            obstacles=config.OBSTACLES,
            campus_buildings=config.CAMPUS_BUILDINGS,
        )
        map_col, status_col = st.columns([1.7, 1.0])
        map_placeholder = map_col.empty()
        status_placeholder = status_col.empty()

        def render_route_step(step_index: int) -> None:
            position = coords_mc[step_index]
            mask = states_mc[step_index][2]
            fig_mc, ax_mc = plt.subplots(figsize=(9, 7))
            visualizer_mc.draw_campus_map(
                ax_mc,
                visited_cells=set(coords_mc[:step_index + 1]),
                mask=mask,
            )
            if step_index < len(coords_mc) - 1:
                dr_mc = coords_mc[step_index + 1][0] - position[0]
                dc_mc = coords_mc[step_index + 1][1] - position[1]
            elif step_index > 0:
                dr_mc = position[0] - coords_mc[step_index - 1][0]
                dc_mc = position[1] - coords_mc[step_index - 1][1]
            else:
                dr_mc, dc_mc = 1, 0

            visualizer_mc.draw_vehicle(
                ax_mc,
                float(position[1]),
                float(position[0]),
                float(dc_mc),
                float(dr_mc),
            )
            ax_mc.set_title(
                f"Monte Carlo | Bước {step_index}/{route_mc['total_steps']} | "
                f"Xe tại {position}",
                fontsize=11,
                fontweight="bold",
            )
            map_placeholder.pyplot(fig_mc, use_container_width=True)
            plt.close(fig_mc)

            delivered_mc = bin(mask).count("1")
            total_customers_mc = len(route_mc["customer_positions"])
            action_mc = (
                route_mc["action_names"][step_index - 1]
                if step_index > 0
                else "Xuất phát"
            )
            cumulative_reward_mc = sum(route_mc["rewards"][:step_index])
            event_mc = "Xe đang ở kho, sẵn sàng xuất phát."
            if route_mc["is_completed"] and step_index == route_mc["total_steps"]:
                event_mc = "🎉 Hoàn thành giao hàng và quay về kho!"
            elif step_index > 0:
                previous_mask_mc = states_mc[step_index - 1][2]
                newly_delivered = [
                    index + 1
                    for index in range(total_customers_mc)
                    if mask & (1 << index)
                    and not previous_mask_mc & (1 << index)
                ]
                if newly_delivered:
                    event_mc = "Đã giao thành công: " + ", ".join(
                        f"C{customer_id}" for customer_id in newly_delivered
                    )
                elif step_index == route_mc["total_steps"]:
                    event_mc = "Route dừng trước khi hoàn thành nhiệm vụ."
                else:
                    event_mc = f"Đã thực hiện hành động {action_mc}."
            with status_placeholder.container():
                st.metric(
                    "Vị trí xe",
                    f"({position[0]}, {position[1]})",
                    f"Bước {step_index}",
                )
                st.metric(
                    "Tiến độ giao hàng",
                    f"{delivered_mc}/{total_customers_mc}",
                )
                st.metric("Hành động", action_mc)
                st.metric("Reward tích lũy", f"{cumulative_reward_mc:+.1f}")
                st.info(event_mc)

        if play_route:
            for playback_step in range(route_mc["total_steps"] + 1):
                render_route_step(playback_step)
                if playback_step < route_mc["total_steps"]:
                    time.sleep(playback_delay)
        else:
            render_route_step(step_mc)

    with tab_learning:
        st.subheader("Reward theo episode")
        reward_frame = pd.DataFrame(
            {
                "Episode": np.arange(1, len(history_mc["episode_rewards"]) + 1),
                "Reward mỗi episode": history_mc["episode_rewards"],
            }
        )
        reward_frame["Trung bình trượt 100 episode"] = reward_frame[
            "Reward mỗi episode"
        ].rolling(window=100, min_periods=1).mean()
        st.line_chart(
            reward_frame.set_index("Episode"),
            use_container_width=True,
        )
        st.caption(
            "Các episode bị giới hạn bước vẫn được tính return từ trải nghiệm thu được."
        )

    with tab_log:
        st.subheader("Nhật ký từng bước của greedy route")
        steps_mc = []
        cumulative_mc = 0.0
        for step_index in range(route_mc["total_steps"]):
            current_position = route_mc["route_coords"][step_index]
            next_position = route_mc["route_coords"][step_index + 1]
            step_reward = route_mc["rewards"][step_index]
            cumulative_mc += step_reward
            steps_mc.append(
                {
                    "Bước": step_index + 1,
                    "Tọa độ hiện tại": current_position,
                    "Hành động": route_mc["action_names"][step_index],
                    "Tọa độ kế tiếp": next_position,
                    "Reward": step_reward,
                    "Reward tích lũy": cumulative_mc,
                    "Bitmask đã giao": bin(
                        route_mc["states_sequence"][step_index + 1][2]
                    ),
                }
            )
        st.dataframe(pd.DataFrame(steps_mc), use_container_width=True)
