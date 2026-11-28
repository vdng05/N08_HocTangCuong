"""
=============================================================================
GIAO DIỆN MÔ PHỎNG VÀ TRỰC QUAN HÓA MÔI TRƯỜNG LƯỚI 12x12 (STREAMLIT)
Đề tài: Ứng dụng Học tăng cường trong tối ưu hóa lộ trình giao hàng
Thành viên: Vương Tiến Dũng (MSV: 2351170586) & Vũ Hải Đăng (MSV: 2351170580)
Nhóm 8 - Khoa CNTT - Trường Đại học Thủy Lợi
=============================================================================
1. Môi trường độc lập hoàn toàn với thuật toán Policy Iteration.
2. Các tham số mô hình và hàm phần thưởng là CỐ ĐỊNH.
3. Khung bên phải mặc định hiển thị BẢNG CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG (Kích thước, ô đen vật cản, ô trắng đường đi, tọa độ kho và 3 điểm giao).
4. Tách biệt phần Huấn luyện thuật toán sang chế độ riêng.
=============================================================================
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

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
st.sidebar.header("📋 THÔNG SỐ CỐ ĐỊNH")
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
    [
        "1. Môi Trường Mô Phỏng Độc Lập (Báo Cáo Nhóm)",
        "2. Kiểm Thử Thuật Toán Policy Iteration",
        "3. Kiểm Thử Thuật Toán Monte Carlo",
        "4. Kiểm Thử Thuật Toán Q-Learning",
    ],
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

# Bố trí điểm giao dùng chung cho cả 3 thuật toán (có thể Reset & Random 3–5 điểm)
if "algo_customers" not in st.session_state:
    st.session_state.algo_customers = list(config.CUSTOMER_POSITIONS)


def render_algo_env_controls(key_prefix: str):
    """Hiển thị bố trí môi trường hiện tại và nút Reset & Random cho các thuật toán."""
    custs = st.session_state.algo_customers
    cust_text = ", ".join(f"C{i+1} {pt}" for i, pt in enumerate(custs))
    st.info(f"📍 **Môi trường huấn luyện hiện tại ({len(custs)} điểm giao):** {cust_text}")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🎲 Reset & Random Vị Trí Mới (3–5 điểm)", use_container_width=True, key=f"{key_prefix}_rand_env"):
            tmp_env = DeliveryGridEnv()
            st.session_state.algo_customers = tmp_env.generate_random_customers(3, 5)
            for res_key in ("pi_results", "mc_results", "ql_results"):
                st.session_state.pop(res_key, None)
            st.rerun()
    with col_r2:
        if st.button("📌 Reset Về Mặc Định (3 điểm)", use_container_width=True, key=f"{key_prefix}_def_env"):
            st.session_state.algo_customers = list(config.CUSTOMER_POSITIONS)
            for res_key in ("pi_results", "mc_results", "ql_results"):
                st.session_state.pop(res_key, None)
            st.rerun()

# =============================================================================
# CHẾ ĐỘ 1: MÔI TRƯỜNG MÔ PHỎNG ĐỘC LẬP (NỘI DUNG CHÍNH ĐỂ TRÌNH BÀY)
# =============================================================================
if mode.startswith("1."):
    st.header("1. KHẢO SÁT & TRỰC QUAN HÓA MÔ HÌNH MÔI TRƯỜNG ĐỘC LẬP")
    st.caption("⌨️ **Click vào khung bản đồ bên dưới**, sau đó dùng phím mũi tên **`↑` `↓` `←` `→`** (hoặc **`W` `A` `S` `D`**) để lái xe mượt mà 60 FPS không độ trễ. Bấm **`R`** để Reset & Random điểm giao mới (> 2 điểm).")

    # Chuẩn bị dữ liệu cấu hình từ Python sang JS
    env_payload = {
        "gridSize": config.GRID_SIZE,
        "depot": list(config.DEPOT_POS),
        "buildings": {name: [list(pt) for pt in pts] for name, pts in config.CAMPUS_BUILDINGS.items()},
        "obstacles": [list(pt) for pt in config.OBSTACLES],
        "rewards": {
            "step": config.REWARD_STEP,
            "collision": config.REWARD_COLLISION,
            "delivery": config.REWARD_DELIVERY,
            "terminal": config.REWARD_TERMINAL,
        },
        "minCustomers": 3,
        "maxCustomers": 5,
    }
    env_json = json.dumps(env_payload, ensure_ascii=False)

    html_code = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }}
  body {{ margin: 0; padding: 0; background: transparent; color: #0f172a; }}
  .wrapper {{
    display: flex; gap: 16px; align-items: stretch;
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px;
    padding: 14px; outline: none;
  }}
  .wrapper:focus {{ border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.18); }}
  .left-pane {{ flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; }}
  canvas {{
    border: 2px solid #94a3b8; border-radius: 8px; background: #f8fafc;
    cursor: pointer; display: block;
  }}
  .kb-badge {{
    margin-top: 8px; font-size: 12.5px; font-weight: 600;
    padding: 4px 12px; border-radius: 999px;
    background: #dbeafe; color: #1d4ed8;
  }}
  .kb-badge.unfocused {{ background: #fef3c7; color: #b45309; }}
  .right-pane {{
    flex: 1; min-width: 280px; display: flex; flex-direction: column; gap: 10px;
  }}
  .card {{
    background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 12px;
  }}
  .card-title {{
    font-size: 13px; font-weight: 700; color: #1e3a8a; margin-bottom: 6px;
    border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;
  }}
  .metrics-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  }}
  .metric-box {{
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 6px 8px;
  }}
  .metric-label {{ font-size: 11px; color: #64748b; font-weight: 600; }}
  .metric-val {{ font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 2px; }}
  .toast {{
    padding: 8px 10px; border-radius: 8px; font-size: 12.5px; font-weight: 600;
    background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe;
  }}
  .toast.warn {{ background: #fef3c7; color: #b45309; border-color: #fde68a; }}
  .toast.ok {{ background: #dcfce7; color: #15803d; border-color: #86efac; }}
  .dpad {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; max-width: 240px; margin: 4px auto 0;
  }}
  .dpad button {{
    padding: 8px 6px; border: 1px solid #cbd5e1; border-radius: 8px;
    background: #ffffff; font-weight: 700; font-size: 13px; cursor: pointer;
    transition: background 0.1s, transform 0.05s;
  }}
  .dpad button:hover {{ background: #eff6ff; border-color: #3b82f6; }}
  .dpad button:active {{ transform: scale(0.96); background: #dbeafe; }}
  .btn-reset {{
    grid-column: span 3; background: #fee2e2 !important; color: #b91c1c; border-color: #fca5a5 !important;
  }}
  .btn-reset:hover {{ background: #fecaca !important; }}
  .legend-list {{ font-size: 12px; line-height: 1.55; color: #334155; margin: 0; padding-left: 16px; }}
  .cust-chips {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }}
  .chip {{
    font-size: 11.5px; font-weight: 600; padding: 2px 7px; border-radius: 6px;
    background: #ffedd5; color: #c2410c; border: 1px solid #fdba74;
  }}
  .chip.done {{ background: #dcfce7; color: #15803d; border-color: #86efac; text-decoration: line-through; }}
  .log-section {{
    margin-top: 14px; background: #ffffff; border: 1px solid #cbd5e1;
    border-radius: 12px; padding: 12px 14px;
  }}
  .log-box {{
    max-height: 200px; overflow-y: auto; font-size: 12.5px; line-height: 1.6;
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px;
    font-family: Consolas, monospace;
  }}
  .log-item {{ padding: 3px 6px; border-bottom: 1px dashed #e2e8f0; }}
  .log-item.warn {{ color: #b45309; background: #fffbeb; }}
  .log-item.ok {{ color: #15803d; background: #f0fdf4; font-weight: 600; }}
  .log-item.done {{ color: #1d4ed8; background: #eff6ff; font-weight: 700; }}
</style>
</head>
<body>
<div class="wrapper" id="simWrapper" tabindex="0">
  <div class="left-pane">
    <canvas id="gridCanvas" width="552" height="552"></canvas>
    <div class="kb-badge" id="kbStatus">⌨️ Bàn phím: Sẵn sàng (↑ ↓ ← → / WASD / R)</div>
  </div>
  <div class="right-pane">
    <div class="toast" id="eventToast">Xe sẵn sàng tại Kho (3, 1). Hãy lái xe giao hết các điểm!</div>

    <div class="card">
      <div class="card-title">📊 TRẠNG THÁI MÔI TRƯỜNG (THỜI GIAN THỰC)</div>
      <div class="metrics-grid">
        <div class="metric-box"><div class="metric-label">Vị trí xe (r, c)</div><div class="metric-val" id="mPos">(3, 1)</div></div>
        <div class="metric-box"><div class="metric-label">Tiến độ giao</div><div class="metric-val" id="mProg">0 / 3</div></div>
        <div class="metric-box"><div class="metric-label">Tổng thưởng</div><div class="metric-val" id="mRew">0 đ</div></div>
        <div class="metric-box"><div class="metric-label">Bước / Đã xong</div><div class="metric-val" id="mSteps">0 | 0 lượt</div></div>
      </div>
      <div style="margin-top:8px; font-size:12px; font-weight:600; color:#475569;">Các điểm giao hàng hiện tại:</div>
      <div class="cust-chips" id="custChips"></div>
    </div>

    <div class="card">
      <div class="card-title">🎮 ĐIỀU KHIỂN NHANH (PHÍM HOẶC CLICK)</div>
      <div class="dpad">
        <div></div>
        <button onclick="moveAction(0)">⬆️ LÊN</button>
        <div></div>
        <button onclick="moveAction(2)">⬅️ TRÁI</button>
        <button onclick="moveAction(1)">⬇️ XUỐNG</button>
        <button onclick="moveAction(3)">➡️ PHẢI</button>
        <button class="btn-reset" onclick="resetEnv(true)">🔄 Reset & Random điểm mới (R)</button>
      </div>
    </div>

    <div class="card">
      <div class="card-title">📋 CHÚ THÍCH MÔ HÌNH MÔI TRƯỜNG</div>
      <ul class="legend-list">
        <li><b>Kích thước:</b> Lưới 12x12 (144 ô) | <b>Kho (D):</b> (3, 1)</li>
        <li><b>Ô xám/đen:</b> Vật cản (35 ô tòa nhà) | <b>Ô trắng:</b> Đường đi (109 ô)</li>
        <li><b>MDP:</b> S = (x, y, mask) | A = {{Lên, Xuống, Trái, Phải}}</li>
        <li><b>Thưởng cố định:</b> Bước <code>-1</code> | Va chạm <code>-10</code> | Giao <code>+50</code> | Về kho <code>+100</code></li>
      </ul>
    </div>
  </div>
</div>

<div class="log-section">
  <div class="card-title">📜 NHẬT KÝ DI CHUYỂN CỦA XE (TỪNG BƯỚC)</div>
  <div class="log-box" id="moveLog"><div class="log-item">Chưa có bước di chuyển nào.</div></div>
</div>

<script>
const CFG = {env_json};
const N = CFG.gridSize;
const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');
const CELL = canvas.width / N;

const obstacleSet = new Set(CFG.obstacles.map(p => p[0] + ',' + p[1]));
const validCells = [];
for (let r = 0; r < N; r++) {{
  for (let c = 0; c < N; c++) {{
    const key = r + ',' + c;
    if (!obstacleSet.has(key) && !(r === CFG.depot[0] && c === CFG.depot[1])) {{
      validCells.push([r, c]);
    }}
  }}
}}

let customers = [];
let agentR = CFG.depot[0], agentC = CFG.depot[1];
let renderR = agentR, renderC = agentC;
let targetDir = [1, 0]; // [dr, dc]
let mask = 0;
let stepCount = 0;
let totalReward = 0;
let completedRounds = 0;
let visitedSet = new Set([agentR + ',' + agentC]);
let floatingTexts = [];
let collisionFlash = 0;
let moveLogs = [];

function renderLogs() {{
  const logBox = document.getElementById('moveLog');
  if (moveLogs.length === 0) {{
    logBox.innerHTML = '<div class="log-item">Chưa có bước di chuyển nào.</div>';
    return;
  }}
  logBox.innerHTML = moveLogs.slice().reverse().map(item =>
    `<div class="log-item ${{item.cls}}">${{item.text}}</div>`
  ).join('');
}}

function addLog(text, cls = '') {{
  moveLogs.push({{ text, cls }});
  if (moveLogs.length > 100) moveLogs.shift();
  renderLogs();
}}

function randomizeCustomers() {{
  const count = Math.floor(Math.random() * (CFG.maxCustomers - CFG.minCustomers + 1)) + CFG.minCustomers;
  const pool = [...validCells];
  for (let i = pool.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }}
  customers = pool.slice(0, count);
}}

function setToast(msg, type = 'info') {{
  const el = document.getElementById('eventToast');
  el.textContent = msg;
  el.className = 'toast' + (type === 'warn' ? ' warn' : type === 'ok' ? ' ok' : '');
}}

function updateHUD() {{
  let delivered = 0;
  for (let k = 0; k < customers.length; k++) {{
    if (mask & (1 << k)) delivered++;
  }}
  document.getElementById('mPos').textContent = `(${{agentR}}, ${{agentC}})`;
  document.getElementById('mProg').textContent = `${{delivered}} / ${{customers.length}}`;
  document.getElementById('mRew').textContent = `${{totalReward >= 0 ? '+' : ''}}${{totalReward}} đ`;
  document.getElementById('mSteps').textContent = `${{stepCount}} | ${{completedRounds}} lượt`;

  const chips = document.getElementById('custChips');
  chips.innerHTML = customers.map((pt, k) => {{
    const done = (mask & (1 << k)) !== 0;
    return `<span class="chip ${{done ? 'done' : ''}}">C${{k+1}} (${{pt[0]}},${{pt[1]}}) ${{done ? '✓' : ''}}</span>`;
  }}).join('');
}}

function resetEnv(manual = false) {{
  randomizeCustomers();
  agentR = CFG.depot[0];
  agentC = CFG.depot[1];
  renderR = agentR;
  renderC = agentC;
  targetDir = [1, 0];
  mask = 0;
  stepCount = 0;
  if (manual) totalReward = 0;
  visitedSet = new Set([agentR + ',' + agentC]);
  if (manual) {{
    moveLogs = [];
    renderLogs();
    setToast(`🔄 Đã Reset & Random ${{customers.length}} điểm giao mới!`, 'info');
  }}
  updateHUD();
}}

function addFloat(r, c, text, color) {{
  floatingTexts.push({{ x: (c + 0.5) * CELL, y: (r + 0.3) * CELL, text, color, alpha: 1.0 }});
}}

const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // UP, DOWN, LEFT, RIGHT
const DIR_NAMES = ['LÊN', 'XUỐNG', 'TRÁI', 'PHẢI'];

function moveAction(act) {{
  const prevR = agentR, prevC = agentC;
  const [dr, dc] = DIRS[act];
  targetDir = [dr, dc];
  stepCount++;
  const nr = agentR + dr;
  const nc = agentC + dc;

  // Kiểm tra biên hoặc vật cản
  if (nr < 0 || nr >= N || nc < 0 || nc >= N || obstacleSet.has(nr + ',' + nc)) {{
    totalReward += CFG.rewards.collision;
    collisionFlash = 1.0;
    addFloat(agentR, agentC, `${{CFG.rewards.collision}}đ`, '#dc2626');
    setToast(`⚠️ Va chạm vật cản / biên! Xe giữ nguyên tại (${{agentR}}, ${{agentC}}), phạt ${{CFG.rewards.collision}}đ`, 'warn');
    addLog(`#${{stepCount}} | ${{DIR_NAMES[act]}} | (${{prevR}},${{prevC}}) ✖ (${{nr}},${{nc}}) | R=${{CFG.rewards.collision}} (Va chạm)`, 'warn');
    updateHUD();
    return;
  }}

  agentR = nr;
  agentC = nc;
  visitedSet.add(agentR + ',' + agentC);
  let stepRew = CFG.rewards.step;
  let eventMsg = `Di chuyển ${{DIR_NAMES[act]}} tới (${{agentR}}, ${{agentC}}) (${{CFG.rewards.step}}đ)`;
  let eventType = 'info';
  let logCls = '';
  let logNote = '';

  // Kiểm tra giao hàng
  for (let k = 0; k < customers.length; k++) {{
    if (customers[k][0] === agentR && customers[k][1] === agentC && !(mask & (1 << k))) {{
      mask |= (1 << k);
      stepRew += CFG.rewards.delivery;
      addFloat(agentR, agentC, `+${{CFG.rewards.delivery}}đ`, '#16a34a');
      eventMsg = `★ GIAO THÀNH CÔNG C${{k+1}} tại (${{agentR}}, ${{agentC}})! (+${{CFG.rewards.delivery}}đ)`;
      eventType = 'ok';
      logCls = 'ok';
      logNote = ` ★ Giao C${{k+1}}`;
      break;
    }}
  }}

  totalReward += stepRew;

  // Kiểm tra hoàn thành về Kho
  const targetMask = (1 << customers.length) - 1;
  if (mask === targetMask && agentR === CFG.depot[0] && agentC === CFG.depot[1]) {{
    totalReward += CFG.rewards.terminal;
    stepRew += CFG.rewards.terminal;
    completedRounds++;
    addFloat(agentR, agentC, `+${{CFG.rewards.terminal}}đ HOÀN THÀNH!`, '#2563eb');
    addLog(`#${{stepCount}} | ${{DIR_NAMES[act]}} | (${{prevR}},${{prevC}}) → (${{agentR}},${{agentC}}) | R=+${{stepRew}} 🎉 XONG LƯỢT #${{completedRounds}}!`, 'done');
    const roundTotal = totalReward;
    totalReward = 0;
    resetEnv(false);
    setToast(`🎉 HOÀN THÀNH LƯỢT #${{completedRounds}} (+${{roundTotal}}đ)! Đã tự động Reset & Random ${{customers.length}} điểm mới!`, 'ok');
    return;
  }}

  const maskBin = mask.toString(2).padStart(customers.length, '0');
  addLog(`#${{stepCount}} | ${{DIR_NAMES[act]}} | (${{prevR}},${{prevC}}) → (${{agentR}},${{agentC}}) | R=${{stepRew >= 0 ? '+' : ''}}${{stepRew}} | Mask:${{maskBin}}${{logNote}}`, logCls);
  setToast(eventMsg, eventType);
  updateHUD();
}}

function drawScene() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Smooth interpolation
  renderR += (agentR - renderR) * 0.35;
  renderC += (agentC - renderC) * 0.35;

  // 1. Nền & ô đã đi qua
  for (let r = 0; r < N; r++) {{
    for (let c = 0; c < N; c++) {{
      const x = c * CELL, y = r * CELL;
      if (visitedSet.has(r + ',' + c)) {{
        ctx.fillStyle = '#dcfce7';
        ctx.fillRect(x, y, CELL, CELL);
      }} else {{
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x, y, CELL, CELL);
      }}
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, CELL, CELL);
    }}
  }}

  // 2. Tòa nhà vật cản
  for (const [bName, pts] of Object.entries(CFG.buildings)) {{
    const isField = bName.includes('Sân');
    ctx.fillStyle = isField ? '#065f46' : '#334155';
    let sumR = 0, sumC = 0;
    for (const [r, c] of pts) {{
      ctx.fillRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
      sumR += r; sumC += c;
    }}
    const cx = (sumC / pts.length + 0.5) * CELL;
    const cy = (sumR / pts.length + 0.5) * CELL;
    ctx.fillStyle = 'rgba(15,23,42,0.75)';
    const shortName = bName.replace('Trung Tâm', '').replace('Khu', '').trim();
    ctx.font = 'bold 9px Segoe UI';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(shortName, cx, cy);
  }}

  // 3. Kho Depot
  const dx = CFG.depot[1] * CELL, dy = CFG.depot[0] * CELL;
  ctx.fillStyle = '#dbeafe';
  ctx.fillRect(dx + 2, dy + 2, CELL - 4, CELL - 4);
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(dx + 2, dy + 2, CELL - 4, CELL - 4);
  ctx.setLineDash([]);
  ctx.fillStyle = '#1d4ed8';
  ctx.font = 'bold 11px Segoe UI';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('DEPOT', dx + CELL / 2, dy + CELL / 2);

  // 4. Các điểm khách hàng
  customers.forEach((pt, k) => {{
    const cx = (pt[1] + 0.5) * CELL, cy = (pt[0] + 0.5) * CELL;
    const done = (mask & (1 << k)) !== 0;
    if (done) {{
      ctx.fillStyle = '#bbf7d0';
      ctx.fillRect(pt[1] * CELL + 3, pt[0] * CELL + 3, CELL - 6, CELL - 6);
      ctx.strokeStyle = '#16a34a';
      ctx.lineWidth = 2;
      ctx.strokeRect(pt[1] * CELL + 3, pt[0] * CELL + 3, CELL - 6, CELL - 6);
      ctx.fillStyle = '#15803d';
      ctx.font = 'bold 12px Segoe UI';
      ctx.fillText(`C${{k+1}}✓`, cx, cy);
    }} else {{
      ctx.fillStyle = '#ea580c';
      ctx.beginPath();
      ctx.arc(cx, cy, CELL * 0.32, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#7c2d12';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Segoe UI';
      ctx.fillText(`C${{k+1}}`, cx, cy);
    }}
  }});

  // 5. Vẽ xe giao hàng mượt mà
  const vx = (renderC + 0.5) * CELL;
  const vy = (renderR + 0.5) * CELL;
  const angle = Math.atan2(targetDir[0], targetDir[1]); // dr, dc

  ctx.save();
  ctx.translate(vx, vy);
  ctx.rotate(angle);

  // Chùm đèn pha
  ctx.fillStyle = 'rgba(254, 240, 138, 0.45)';
  ctx.beginPath();
  ctx.moveTo(CELL * 0.3, -CELL * 0.16);
  ctx.lineTo(CELL * 0.85, -CELL * 0.34);
  ctx.lineTo(CELL * 0.85, CELL * 0.34);
  ctx.lineTo(CELL * 0.3, CELL * 0.16);
  ctx.closePath();
  ctx.fill();

  // Bánh xe
  ctx.fillStyle = '#111827';
  [[-0.22, -0.26], [0.16, -0.26], [-0.22, 0.20], [0.16, 0.20]].forEach(([wx, wy]) => {{
    ctx.fillRect(wx * CELL, wy * CELL, CELL * 0.14, CELL * 0.08);
  }});

  // Thân xe
  ctx.fillStyle = collisionFlash > 0.1 ? '#dc2626' : '#2563eb';
  ctx.strokeStyle = '#1e3a8a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(-CELL * 0.32, -CELL * 0.21, CELL * 0.64, CELL * 0.42, 5);
  ctx.fill();
  ctx.stroke();

  // Kính chắn gió
  ctx.fillStyle = '#93c5fd';
  ctx.fillRect(CELL * 0.12, -CELL * 0.15, CELL * 0.13, CELL * 0.30);

  // Đèn pha
  ctx.fillStyle = '#facc15';
  ctx.fillRect(CELL * 0.29, -CELL * 0.16, 4, 5);
  ctx.fillRect(CELL * 0.29, CELL * 0.11, 4, 5);
  ctx.restore();

  if (collisionFlash > 0) collisionFlash *= 0.85;

  // 6. Chữ nổi phần thưởng
  for (let i = floatingTexts.length - 1; i >= 0; i--) {{
    const ft = floatingTexts[i];
    ctx.save();
    ctx.globalAlpha = ft.alpha;
    ctx.fillStyle = ft.color;
    ctx.font = 'bold 13px Segoe UI';
    ctx.textAlign = 'center';
    ctx.fillText(ft.text, ft.x, ft.y);
    ctx.restore();
    ft.y -= 0.8;
    ft.alpha -= 0.025;
    if (ft.alpha <= 0) floatingTexts.splice(i, 1);
  }}

  requestAnimationFrame(drawScene);
}}

// Bắt phím chắc chắn trên wrapper & window
const wrapper = document.getElementById('simWrapper');
const kbStatus = document.getElementById('kbStatus');

function handleKey(e) {{
  if (e.repeat) return;
  if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') {{ e.preventDefault(); moveAction(0); }}
  else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') {{ e.preventDefault(); moveAction(1); }}
  else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{ e.preventDefault(); moveAction(2); }}
  else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {{ e.preventDefault(); moveAction(3); }}
  else if (e.key === 'r' || e.key === 'R') {{ e.preventDefault(); resetEnv(true); }}
}}

// Chỉ gắn 1 listener duy nhất trên window để tránh event bubbling làm xe đi 2 bước!
window.addEventListener('keydown', handleKey);
wrapper.addEventListener('focus', () => {{
  kbStatus.textContent = '⌨️ Bàn phím: ĐÃ KẾT NỐI (↑ ↓ ← → / WASD / R)';
  kbStatus.classList.remove('unfocused');
}});
wrapper.addEventListener('blur', () => {{
  kbStatus.textContent = '⚠️ Click vào bản đồ để bật điều khiển bàn phím';
  kbStatus.classList.add('unfocused');
}});
canvas.addEventListener('click', () => wrapper.focus());

// Khởi chạy
resetEnv(false);
wrapper.focus();
requestAnimationFrame(drawScene);
</script>
</body>
</html>
"""

    components.html(html_code, height=880)

# =============================================================================
# CHẾ ĐỘ 2: KIỂM THỬ THUẬT TOÁN POLICY ITERATION (ĐƯỢC TÁCH RIÊNG)
# =============================================================================
elif mode.startswith("2."):
    from algorithms.policy_iteration import PolicyIteration

    st.header("2. KẾT QUẢ THỰC NGHIỆM THUẬT TOÁN POLICY ITERATION")
    render_algo_env_controls("pi")

    with st.sidebar:
        st.markdown("---")
        train_pi = st.button("🚀 Huấn luyện Policy Iteration", type="primary", use_container_width=True)

    if train_pi or "pi_results" not in st.session_state:
        with st.spinner("Đang giải bài toán MDP bằng Policy Iteration..."):
            env_instance = DeliveryGridEnv(random_customers=False)
            env_instance.set_customer_positions(st.session_state.algo_customers)
            solver_instance = PolicyIteration(env_instance, gamma=config.GAMMA, theta=config.THETA)
            hist = solver_instance.train(verbose=False)
            route = solver_instance.extract_optimal_route(max_steps=200)
            st.session_state.pi_results = {
                "env": env_instance,
                "solver": solver_instance,
                "history": hist,
                "route": route,
                "customer_positions": list(env_instance.customer_positions),
            }

    pi_res = st.session_state.pi_results
    env_pi = pi_res["env"]
    solver_pi = pi_res["solver"]
    hist_pi = pi_res["history"]
    route_pi = pi_res["route"]
    pi_custs = pi_res["customer_positions"]
    visualizer.customer_positions = pi_custs

    # Metric kết quả
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng Quãng Đường Tối Ưu", f"{route_pi['total_steps']} bước", "Tối ưu tuyệt đối")
    m2.metric("Tổng Phần Thưởng", f"{route_pi['total_reward']:.1f} điểm", "Đúng hàm thưởng mới")
    m3.metric("Số Vòng Lặp Hội Tụ", f"{hist_pi['pi_iterations']} vòng", "Ổn định chính sách")
    m4.metric("Thời Gian Huấn Luyện", f"{hist_pi['training_time_seconds']:.3f} s", f"{env_pi.num_states} states")

    st.markdown("---")

    tab_pi1, tab_pi2, tab_pi3 = st.tabs([
        "🎬 Hoạt Ảnh Mô Phỏng Lộ Trình",
        "📈 Phân Tích Độ Hội Tụ & Bản Đồ Nhiệt V(s)",
        "📋 Nhật Ký Chi Tiết Từng Bước Đi"
    ])

    with tab_pi1:
        st.subheader("Lộ Trình Tối Ưu và Bảng Telemetry Thời Gian Thực")
        step_val = st.slider("Kéo xem vị trí xe:", 0, route_pi["total_steps"], route_pi["total_steps"], 1, key="pi_step_slider")

        fig_pi = plt.figure(figsize=(15, 8.5))
        gs_pi = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.15)
        ax_m = fig_pi.add_subplot(gs_pi[0])
        ax_t = fig_pi.add_subplot(gs_pi[1])

        coords = route_pi["route_coords"]
        states_s = route_pi["states_sequence"]
        visited = set(coords[:step_val + 1])
        c_pos = coords[step_val]
        c_mask = states_s[step_val][2]

        visualizer.draw_campus_map(ax_m, visited_cells=visited, mask=c_mask)

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
        if c_mask == env_pi.target_mask and c_pos == config.DEPOT_POS:
            evt_txt = "★ HOÀN TẤT VÀ VỀ KHO THÀNH CÔNG (+100đ)!"
        elif step_val > 0:
            for k, cp in enumerate(pi_custs):
                if c_pos == cp and (c_mask & (1 << k)):
                    evt_txt = f"★ GIAO THÀNH CÔNG C{k+1} (+50đ)!"

        visualizer.draw_telemetry_hud(
            ax_t, step_val, route_pi["total_steps"], c_pos, act_text, c_mask, cum_rew, evt_txt
        )

        st.pyplot(fig_pi)
        plt.close(fig_pi)

    with tab_pi2:
        st.subheader("Bản Đồ Nhiệt Hàm Giá Trị Trạng Thái V(s) Theo Tiến Trình Giao Hàng")
        h1, h2 = st.columns(2)
        # Lấy tối đa 4 mốc mask tiêu biểu dọc theo lộ trình tối ưu
        stage_masks = [(0, "Khởi đầu: Chưa giao điểm nào (Mask = 0)")]
        running_mask = 0
        for order_item in route_pi["visited_customers_order"]:
            cid = int(order_item["customer_id"][1:]) - 1
            running_mask |= (1 << cid)
            if running_mask != env_pi.target_mask and len(stage_masks) < 3:
                stage_masks.append((running_mask, f"Sau khi giao {order_item['customer_id']} {order_item['position']}"))
        stage_masks.append((env_pi.target_mask, f"Đã giao đủ {env_pi.num_customers} điểm -> Quay về Kho {config.DEPOT_POS}"))

        for idx, (m, stitle) in enumerate(stage_masks):
            grid_v = np.full((config.GRID_SIZE, config.GRID_SIZE), np.nan)
            for r in range(config.GRID_SIZE):
                for c in range(config.GRID_SIZE):
                    s = (r, c, m)
                    if s in env_pi.state_to_idx:
                        grid_v[r, c] = solver_pi.V[env_pi.state_to_idx[s]]

            fig_h, ax_h = plt.subplots(figsize=(6, 5))
            sns.heatmap(grid_v, ax=ax_h, cmap="viridis" if m != env_pi.target_mask else "plasma", cbar=True)
            ax_h.set_title(stitle, fontsize=9.5, fontweight="bold")

            target_col = h1 if idx % 2 == 0 else h2
            target_col.pyplot(fig_h)
            plt.close(fig_h)

    with tab_pi3:
        st.subheader(f"Bảng Nhật Ký {route_pi['total_steps']} Bước Di Chuyển")
        render_route_log_table(route_pi, "pi")


# =============================================================================
# HÀM HỖ TRỢ TRỰC QUAN HÓA LỘ TRÌNH CHUNG CHO MONTE CARLO & Q-LEARNING
# =============================================================================
def render_algorithm_playback(route_data, customer_positions, algo_title: str, key_prefix: str):
    """Hiển thị lộ trình từng bước kèm nút chạy tự động và bảng trạng thái."""
    playback_col, speed_col = st.columns([1, 2])
    with playback_col:
        play_route = st.button("▶️ Chạy lộ trình tự động", key=f"{key_prefix}_play")
    with speed_col:
        playback_delay = st.slider(
            "Thời gian giữa các bước (giây)",
            min_value=0.05, max_value=0.5, value=0.15, step=0.05,
            key=f"{key_prefix}_delay",
        )

    total_steps = route_data["total_steps"]
    step_val = st.slider(
        "Hoặc kéo để xem từng bước",
        min_value=0, max_value=total_steps, value=total_steps, step=1,
        key=f"{key_prefix}_step",
    )

    coords = route_data["route_coords"]
    states_seq = route_data["states_sequence"]
    viz = GridVisualizer(
        grid_size=config.GRID_SIZE,
        depot_pos=config.DEPOT_POS,
        customer_positions=list(customer_positions),
        obstacles=config.OBSTACLES,
        campus_buildings=config.CAMPUS_BUILDINGS,
    )

    map_col, status_col = st.columns([1.7, 1.0])
    map_ph = map_col.empty()
    status_ph = status_col.empty()

    def draw_step(step_idx: int):
        pos = coords[step_idx]
        mask = states_seq[step_idx][2]
        fig_s, ax_s = plt.subplots(figsize=(9, 7))
        viz.draw_campus_map(ax_s, visited_cells=set(coords[:step_idx + 1]), mask=mask)

        if step_idx < len(coords) - 1:
            dr_s = coords[step_idx + 1][0] - pos[0]
            dc_s = coords[step_idx + 1][1] - pos[1]
        elif step_idx > 0:
            dr_s = pos[0] - coords[step_idx - 1][0]
            dc_s = pos[1] - coords[step_idx - 1][1]
        else:
            dr_s, dc_s = 1, 0

        viz.draw_vehicle(ax_s, float(pos[1]), float(pos[0]), float(dc_s), float(dr_s))
        ax_s.set_title(
            f"{algo_title} | Bước {step_idx}/{total_steps} | Xe tại {pos}",
            fontsize=11, fontweight="bold",
        )
        map_ph.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

        delivered = bin(mask).count("1")
        total_c = len(customer_positions)
        act_name = route_data["action_names"][step_idx - 1] if step_idx > 0 else "Xuất phát"
        cum_rew = sum(route_data["rewards"][:step_idx])

        evt = "Xe đang ở kho, sẵn sàng xuất phát."
        if route_data["is_completed"] and step_idx == total_steps:
            evt = "🎉 Hoàn thành giao hàng và quay về kho!"
        elif step_idx > 0:
            prev_mask = states_seq[step_idx - 1][2]
            newly = [i + 1 for i in range(total_c) if (mask & (1 << i)) and not (prev_mask & (1 << i))]
            if newly:
                evt = "Đã giao thành công: " + ", ".join(f"C{cid}" for cid in newly)
            elif step_idx == total_steps:
                evt = "Lộ trình dừng trước khi hoàn thành nhiệm vụ."
            else:
                evt = f"Đã thực hiện hành động {act_name}."

        with status_ph.container():
            st.metric("Vị trí xe", f"({pos[0]}, {pos[1]})", f"Bước {step_idx}")
            st.metric("Tiến độ giao hàng", f"{delivered}/{total_c}")
            st.metric("Hành động", act_name)
            st.metric("Reward tích lũy", f"{cum_rew:+.1f}")
            st.info(evt)

    if play_route:
        for s_i in range(total_steps + 1):
            draw_step(s_i)
            if s_i < total_steps:
                time.sleep(playback_delay)
    else:
        draw_step(step_val)


def render_route_log_table(route_data, key_prefix: str):
    """Hiển thị bảng nhật ký từng bước của lộ trình."""
    steps_rows = []
    cum = 0.0
    for idx in range(route_data["total_steps"]):
        rew = route_data["rewards"][idx]
        cum += rew
        steps_rows.append({
            "Bước": idx + 1,
            "Tọa độ hiện tại": str(route_data["route_coords"][idx]),
            "Hành động": route_data["action_names"][idx],
            "Tọa độ kế tiếp": str(route_data["route_coords"][idx + 1]),
            "Reward": rew,
            "Reward tích lũy": cum,
            "Mask": bin(route_data["states_sequence"][idx + 1][2]),
        })
    df_log = pd.DataFrame(steps_rows)
    st.dataframe(df_log, use_container_width=True, key=f"{key_prefix}_df")


# =============================================================================
# CHẾ ĐỘ 3: KIỂM THỬ THUẬT TOÁN MONTE CARLO (TÍCH HỢP TỪ demo_monte_carlo.py)
# =============================================================================
if mode.startswith("3."):
    from algorithms.monte_carlo import OnPolicyFirstVisitMC

    st.header("3. ON-POLICY FIRST-VISIT MONTE CARLO")
    render_algo_env_controls("mc")

    with st.sidebar:
        st.markdown("---")
        st.header("⚙️ Tham số Monte Carlo")
        mc_episodes = st.number_input(
            "Số episode", min_value=1, max_value=100_000, value=10_000, step=1_000, key="mc_episode_input"
        )
        mc_epsilon = st.number_input(
            "Epsilon (xác suất khám phá)", min_value=0.01, max_value=1.0, value=0.1, step=0.05, format="%.2f", key="mc_epsilon_input"
        )
        mc_seed = st.number_input(
            "Seed", min_value=0, max_value=2_147_483_647, value=42, step=1, key="mc_seed_input"
        )
        mc_random_per_ep = st.checkbox("Random lại điểm giao mỗi episode", value=False, key="mc_rand_ep")
        train_mc = st.button("🚀 Huấn luyện Monte Carlo", type="primary", use_container_width=True)

    if train_mc:
        with st.spinner(f"Đang huấn luyện Monte Carlo ({int(mc_episodes):,} episodes)..."):
            if mc_random_per_ep:
                env_mc = DeliveryGridEnv(random_customers=True, min_customers=3, max_customers=5)
            else:
                env_mc = DeliveryGridEnv(random_customers=False)
                env_mc.set_customer_positions(st.session_state.algo_customers)
            solver_mc = OnPolicyFirstVisitMC(
                env=env_mc, gamma=config.GAMMA, epsilon=float(mc_epsilon), seed=int(mc_seed)
            )
            history_mc = solver_mc.train(num_episodes=int(mc_episodes), verbose=False)
            route_mc = solver_mc.extract_greedy_route()

            st.session_state.mc_results = {
                "history": history_mc,
                "route": route_mc,
                "episodes": int(mc_episodes),
                "epsilon": float(mc_epsilon),
                "seed": int(mc_seed),
            }

    if "mc_results" not in st.session_state:
        st.info("Chọn tham số ở thanh bên rồi bấm **🚀 Huấn luyện Monte Carlo**.")
    else:
        mc_res = st.session_state.mc_results
        history_mc = mc_res["history"]
        route_mc = mc_res["route"]

        st.caption(f"Kết quả gần nhất: {mc_res['episodes']:,} episode | ε = {mc_res['epsilon']:.2f} | seed = {mc_res['seed']}")

        mc_cols = st.columns(4)
        mc_cols[0].metric("Tỷ lệ episode hoàn thành", f"{history_mc['completion_rate']:.2%}", f"{history_mc['completed_episodes']} episode")
        mc_cols[1].metric("Reward TB gần nhất", f"{np.mean(history_mc['episode_rewards'][-1000:]):.1f}", "Tối đa 1000 episode cuối")
        mc_cols[2].metric("Số bước greedy route", f"{route_mc['total_steps']} bước", "Trên bố trí episode cuối")
        mc_cols[3].metric("Thời gian huấn luyện", f"{history_mc['training_time_seconds']:.2f} giây", f"{history_mc['truncated_episodes']} chạm giới hạn bước")

        if not route_mc["is_completed"]:
            st.warning("Greedy route chưa giao đủ khách và quay về kho trong giới hạn bước.")

        tab_mc1, tab_mc2, tab_mc3 = st.tabs(["🗺️ Lộ trình greedy", "📈 Quá trình học", "📋 Nhật ký lộ trình"])

        with tab_mc1:
            st.subheader("Lộ trình trên bố trí khách hàng của episode cuối")
            st.write("**Tọa độ khách hàng:** " + ", ".join(f"C{i+1} {p}" for i, p in enumerate(route_mc["customer_positions"])))
            render_algorithm_playback(route_mc, route_mc["customer_positions"], "Monte Carlo", "mc")

        with tab_mc2:
            st.subheader("Reward theo episode")
            df_mc_rew = pd.DataFrame({
                "Episode": np.arange(1, len(history_mc["episode_rewards"]) + 1),
                "Reward mỗi episode": history_mc["episode_rewards"],
            })
            df_mc_rew["Trung bình trượt 100 episode"] = df_mc_rew["Reward mỗi episode"].rolling(window=100, min_periods=1).mean()
            st.line_chart(df_mc_rew.set_index("Episode"), use_container_width=True)

        with tab_mc3:
            st.subheader("Nhật ký từng bước của greedy route")
            render_route_log_table(route_mc, "mc")


# =============================================================================
# CHẾ ĐỘ 4: KIỂM THỬ THUẬT TOÁN Q-LEARNING (DỰA VÀO run_q_learning.py)
# =============================================================================
elif mode.startswith("4."):
    from algorithms.q_learning import QLearning

    st.header("4. THUẬT TOÁN Q-LEARNING (MODEL-FREE TD CONTROL)")
    render_algo_env_controls("ql")

    with st.sidebar:
        st.markdown("---")
        st.header("⚙️ Tham số Q-Learning")
        ql_episodes = st.number_input(
            "Số episode", min_value=100, max_value=50_000, value=5_000, step=500, key="ql_episode_input"
        )
        ql_alpha = st.number_input(
            "Learning rate (alpha)", min_value=0.01, max_value=1.0, value=0.10, step=0.05, format="%.2f", key="ql_alpha_input"
        )
        ql_eps_decay = st.number_input(
            "Epsilon decay", min_value=0.900, max_value=0.999, value=0.995, step=0.001, format="%.3f", key="ql_decay_input"
        )
        train_ql = st.button("🚀 Huấn luyện Q-Learning", type="primary", use_container_width=True)

    if train_ql:
        with st.spinner(f"Đang huấn luyện Q-Learning trong {int(ql_episodes):,} episodes..."):
            env_ql = DeliveryGridEnv(random_customers=False)
            env_ql.set_customer_positions(st.session_state.algo_customers)
            agent_ql = QLearning(
                env=env_ql,
                alpha=float(ql_alpha),
                gamma=config.GAMMA,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=float(ql_eps_decay),
                num_episodes=int(ql_episodes),
            )
            history_ql = agent_ql.train(verbose=False)
            route_ql = agent_ql.extract_optimal_route(max_steps=200)

            st.session_state.ql_results = {
                "history": history_ql,
                "route": route_ql,
                "customer_positions": list(env_ql.customer_positions),
                "episodes": int(ql_episodes),
                "alpha": float(ql_alpha),
                "epsilon_decay": float(ql_eps_decay),
            }

    if "ql_results" not in st.session_state:
        st.info("Chọn tham số ở thanh bên rồi bấm **🚀 Huấn luyện Q-Learning**.")
    else:
        ql_res = st.session_state.ql_results
        history_ql = ql_res["history"]
        route_ql = ql_res["route"]

        st.caption(
            f"Kết quả gần nhất: {ql_res['episodes']:,} episodes | α = {ql_res['alpha']:.2f} | "
            f"ε_decay = {ql_res['epsilon_decay']:.3f}"
        )

        recent_success_rate = float(np.mean(history_ql["success_per_episode"][-100:]) * 100)
        ql_cols = st.columns(4)
        ql_cols[0].metric("Tỷ lệ thành công (100 ep cuối)", f"{recent_success_rate:.1f}%", f"{sum(history_ql['success_per_episode'])} ep thành công")
        ql_cols[1].metric("Số bước lộ trình", f"{route_ql['total_steps']} bước", "Hoàn thành" if route_ql["is_completed"] else "Chưa hoàn thành")
        ql_cols[2].metric("Tổng phần thưởng lộ trình", f"{route_ql['total_reward']:+.1f} điểm", "Chính sách Greedy")
        ql_cols[3].metric("Thời gian huấn luyện", f"{history_ql['training_time_seconds']:.2f} giây", f"Epsilon cuối: {history_ql['epsilon_history'][-1]:.4f}")

        if not route_ql["is_completed"]:
            st.warning("Chính sách Q-Learning hiện tại chưa hoàn thành nhiệm vụ. Hãy thử tăng số episodes lên (ví dụ 3,000–10,000).")

        tab_ql1, tab_ql2, tab_ql3 = st.tabs(["🗺️ Lộ trình tối ưu", "📈 Quá trình học", "📋 Nhật ký lộ trình"])

        with tab_ql1:
            st.subheader("Lộ trình trích xuất từ bảng Q(s, a)")
            if route_ql["visited_customers_order"]:
                st.write(
                    "**Thứ tự giao hàng:** "
                    + " → ".join(f"{c['customer_id']} {c['position']} (Bước {c['step']})" for c in route_ql["visited_customers_order"])
                )
            render_algorithm_playback(route_ql, ql_res["customer_positions"], "Q-Learning", "ql")

        with tab_ql2:
            st.subheader("Biểu đồ hội tụ Q-Learning")
            ep_idx = np.arange(1, len(history_ql["episode_rewards"]) + 1)
            df_ql_rew = pd.DataFrame({
                "Episode": ep_idx,
                "Reward mỗi episode": history_ql["episode_rewards"],
            })
            df_ql_rew["Trung bình trượt 100 episode"] = df_ql_rew["Reward mỗi episode"].rolling(window=100, min_periods=1).mean()
            st.line_chart(df_ql_rew.set_index("Episode"), use_container_width=True)

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                df_ql_succ = pd.DataFrame({
                    "Episode": ep_idx,
                    "Tỷ lệ thành công (%)": pd.Series(history_ql["success_per_episode"], dtype=float).rolling(window=100, min_periods=1).mean() * 100,
                })
                st.line_chart(df_ql_succ.set_index("Episode"), use_container_width=True)
            with col_q2:
                df_ql_eps = pd.DataFrame({
                    "Episode": ep_idx,
                    "Epsilon": history_ql["epsilon_history"],
                })
                st.line_chart(df_ql_eps.set_index("Episode"), use_container_width=True)

        with tab_ql3:
            st.subheader("Nhật ký từng bước của lộ trình Q-Learning")
            render_route_log_table(route_ql, "ql")
