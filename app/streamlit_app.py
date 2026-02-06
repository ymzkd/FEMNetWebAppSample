"""
streamlit_app.py - Rectangular Slab Design Tool

Interactive Streamlit UI for FEM plate bending analysis using FEMNet.
"""

import streamlit as st
import numpy as np
from fem_solver import run_analysis, BC_FREE, BC_SIMPLY_SUPPORTED, BC_FIXED
from visualization import plot_3d_deformation, elem_to_node, CAMERA_PRESETS

st.set_page_config(page_title="矩形床板設計ツール", layout="wide")

st.title("矩形床板設計ツール")
st.caption("FEMNet 板要素解析による矩形スラブの応力・変形解析")

# =====================================================================
# Input parameters (inline)
# =====================================================================
BC_OPTIONS = {
    "自由": BC_FREE,
    "単純支持": BC_SIMPLY_SUPPORTED,
    "固定": BC_FIXED,
}

with st.expander("入力パラメータ", expanded=True):
    # --- Geometry & Material ---
    st.subheader("形状・材料")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        Lx = st.slider("スパン Lx [m]", 1.0, 20.0, 6.0, 0.5)
    with col_g2:
        Ly = st.slider("スパン Ly [m]", 1.0, 20.0, 6.0, 0.5)
    with col_g3:
        thickness = st.slider("板厚 t [mm]", 100, 500, 200, 10)
    with col_g4:
        E = st.number_input("ヤング率 E [kN/m²]", value=2.05e7, format="%.2e",
                            help="コンクリート: ~2.5e7, 鉄: ~2.05e8")

    # --- Mesh ---
    st.subheader("メッシュ")
    mesh_factor = st.slider(
        "メッシュ係数 f (要素辺長 = t × f)",
        1.0, 10.0, 2.0, 0.5,
        help="分割数は nx = Lx / (t × f), ny = Ly / (t × f) で決定",
    )
    elem_size_mm = thickness * mesh_factor
    nx = max(2, round(Lx * 1000.0 / elem_size_mm))
    ny = max(2, round(Ly * 1000.0 / elem_size_mm))
    st.caption(f"要素辺長: {elem_size_mm:.0f} mm → 分割数: nx={nx}, ny={ny}")

    # --- Boundary conditions ---
    st.subheader("境界条件")
    col_bc1, col_bc2, col_bc3, col_bc4 = st.columns(4)
    with col_bc1:
        bc_left_label = st.selectbox("左辺 (x=0)", list(BC_OPTIONS.keys()), index=1)
    with col_bc2:
        bc_right_label = st.selectbox("右辺 (x=Lx)", list(BC_OPTIONS.keys()), index=1)
    with col_bc3:
        bc_bottom_label = st.selectbox("下辺 (y=0)", list(BC_OPTIONS.keys()), index=1)
    with col_bc4:
        bc_top_label = st.selectbox("上辺 (y=Ly)", list(BC_OPTIONS.keys()), index=1)

    bc_left = BC_OPTIONS[bc_left_label]
    bc_right = BC_OPTIONS[bc_right_label]
    bc_bottom = BC_OPTIONS[bc_bottom_label]
    bc_top = BC_OPTIONS[bc_top_label]

    # --- Load ---
    st.subheader("荷重")
    q = st.number_input("等分布荷重 q [kN/m²]", value=5.0, min_value=0.1, step=0.5)


# =====================================================================
# Run analysis (cached)
# =====================================================================
@st.cache_data
def cached_analysis(Lx, Ly, thickness, E, nx, ny,
                    bc_left, bc_right, bc_bottom, bc_top, q):
    res = run_analysis(Lx, Ly, thickness, E,
                       nx, ny, bc_left, bc_right, bc_bottom, bc_top, q)
    # Ensure all values are plain Python/numpy types (avoid SWIG objects in cache)
    return {k: (v.copy() if hasattr(v, 'copy') else float(v)) for k, v in res.items()}


try:
    results = cached_analysis(Lx, Ly, thickness, E, nx, ny,
                              bc_left, bc_right, bc_bottom, bc_top, q)
except Exception as e:
    import traceback
    st.error(f"解析エラー: {e}")
    st.code(traceback.format_exc())
    st.stop()


# =====================================================================
# Summary metrics
# =====================================================================
st.header("解析結果")

dz = results["dz_grid_mm"]
max_dz = np.max(np.abs(dz))
max_Mx = np.max(np.abs(results["Mx_kNm_m"]))
max_My = np.max(np.abs(results["My_kNm_m"]))
max_Qx = np.max(np.abs(results["Qx_kN_m"]))
max_Qy = np.max(np.abs(results["Qy_kN_m"]))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("最大たわみ", f"{max_dz:.4f} mm")
col2.metric("|Mx|max", f"{max_Mx:.3f} kN·m/m")
col3.metric("|My|max", f"{max_My:.3f} kN·m/m")
col4.metric("|Qx|max", f"{max_Qx:.3f} kN/m")
col5.metric("|Qy|max", f"{max_Qy:.3f} kN/m")


# =====================================================================
# Equilibrium check
# =====================================================================
reaction = results["reaction_sum_kN"]
total_load = results["total_load_kN"]
applied_load_kN = q * Lx * Ly

eq_error = abs(reaction - (-total_load))
if abs(total_load) > 1e-12:
    eq_pct = eq_error / abs(total_load) * 100
else:
    eq_pct = 0.0

with st.expander("釣合チェック", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("荷重合計", f"{applied_load_kN:.2f} kN")
    c2.metric("反力合計 (上向き+)", f"{reaction:.2f} kN")
    c3.metric("不釣合誤差", f"{eq_pct:.4f} %")


# =====================================================================
# 3D result plot
# =====================================================================
COLOR_OPTIONS = {
    "Dz [mm]": ("Dz", "mm"),
    "Mx [kN·m/m]": ("Mx", "kN·m/m"),
    "My [kN·m/m]": ("My", "kN·m/m"),
    "Mxy [kN·m/m]": ("Mxy", "kN·m/m"),
    "Qx [kN/m]": ("Qx", "kN/m"),
    "Qy [kN/m]": ("Qy", "kN/m"),
}

col_v1, col_v2, col_v3 = st.columns([2, 2, 3])
with col_v1:
    color_choice = st.selectbox("表示成分", list(COLOR_OPTIONS.keys()))
with col_v2:
    camera_choice = st.selectbox("視点", list(CAMERA_PRESETS.keys()))
with col_v3:
    scale_ratio = st.slider(
        "変形/短辺比", 0.05, 1.0, 0.3, 0.05,
        help="Z方向最大変形が短辺 min(Lx, Ly) の何倍に見えるか",
    )

color_label, color_unit = COLOR_OPTIONS[color_choice]
color_data_map = {
    "Dz": dz,
    "Mx": elem_to_node(results["Mx_kNm_m"]),
    "My": elem_to_node(results["My_kNm_m"]),
    "Mxy": elem_to_node(results["Mxy_kNm_m"]),
    "Qx": elem_to_node(results["Qx_kN_m"]),
    "Qy": elem_to_node(results["Qy_kN_m"]),
}

fig_3d = plot_3d_deformation(
    results["x_nodes_m"], results["y_nodes_m"], dz,
    color_data_map[color_label], color_label, color_unit,
    scale_ratio, camera_choice,
)
st.plotly_chart(fig_3d, use_container_width=True)
