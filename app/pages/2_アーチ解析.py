"""
アーチ解析 - Cylindrical arch (barrel vault) analysis page.

Circular arc profile defined by span and rise, analyzed with FEMNet QuadPlateElements.
"""

import streamlit as st
import numpy as np
import math
from fem_solver import (
    run_arch_analysis, BC_FREE, BC_SIMPLY_SUPPORTED, BC_FIXED,
    BC_ARCH_PIN, BC_ARCH_FIXED,
)
from visualization import plot_3d_arch, elem_to_node, CAMERA_PRESETS

st.title("アーチ解析ツール")
st.caption("FEMNet 板要素解析による円筒アーチの応力・変形解析")

# =====================================================================
# Input parameters
# =====================================================================
ARCH_BC_OPTIONS = {
    "ピン支持": BC_ARCH_PIN,
    "固定": BC_ARCH_FIXED,
}

WIDTH_BC_OPTIONS = {
    "自由": BC_FREE,
    "単純支持": BC_SIMPLY_SUPPORTED,
    "固定": BC_FIXED,
}

with st.expander("入力パラメータ", expanded=True):
    # --- Geometry & Material ---
    st.subheader("形状・材料")
    col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
    with col_g1:
        L = st.slider("スパン L [m]", 1.0, 30.0, 10.0, 0.5)
    with col_g2:
        W = st.slider("幅 W [m]", 1.0, 20.0, 6.0, 0.5)
    with col_g3:
        f_max = L / 2.0 - 0.01
        f = st.slider("サグ f [m]", 0.1, f_max, min(2.0, f_max), 0.1)
    with col_g4:
        thickness = st.slider("板厚 t [mm]", 100, 500, 200, 10)
    with col_g5:
        E = st.number_input("ヤング率 E [kN/m²]", value=2.05e7, format="%.2e",
                            help="コンクリート: ~2.5e7, 鉄: ~2.05e8")

    # Derived values
    half_L = L / 2.0
    R = (half_L ** 2 + f ** 2) / (2.0 * f)
    theta = math.asin(L / (2.0 * R))
    arc_length = 2.0 * R * theta
    st.caption(
        f"曲率半径 R = {R:.3f} m, f/L = {f / L:.3f}, "
        f"弧長 S = {arc_length:.3f} m"
    )

    # --- Mesh ---
    st.subheader("メッシュ")
    mesh_factor = st.slider(
        "メッシュ係数 (要素辺長 = t × f_mesh)",
        1.0, 10.0, 2.0, 0.5,
        help="分割数は nx = L / (t × f_mesh), ny = W / (t × f_mesh) で決定",
    )
    elem_size_mm = thickness * mesh_factor
    nx = max(2, round(L * 1000.0 / elem_size_mm))
    ny = max(2, round(W * 1000.0 / elem_size_mm))
    st.caption(f"要素辺長: {elem_size_mm:.0f} mm → 分割数: nx={nx}, ny={ny}")

    # --- Boundary conditions ---
    st.subheader("境界条件")
    col_bc1, col_bc2, col_bc3, col_bc4 = st.columns(4)
    with col_bc1:
        bc_left_label = st.selectbox("左端 (x=0)", list(ARCH_BC_OPTIONS.keys()), index=0)
    with col_bc2:
        bc_right_label = st.selectbox("右端 (x=L)", list(ARCH_BC_OPTIONS.keys()), index=0)
    with col_bc3:
        bc_y0_label = st.selectbox("y=0 辺", list(WIDTH_BC_OPTIONS.keys()), index=0)
    with col_bc4:
        bc_yW_label = st.selectbox("y=W 辺", list(WIDTH_BC_OPTIONS.keys()), index=0)

    bc_left = ARCH_BC_OPTIONS[bc_left_label]
    bc_right = ARCH_BC_OPTIONS[bc_right_label]
    bc_y0 = WIDTH_BC_OPTIONS[bc_y0_label]
    bc_yW = WIDTH_BC_OPTIONS[bc_yW_label]

    # --- Load ---
    st.subheader("荷重")
    q = st.number_input("等分布荷重 q [kN/m²]（投影面積あたり）",
                        value=5.0, min_value=0.1, step=0.5)


# =====================================================================
# Run analysis (cached)
# =====================================================================
@st.cache_data
def cached_arch_analysis(L, W, f, thickness, E, nx, ny,
                          bc_left, bc_right, bc_y0, bc_yW, q):
    res = run_arch_analysis(L, W, f, thickness, E,
                             nx, ny, bc_left, bc_right, bc_y0, bc_yW, q)
    return {k: (v.copy() if hasattr(v, 'copy') else float(v)) for k, v in res.items()}


try:
    results = cached_arch_analysis(L, W, f, thickness, E, nx, ny,
                                    bc_left, bc_right, bc_y0, bc_yW, q)
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
max_Nx = np.max(np.abs(results["Nx_kN_m"]))
max_Ny = np.max(np.abs(results["Ny_kN_m"]))
max_Qx = np.max(np.abs(results["Qx_kN_m"]))
max_Qy = np.max(np.abs(results["Qy_kN_m"]))

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("最大たわみ", f"{max_dz:.4f} mm")
c2.metric("|Mx|max", f"{max_Mx:.3f} kN·m/m")
c3.metric("|My|max", f"{max_My:.3f} kN·m/m")
c4.metric("|Nx|max", f"{max_Nx:.3f} kN/m")
c5.metric("|Ny|max", f"{max_Ny:.3f} kN/m")
c6.metric("|Qx|max", f"{max_Qx:.3f} kN/m")
c7.metric("|Qy|max", f"{max_Qy:.3f} kN/m")


# =====================================================================
# Equilibrium check
# =====================================================================
reaction = results["reaction_sum_kN"]
total_load = results["total_load_kN"]
applied_load_kN = q * L * W

eq_error = abs(reaction - (-total_load))
if abs(total_load) > 1e-12:
    eq_pct = eq_error / abs(total_load) * 100
else:
    eq_pct = 0.0

with st.expander("釣合チェック", expanded=False):
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("荷重合計", f"{applied_load_kN:.2f} kN")
    ec2.metric("反力合計 (上向き+)", f"{reaction:.2f} kN")
    ec3.metric("不釣合誤差", f"{eq_pct:.4f} %")


# =====================================================================
# 3D result plot
# =====================================================================
COLOR_OPTIONS = {
    "Dz [mm]": ("Dz", "mm"),
    "Mx [kN·m/m]": ("Mx", "kN·m/m"),
    "My [kN·m/m]": ("My", "kN·m/m"),
    "Mxy [kN·m/m]": ("Mxy", "kN·m/m"),
    "Nx [kN/m]": ("Nx", "kN/m"),
    "Ny [kN/m]": ("Ny", "kN/m"),
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
        "変形/サグ比", 0.05, 1.0, 0.3, 0.05,
        help="Z方向最大変形がサグ f の何倍に見えるか",
    )

color_label, color_unit = COLOR_OPTIONS[color_choice]
color_data_map = {
    "Dz": dz,
    "Mx": elem_to_node(results["Mx_kNm_m"]),
    "My": elem_to_node(results["My_kNm_m"]),
    "Mxy": elem_to_node(results["Mxy_kNm_m"]),
    "Nx": elem_to_node(results["Nx_kN_m"]),
    "Ny": elem_to_node(results["Ny_kN_m"]),
    "Qx": elem_to_node(results["Qx_kN_m"]),
    "Qy": elem_to_node(results["Qy_kN_m"]),
}

fig_3d = plot_3d_arch(
    results["x_nodes_m"], results["y_nodes_m"],
    results["z_arch_m"], dz,
    color_data_map[color_label], color_label, color_unit,
    f, scale_ratio, camera_choice,
)
st.plotly_chart(fig_3d, use_container_width=True)
