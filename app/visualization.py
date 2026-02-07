"""
visualization.py - Plotly-based contour and 3D deformation plots for slab analysis.
"""

import numpy as np
import plotly.graph_objects as go


def plot_contour(x, y, values, title, unit, colorscale="Viridis"):
    """Create a 2D heatmap/contour figure.

    Parameters
    ----------
    x, y : 1D arrays of grid coordinates [m]
    values : 2D array of values on the grid
    title : str
    unit : str (e.g. "mm", "kN*m/m")
    colorscale : Plotly colorscale name
    """
    fig = go.Figure(data=go.Heatmap(
        x=x,
        y=y,
        z=values,
        colorscale=colorscale,
        colorbar=dict(title=unit),
        hoverongaps=False,
        hovertemplate="x: %{x:.3f} m<br>y: %{y:.3f} m<br>%{z:.4f} " + unit + "<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(constrain="domain"),
        margin=dict(l=60, r=60, t=50, b=50),
        height=500,
    )
    return fig


def elem_to_node(elem_values):
    """Average element-center values (ny, nx) to node grid (ny+1, nx+1)."""
    ny, nx = elem_values.shape
    node = np.zeros((ny + 1, nx + 1))
    count = np.zeros((ny + 1, nx + 1))
    node[:-1, :-1] += elem_values
    node[:-1, 1:] += elem_values
    node[1:, :-1] += elem_values
    node[1:, 1:] += elem_values
    count[:-1, :-1] += 1
    count[:-1, 1:] += 1
    count[1:, :-1] += 1
    count[1:, 1:] += 1
    return node / count


CAMERA_PRESETS = {
    "斜め": dict(eye=dict(x=1.25, y=-1.25, z=1.0), up=dict(x=0, y=0, z=1)),
    "真上": dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
    "正面": dict(eye=dict(x=0, y=-2.5, z=0.2), up=dict(x=0, y=0, z=1)),
    "側面": dict(eye=dict(x=-2.5, y=0, z=0.2), up=dict(x=0, y=0, z=1)),
}


def plot_3d_deformation(x_m, y_m, dz_mm, surfacecolor, color_label, color_unit,
                        scale_ratio=0.3, camera="斜め"):
    """Create a 3D surface plot of the deformed slab.

    Parameters
    ----------
    x_m, y_m : 1D arrays [m]
    dz_mm : 2D array of vertical displacements [mm] (defines geometry shape)
    surfacecolor : 2D array on node grid for coloring
    color_label, color_unit : str for colorbar title and hover
    scale_ratio : visual max deformation / min(Lx, Ly)
    camera : key in CAMERA_PRESETS
    """
    X, Y = np.meshgrid(x_m, y_m)

    max_span = max(max(x_m), max(y_m))
    min_span = min(max(x_m), max(y_m))
    z_ratio = max(1e-6, scale_ratio * min_span / max_span)

    fig = go.Figure(data=go.Surface(
        x=X, y=Y, z=dz_mm,
        surfacecolor=surfacecolor,
        colorscale="RdBu_r",
        colorbar=dict(title=f"{color_label} [{color_unit}]"),
        hovertemplate=(
            f"x: %{{x:.3f}} m<br>y: %{{y:.3f}} m<br>"
            f"{color_label}: %{{surfacecolor:.4f}} {color_unit}<extra></extra>"
        ),
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis=dict(title="", showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(
                x=max(x_m) / max_span,
                y=max(y_m) / max_span,
                z=z_ratio,
            ),
            camera=CAMERA_PRESETS.get(camera, CAMERA_PRESETS["斜め"]),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=550,
    )
    return fig


def plot_3d_arch(x_m, y_m, z_arch_m, dz_mm, surfacecolor,
                  color_label, color_unit, f_m,
                  scale_ratio=0.3, camera="斜め"):
    """Create a 3D surface plot of the deformed arch.

    Z = arch shape (true scale) + deformation (scaled for visibility).

    Parameters
    ----------
    x_m, y_m : 1D arrays [m]
    z_arch_m : 1D array (nx+1) of arch profile Z coords [m]
    dz_mm : 2D array (ny+1, nx+1) of vertical displacements [mm]
    surfacecolor : 2D array on node grid for coloring
    color_label, color_unit : str for colorbar
    f_m : arch rise [m]
    scale_ratio : deformation / rise ratio for visual scaling
    camera : key in CAMERA_PRESETS
    """
    X, Y = np.meshgrid(x_m, y_m)
    n_nodes_y = len(y_m)

    # Base arch shape (tile 1D profile to 2D)
    Z_base = np.tile(z_arch_m, (n_nodes_y, 1))

    # Deformation scale: make max displacement visible relative to rise
    max_disp_m = np.max(np.abs(dz_mm)) / 1000.0
    if max_disp_m > 1e-12:
        scale = scale_ratio * f_m / max_disp_m
    else:
        scale = 1.0

    Z_total = Z_base + dz_mm / 1000.0 * scale

    fig = go.Figure(data=go.Surface(
        x=X, y=Y, z=Z_total,
        surfacecolor=surfacecolor,
        colorscale="RdBu_r",
        colorbar=dict(title=f"{color_label} [{color_unit}]"),
        hovertemplate=(
            f"x: %{{x:.3f}} m<br>y: %{{y:.3f}} m<br>"
            f"{color_label}: %{{surfacecolor:.4f}} {color_unit}<extra></extra>"
        ),
    ))

    max_span = max(max(x_m), max(y_m))

    fig.update_layout(
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
            camera=CAMERA_PRESETS.get(camera, CAMERA_PRESETS["斜め"]),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=550,
    )
    return fig
