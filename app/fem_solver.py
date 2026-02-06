"""
fem_solver.py - FEMNet wrapper for rectangular slab plate analysis.

Handles mesh generation, boundary conditions, load application,
solving, and result extraction using FEMNet's QuadPlateElement.

Unit convention:
  UI input:  m, kN, kPa
  FEMNet:    mm, N, MPa (N/mm^2)
  UI output: m, kN*m/m, kN/m
"""

from femnet import *
import numpy as np


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
def m_to_mm(val):
    return val * 1000.0


def kpa_to_mpa(val):
    return val / 1000.0


def kn_per_m2_to_mpa(val):
    return val / 1000.0


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------
def create_nodes(model, Lx_mm, Ly_mm, nx, ny):
    """Create grid nodes for a rectangular mesh.

    Nodes are numbered row-by-row from bottom-left:
      node_id = j * (nx + 1) + i   where i=0..nx, j=0..ny
    """
    dx = Lx_mm / nx
    dy = Ly_mm / ny

    node_id = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            model.AddNode(node_id, x, y, 0.0)
            node_id += 1


def create_elements(model, nx, ny, thickness_mm, mat_id):
    """Create QuadPlateElements on the existing node grid.

    Elements are numbered:
      elem_id = j * nx + i   where i=0..nx-1, j=0..ny-1
    """
    elem_id = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = j * (nx + 1) + (i + 1)
            n2 = (j + 1) * (nx + 1) + (i + 1)
            n3 = (j + 1) * (nx + 1) + i
            model.add_quad_plate_element(elem_id, n0, n1, n2, n3, thickness_mm, mat_id)
            elem_id += 1


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
BC_FREE = "free"
BC_SIMPLY_SUPPORTED = "simply_supported"
BC_FIXED = "fixed"


def _is_on_edge(x, y, Lx_mm, Ly_mm, tol=1e-6):
    """Return booleans for which edges a node lies on."""
    on_left = abs(x) < tol
    on_right = abs(x - Lx_mm) < tol
    on_bottom = abs(y) < tol
    on_top = abs(y - Ly_mm) < tol
    return on_left, on_right, on_bottom, on_top


def _bc_strength(bc_type):
    """Return a numeric strength for picking the stricter BC at corners."""
    if bc_type == BC_FREE:
        return 0
    elif bc_type == BC_SIMPLY_SUPPORTED:
        return 1
    else:  # BC_FIXED
        return 2


def apply_boundary_conditions(model, Lx_mm, Ly_mm, nx, ny,
                               bc_left, bc_right, bc_bottom, bc_top):
    """Apply boundary conditions to edge nodes.

    For each edge node, the relevant edge BC is applied:
      Free:             no constraint on Uz, Rx, Ry
      Simply Supported: Uz fixed, Rx/Ry free
      Fixed:            Uz, Rx, Ry all fixed

    Corner nodes receive the stricter of the two adjoining edges.

    All nodes: Rz (drilling rotation) is fixed (required for plate elements).
    In-plane DOFs (Ux, Uy): fixed at select nodes to prevent rigid body motion.
    """
    # Determine which BCs to apply to each edge
    edge_bcs = {
        "left": bc_left,
        "right": bc_right,
        "bottom": bc_bottom,
        "top": bc_top,
    }

    # Track whether any edge is actually constrained (not free)
    any_constrained = any(bc != BC_FREE for bc in edge_bcs.values())

    for i in range(model.NodeNum()):
        node = model.GetNode(i)
        x = node.Location.x
        y = node.Location.y

        on_left, on_right, on_bottom, on_top = _is_on_edge(x, y, Lx_mm, Ly_mm)
        on_boundary = on_left or on_right or on_bottom or on_top

        # Determine the strongest BC from all edges this node belongs to
        strongest_bc = BC_FREE
        strongest_val = 0

        if on_left and _bc_strength(bc_left) > strongest_val:
            strongest_bc = bc_left
            strongest_val = _bc_strength(bc_left)
        if on_right and _bc_strength(bc_right) > strongest_val:
            strongest_bc = bc_right
            strongest_val = _bc_strength(bc_right)
        if on_bottom and _bc_strength(bc_bottom) > strongest_val:
            strongest_bc = bc_bottom
            strongest_val = _bc_strength(bc_bottom)
        if on_top and _bc_strength(bc_top) > strongest_val:
            strongest_bc = bc_top
            strongest_val = _bc_strength(bc_top)

        # Determine Uz, Rx, Ry constraints
        if strongest_bc == BC_FIXED:
            fix_uz, fix_rx, fix_ry = True, True, True
        elif strongest_bc == BC_SIMPLY_SUPPORTED:
            fix_uz, fix_rx, fix_ry = True, False, False
        else:
            fix_uz, fix_rx, fix_ry = False, False, False

        # In-plane DOFs: fix Ux/Uy at specific nodes to prevent rigid body motion
        fix_ux = False
        fix_uy = False

        if any_constrained:
            # Fix Ux, Uy at bottom-left corner
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                fix_ux = True
                fix_uy = True
            # Fix Uy at bottom-right corner (prevents rotation in XY plane)
            elif abs(x - Lx_mm) < 1e-6 and abs(y) < 1e-6:
                fix_uy = True

        # Rz is always fixed for plate elements
        fix_rz = True

        node.Fix = Support(fix_ux, fix_uy, fix_uz, fix_rx, fix_ry, fix_rz)


# ---------------------------------------------------------------------------
# Load application
# ---------------------------------------------------------------------------
def apply_uniform_load(model, nx, ny, Lx_mm, Ly_mm, q_mpa):
    """Convert uniform pressure to equivalent nodal loads using tributary area.

    Returns a VectorLoad suitable for the solver.
    """
    dx = Lx_mm / nx
    dy = Ly_mm / ny

    loads = VectorLoad()

    for i in range(model.NodeNum()):
        node = model.GetNode(i)
        x = node.Location.x
        y = node.Location.y

        on_left, on_right, on_bottom, on_top = _is_on_edge(x, y, Lx_mm, Ly_mm)

        is_corner = (on_left or on_right) and (on_bottom or on_top)
        is_edge = (on_left or on_right or on_bottom or on_top) and not is_corner

        if is_corner:
            trib_area = dx * dy / 4.0
        elif is_edge:
            trib_area = dx * dy / 2.0
        else:
            trib_area = dx * dy

        Fz = -q_mpa * trib_area  # Negative = downward
        node_load = NodeLoad(i, 0.0, 0.0, Fz)
        loads.append(node_load)

    return loads


# ---------------------------------------------------------------------------
# Solve and extract results
# ---------------------------------------------------------------------------
def solve_and_extract(model, loads, nx, ny, Lx_mm, Ly_mm, q_mpa):
    """Run FE linear static analysis and return results as numpy arrays.

    Returns dict with:
      x_nodes_mm, y_nodes_mm: 1D arrays of node coordinates
      dz_grid_mm: (ny+1, nx+1) array of vertical displacements [mm]
      elem_cx_mm, elem_cy_mm: 1D arrays of element center coordinates
      Mx, My, Mxy, Qx, Qy: (ny, nx) arrays of plate stresses at element centers
                             Mx/My/Mxy in [N*mm/mm], Qx/Qy in [N/mm]
      reaction_sum_z_N: sum of vertical reactions [N]
      total_load_N: total applied load [N]
    """
    solver = FELinearStaticOp(model, loads)
    solver.Compute()

    if not solver.Computed():
        raise RuntimeError("FEM analysis failed to converge")

    # --- Displacements ---
    displacements = solver.GetDisplacements()
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    dz_grid = np.zeros((n_nodes_y, n_nodes_x))
    x_nodes = np.zeros(n_nodes_x)
    y_nodes = np.zeros(n_nodes_y)

    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            nid = j * n_nodes_x + i
            dz_grid[j, i] = displacements[nid].Dz()
            if j == 0:
                x_nodes[i] = model.GetNode(nid).Location.x
            if i == 0:
                y_nodes[j] = model.GetNode(nid).Location.y

    # --- Plate stress data at element centers (xi=0, eta=0) ---
    n_elem = nx * ny
    Mx_arr = np.zeros((ny, nx))
    My_arr = np.zeros((ny, nx))
    Mxy_arr = np.zeros((ny, nx))
    Qx_arr = np.zeros((ny, nx))
    Qy_arr = np.zeros((ny, nx))

    dx = Lx_mm / nx
    dy = Ly_mm / ny
    elem_cx = np.zeros(nx)
    elem_cy = np.zeros(ny)

    for j in range(ny):
        for i in range(nx):
            eid = j * nx + i
            stress = solver.GetPlateStressData(eid, 0.0, 0.0)
            Mx_arr[j, i] = stress.Mx
            My_arr[j, i] = stress.My
            Mxy_arr[j, i] = stress.Mxy
            Qx_arr[j, i] = stress.Qx
            Qy_arr[j, i] = stress.Qy
            if j == 0:
                elem_cx[i] = (i + 0.5) * dx
            if i == 0:
                elem_cy[j] = (j + 0.5) * dy

    # --- Reactions (use GetReactionData: SWIG extension returning NodeLoadData) ---
    reactions = solver.GetReactionData()
    reaction_sum_z = sum(r.Pz() for r in reactions)

    # Total applied load (computed from geometry, avoids shared_ptr iteration issues)
    total_load = -q_mpa * Lx_mm * Ly_mm  # negative = downward

    return {
        "x_nodes_mm": x_nodes,
        "y_nodes_mm": y_nodes,
        "dz_grid_mm": dz_grid,
        "elem_cx_mm": elem_cx,
        "elem_cy_mm": elem_cy,
        "Mx": Mx_arr,
        "My": My_arr,
        "Mxy": Mxy_arr,
        "Qx": Qx_arr,
        "Qy": Qy_arr,
        "reaction_sum_z_N": reaction_sum_z,
        "total_load_N": total_load,
    }


# ---------------------------------------------------------------------------
# High-level runner (called from Streamlit)
# ---------------------------------------------------------------------------
POISSON_RATIO = 0.2


def run_analysis(Lx_m, Ly_m, thickness_mm, E_kN_m2,
                 nx, ny, bc_left, bc_right, bc_bottom, bc_top, q_kN_m2):
    """Run full plate analysis with UI-level units.

    Parameters use UI units (m, kN, kPa/kN/m^2).
    Poisson's ratio is fixed at 0.2.
    Returns results converted to display units.
    """
    # Unit conversions: UI -> FEMNet
    Lx_mm = m_to_mm(Lx_m)
    Ly_mm = m_to_mm(Ly_m)
    E_mpa = E_kN_m2 / 1000.0       # kN/m^2 -> N/mm^2 (MPa)
    q_mpa = kn_per_m2_to_mpa(q_kN_m2)  # kN/m^2 -> N/mm^2

    # Build model (order matters: nodes -> BCs -> material -> elements)
    model = FEModel()
    create_nodes(model, Lx_mm, Ly_mm, nx, ny)
    apply_boundary_conditions(model, Lx_mm, Ly_mm, nx, ny,
                               bc_left, bc_right, bc_bottom, bc_top)
    model.AddMaterial(E_mpa, POISSON_RATIO)
    create_elements(model, nx, ny, thickness_mm, 0)

    # Apply load
    loads = apply_uniform_load(model, nx, ny, Lx_mm, Ly_mm, q_mpa)

    # Solve
    raw = solve_and_extract(model, loads, nx, ny, Lx_mm, Ly_mm, q_mpa)

    # Unit conversions: FEMNet -> display
    # Coordinates: mm -> m
    x_nodes_m = raw["x_nodes_mm"] / 1000.0
    y_nodes_m = raw["y_nodes_mm"] / 1000.0
    elem_cx_m = raw["elem_cx_mm"] / 1000.0
    elem_cy_m = raw["elem_cy_mm"] / 1000.0

    # Displacement: mm (keep as mm for display)
    dz_grid_mm = raw["dz_grid_mm"]

    # Moments: [N*mm/mm] -> [kN*m/m]
    # Mx * 1000 (per m width) / 1e6 (N*mm -> kN*m) = Mx / 1000
    moment_factor = 1.0 / 1000.0
    Mx_kNm_m = raw["Mx"] * moment_factor
    My_kNm_m = raw["My"] * moment_factor
    Mxy_kNm_m = raw["Mxy"] * moment_factor

    # Shear: [N/mm] -> [kN/m] (factor = 1.0)
    Qx_kN_m = raw["Qx"] * 1.0
    Qy_kN_m = raw["Qy"] * 1.0

    # Reactions: N -> kN
    reaction_kN = raw["reaction_sum_z_N"] / 1000.0
    total_load_kN = raw["total_load_N"] / 1000.0

    return {
        "x_nodes_m": x_nodes_m,
        "y_nodes_m": y_nodes_m,
        "dz_grid_mm": dz_grid_mm,
        "elem_cx_m": elem_cx_m,
        "elem_cy_m": elem_cy_m,
        "Mx_kNm_m": Mx_kNm_m,
        "My_kNm_m": My_kNm_m,
        "Mxy_kNm_m": Mxy_kNm_m,
        "Qx_kN_m": Qx_kN_m,
        "Qy_kN_m": Qy_kN_m,
        "reaction_sum_kN": reaction_kN,
        "total_load_kN": total_load_kN,
    }
