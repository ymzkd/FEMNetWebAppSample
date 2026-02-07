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
import math


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


def create_arch_nodes(model, L_mm, W_mm, f_mm, nx, ny):
    """Create grid nodes for a cylindrical arch mesh.

    Arch profile in X-Z plane (circular arc):
      R = ((L/2)^2 + f^2) / (2*f)
      z(x) = f - R + sqrt(R^2 - (x - L/2)^2)

    Y direction is straight (no curvature).
    X coordinates are equally spaced in horizontal projection.

    Returns: z_profile_mm (1D, nx+1) - arch Z coordinates along X
    """
    half_L = L_mm / 2.0
    R = (half_L ** 2 + f_mm ** 2) / (2.0 * f_mm)

    dx = L_mm / nx
    dy = W_mm / ny

    z_profile = np.zeros(nx + 1)
    for i in range(nx + 1):
        x = i * dx
        arg = R ** 2 - (x - half_L) ** 2
        z_profile[i] = f_mm - R + math.sqrt(max(0.0, arg))

    node_id = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            z = z_profile[i]
            model.AddNode(node_id, x, y, z)
            node_id += 1

    return z_profile


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


BC_ARCH_PIN = "arch_pin"
BC_ARCH_FIXED = "arch_fixed"


def apply_arch_boundary_conditions(model, L_mm, W_mm, f_mm, nx, ny,
                                    bc_left, bc_right, bc_y0, bc_yW):
    """Apply boundary conditions for arch analysis.

    Arch support edges (x=0, x=L):
      Pin:   Ux, Uz fixed / Uy, Rx, Ry free
      Fixed: Ux, Uz, Rx, Ry fixed / Uy free
      Ux is always fixed on arch supports to resist horizontal thrust.

    Width edges (y=0, y=W): free / simply_supported / fixed

    Rz is always fixed (plate element requirement).
    Uy is fixed at bottom-left corner to prevent rigid body motion.
    """
    def _arch_bc_strength(bc):
        if bc == BC_ARCH_PIN:
            return 1
        elif bc == BC_ARCH_FIXED:
            return 2
        return 0

    for i in range(model.NodeNum()):
        node = model.GetNode(i)
        x = node.Location.x
        y = node.Location.y

        on_left = abs(x) < 1e-6
        on_right = abs(x - L_mm) < 1e-6
        on_bottom = abs(y) < 1e-6
        on_top = abs(y - W_mm) < 1e-6

        fix_ux = False
        fix_uy = False
        fix_uz = False
        fix_rx = False
        fix_ry = False

        # Arch support edges (x=0, x=L): always fix Ux and Uz
        arch_bc = None
        if on_left and on_right:
            # Pick stronger
            arch_bc = bc_left if _arch_bc_strength(bc_left) >= _arch_bc_strength(bc_right) else bc_right
        elif on_left:
            arch_bc = bc_left
        elif on_right:
            arch_bc = bc_right

        if arch_bc is not None:
            fix_ux = True
            fix_uz = True
            if arch_bc == BC_ARCH_FIXED:
                fix_rx = True
                fix_ry = True

        # Width direction edges (y=0, y=W)
        width_bc = None
        width_strength = 0
        if on_bottom and _bc_strength(bc_y0) > width_strength:
            width_bc = bc_y0
            width_strength = _bc_strength(bc_y0)
        if on_top and _bc_strength(bc_yW) > width_strength:
            width_bc = bc_yW
            width_strength = _bc_strength(bc_yW)

        if width_bc is not None:
            if width_bc == BC_SIMPLY_SUPPORTED:
                fix_uz = True
            elif width_bc == BC_FIXED:
                fix_uz = True
                fix_rx = True
                fix_ry = True

        # Corner: merge arch + width (stricter wins per DOF, already OR-merged above)

        # Rigid body motion prevention: fix Uy at bottom-left corner
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            fix_uy = True

        # Rz always fixed for plate elements
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
    dx_grid = np.zeros((n_nodes_y, n_nodes_x))
    dy_grid = np.zeros((n_nodes_y, n_nodes_x))
    x_nodes = np.zeros(n_nodes_x)
    y_nodes = np.zeros(n_nodes_y)

    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            nid = j * n_nodes_x + i
            dz_grid[j, i] = displacements[nid].Dz()
            dx_grid[j, i] = displacements[nid].Dx()
            dy_grid[j, i] = displacements[nid].Dy()
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
    Nx_arr = np.zeros((ny, nx))
    Ny_arr = np.zeros((ny, nx))

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
            Nx_arr[j, i] = stress.Nx
            Ny_arr[j, i] = stress.Ny
            if j == 0:
                elem_cx[i] = (i + 0.5) * dx
            if i == 0:
                elem_cy[j] = (j + 0.5) * dy

    # --- Reactions ---
    reactions = solver.GetReactForces()
    reaction_sum_z = sum(r.Pz() for r in reactions)

    # Total applied load (computed from geometry, avoids shared_ptr iteration issues)
    total_load = -q_mpa * Lx_mm * Ly_mm  # negative = downward

    return {
        "x_nodes_mm": x_nodes,
        "y_nodes_mm": y_nodes,
        "dz_grid_mm": dz_grid,
        "dx_grid_mm": dx_grid,
        "dy_grid_mm": dy_grid,
        "elem_cx_mm": elem_cx,
        "elem_cy_mm": elem_cy,
        "Mx": Mx_arr,
        "My": My_arr,
        "Mxy": Mxy_arr,
        "Qx": Qx_arr,
        "Qy": Qy_arr,
        "Nx": Nx_arr,
        "Ny": Ny_arr,
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


def run_arch_analysis(L_m, W_m, f_m, thickness_mm, E_kN_m2,
                       nx, ny, bc_left, bc_right, bc_y0, bc_yW, q_kN_m2):
    """Run cylindrical arch analysis with UI-level units.

    Parameters use UI units (m, kN, kN/m^2).
    Returns results converted to display units, plus arch shape data.
    """
    # Unit conversions: UI -> FEMNet
    L_mm = m_to_mm(L_m)
    W_mm = m_to_mm(W_m)
    f_mm = m_to_mm(f_m)
    E_mpa = E_kN_m2 / 1000.0
    q_mpa = kn_per_m2_to_mpa(q_kN_m2)

    # Build model (order matters: nodes -> BCs -> material -> elements)
    model = FEModel()
    z_profile_mm = create_arch_nodes(model, L_mm, W_mm, f_mm, nx, ny)
    apply_arch_boundary_conditions(model, L_mm, W_mm, f_mm, nx, ny,
                                    bc_left, bc_right, bc_y0, bc_yW)
    model.AddMaterial(E_mpa, POISSON_RATIO)
    create_elements(model, nx, ny, thickness_mm, 0)

    # Apply load (projection-area based, reuse existing function)
    loads = apply_uniform_load(model, nx, ny, L_mm, W_mm, q_mpa)

    # Solve
    raw = solve_and_extract(model, loads, nx, ny, L_mm, W_mm, q_mpa)

    # Unit conversions: FEMNet -> display
    x_nodes_m = raw["x_nodes_mm"] / 1000.0
    y_nodes_m = raw["y_nodes_mm"] / 1000.0
    elem_cx_m = raw["elem_cx_mm"] / 1000.0
    elem_cy_m = raw["elem_cy_mm"] / 1000.0

    dz_grid_mm = raw["dz_grid_mm"]
    dx_grid_mm = raw["dx_grid_mm"]
    dy_grid_mm = raw["dy_grid_mm"]

    moment_factor = 1.0 / 1000.0
    Mx_kNm_m = raw["Mx"] * moment_factor
    My_kNm_m = raw["My"] * moment_factor
    Mxy_kNm_m = raw["Mxy"] * moment_factor

    # Shear: [N/mm] -> [kN/m] (factor = 1.0)
    Qx_kN_m = raw["Qx"] * 1.0
    Qy_kN_m = raw["Qy"] * 1.0

    # Membrane forces: [N/mm] -> [kN/m] (factor = 1.0)
    Nx_kN_m = raw["Nx"] * 1.0
    Ny_kN_m = raw["Ny"] * 1.0

    # Reactions: N -> kN
    reaction_kN = raw["reaction_sum_z_N"] / 1000.0
    total_load_kN = raw["total_load_N"] / 1000.0

    # Arch shape: mm -> m
    z_arch_m = z_profile_mm / 1000.0

    return {
        "x_nodes_m": x_nodes_m,
        "y_nodes_m": y_nodes_m,
        "z_arch_m": z_arch_m,
        "dz_grid_mm": dz_grid_mm,
        "dx_grid_mm": dx_grid_mm,
        "dy_grid_mm": dy_grid_mm,
        "elem_cx_m": elem_cx_m,
        "elem_cy_m": elem_cy_m,
        "Mx_kNm_m": Mx_kNm_m,
        "My_kNm_m": My_kNm_m,
        "Mxy_kNm_m": Mxy_kNm_m,
        "Qx_kN_m": Qx_kN_m,
        "Qy_kN_m": Qy_kN_m,
        "Nx_kN_m": Nx_kN_m,
        "Ny_kN_m": Ny_kN_m,
        "reaction_sum_kN": reaction_kN,
        "total_load_kN": total_load_kN,
    }
