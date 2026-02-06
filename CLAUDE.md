# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

矩形床板設計ツール — A rectangular slab plate bending analysis tool using FEMNet (C++ FEM library with SWIG Python bindings) and Streamlit for the interactive UI.

## Build & Run

```bash
# Build and start (first time or after Dockerfile changes)
docker-compose up --build

# Restart container (after app/ code changes, clears Streamlit cache)
docker-compose restart

# Access the app
# http://localhost:8501
```

The `./app` directory is volume-mounted into the container, so Python code changes take effect on Streamlit rerun. However, **Streamlit's `@st.cache_data` requires a container restart** to clear cached analysis results.

## Architecture

```
streamlit_app.py  →  fem_solver.py  →  FEMNet (C++ / SWIG)
                  →  visualization.py (Plotly 3D)
```

- **streamlit_app.py**: UI layer. Input parameters, result metrics, 3D plot controls. All input parameters are in page content (not sidebar) to support future multi-page layout.
- **fem_solver.py**: FEMNet wrapper. Mesh generation, boundary conditions, load application, solving, result extraction. Handles all unit conversion.
- **visualization.py**: Plotly 3D surface plot with camera presets. `elem_to_node()` averages element-center values to the node grid for visualization.

## Critical: FEMNet Model Build Order

FEMNet requires a specific construction order — element creation overwrites node boundary conditions:

```python
model = FEModel()
create_nodes(...)                    # 1. Nodes first
apply_boundary_conditions(...)       # 2. BCs second
model.AddMaterial(E_mpa, nu)         # 3. Material third
create_elements(...)                 # 4. Elements LAST
```

Violating this order causes BCs to be silently ignored (e.g., simply supported and fixed produce identical results).

## Unit Convention

| Layer | Length | Force | Pressure |
|-------|--------|-------|----------|
| UI input | m | kN | kN/m² |
| FEMNet internal | mm | N | MPa (N/mm²) |
| UI output | m | kN·m/m (moments), kN/m (shear) | — |

Key conversion factors in result extraction:
- Moments: N·mm/mm → kN·m/m = ÷ 1000
- Shear: N/mm → kN/m = × 1.0
- Poisson's ratio is hardcoded at 0.2 (`POISSON_RATIO` constant)

## FEMNet SWIG Binding Workarounds

- **`GetReactForces()`** returns unusable `SwigPyObject` — use **`GetReactionData()`** instead
- **`VectorLoad` iteration** has the same shared_ptr issue — compute total load mathematically (`-q_mpa * Lx_mm * Ly_mm`)
- **`Support` accessors** (`.Uz()` etc.) return `bool*` — use `.Get(index)` instead (0=Ux, 1=Uy, 2=Uz, 3=Rx, 4=Ry, 5=Rz)
- Cached results must be converted to plain Python/numpy types to avoid SWIG objects in Streamlit's pickle cache

## Mesh

Element divisions are computed from a user-specified coefficient: `nx = Lx / (t × f)`, `ny = Ly / (t × f)` where `t` = thickness and `f` = mesh factor (default 2.0).
