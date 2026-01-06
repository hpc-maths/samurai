#!/usr/bin/env python3
"""
Finite Volume example for the advection equation in 2D using multiresolution.

This demo demonstrates:
- 2D adaptive mesh refinement (AMR)
- Upwind operator for advection
- Mesh adaptation based on multiresolution analysis
- Time stepping with Euler method
- HDF5 output for Paraview visualization

The advection equation: du/dt + a·∇u = 0
with velocity a = (1, 1) and a circular initial condition.

Equivalent to: demos/FiniteVolume/advection_2d.cpp
"""

import sys
import os
from pathlib import Path

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam


def init_circular(u, center=(0.3, 0.3), radius=0.2):
    """Initialize field with a circular condition.

    Args:
        u: ScalarField to initialize
        center: Center of the circle (x, y)
        radius: Radius of the circle
    """
    def init_cell(cell):
        cx, cy = cell.center()
        dist_sq = (cx - center[0])**2 + (cy - center[1])**2
        if dist_sq < radius**2:
            u[cell.index] = 1.0
        else:
            u[cell.index] = 0.0

    sam.for_each_cell(u.mesh, init_cell)


def main():
    """Main simulation function."""

    # ============================================================
    # Simulation parameters
    # ============================================================

    # Domain: [0, 1] x [0, 1]
    box = sam.Box2D([0.0, 0.0], [1.0, 1.0])

    # Velocity: a = (1, 1)
    velocity = [1.0, 1.0]

    # Time parameters
    Tf = 0.1        # Final time
    cfl = 0.5       # CFL condition

    # Output parameters
    output_path = Path("./results")
    filename = "FV_advection_2d_python"

    print(f"=== Advection 2D Python Demo ===")
    print(f"Domain: [0, 1] x [0, 1]")
    print(f"Velocity: ({velocity[0]}, {velocity[1]})")
    print(f"CFL: {cfl}")
    print(f"Final time: {Tf}")
    print(f"Output: {output_path}/{filename}_*.h5")
    print(f"==============================\n")

    # ============================================================
    # Mesh configuration
    # ============================================================

    config = sam.MeshConfig2D()
    config.min_level = 4      # Minimum refinement level
    config.max_level = 10     # Maximum refinement level

    # Create mesh and fields
    mesh = sam.MRMesh2D(box, config)
    u = sam.ScalarField2D("u", mesh, 0.0)      # Current solution
    unp1 = sam.ScalarField2D("unp1", mesh, 0.0)  # Next time step

    # ============================================================
    # Initialize with circular condition
    # ============================================================

    print("Initializing field with circular condition...")
    init_circular(u, center=(0.3, 0.3), radius=0.2)

    # Apply boundary conditions
    sam.make_dirichlet_bc(u, 0.0)

    # ============================================================
    # Initial mesh adaptation
    # ============================================================

    MRadaptation = sam.make_MRAdapt(u)
    mra_config = sam.MRAConfig()
    mra_config.epsilon = 2e-4
    mra_config.regularity = 1.0

    print("Performing initial mesh adaptation...")
    MRadaptation(mra_config)
    # Note: No ghost update needed here - will be done in loop before first use

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial state
    it = 0
    print(f"Saving initial state to {output_path}/{filename}_init.h5")
    sam.save(str(output_path), f"{filename}_{it:05d}", u)

    # ============================================================
    # Time stepping
    # ============================================================

    # dt based on CFL condition
    min_cell_length = mesh.min_cell_length  # Property, not a method
    max_velocity = max(abs(v) for v in velocity)
    dt = cfl * min_cell_length / max_velocity

    print(f"Min cell length: {min_cell_length:.6e}")
    print(f"Time step: {dt:.6e}")

    t = 0.0
    nt = 0
    save_interval = int(Tf / (dt * 10))  # Save ~10 times
    if save_interval < 1:
        save_interval = 1

    print(f"Starting time stepping...\n")
    print(f"{'Iter':>6} {'Time':>12} {'Cells':>10} {'Min Level':>10} {'Max Level':>10}")
    print("-" * 54)

    while t < Tf:
        # 1. Adapt mesh FIRST (as in C++ version)
        MRadaptation(mra_config)

        # 2. Update BCs and ghost cells BEFORE computing fluxes
        sam.update_ghost_mr(u)

        # 3. Update time
        t += dt
        nt += 1

        # 4. Apply upwind operator with FRESH ghost values
        upwind_result = sam.upwind(velocity, u)

        # 5. Euler time step: unp1 = u - dt * upwind(a, u)
        unp1 = u - dt * upwind_result

        # 6. Swap arrays (efficient: no memory allocation)
        sam.swap_field_arrays_2d(u, unp1)

        # Print progress and save
        if nt % save_interval == 0 or t >= Tf:
            # Count cells by level
            level_counts = {}
            def count_by_level(cell):
                level = cell.level
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
            sam.for_each_cell(mesh, count_by_level)

            min_level = min(level_counts.keys()) if level_counts else 0
            max_level = max(level_counts.keys()) if level_counts else 0
            n_cells = sum(level_counts.values())

            print(f"{nt:6d} {t:12.6e} {n_cells:10d} {min_level:10d} {max_level:10d}")

            # Save state
            sam.save(str(output_path), f"{filename}_{nt:05d}", u)

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 54)
    print(f"Simulation complete!")
    print(f"\nStatistics:")
    print(f"  Final time: {t:.6e}")
    print(f"  Time steps: {nt}")
    print(f"  Output files: {nt // save_interval + 2}")
    print(f"\nGenerated files in {output_path}:")
    print(f"  - {filename}_*.h5/.xdmf     (time series)")
    print(f"\nTo visualize in Paraview:")
    print(f"  paraview {output_path}/{filename}_00000.xdmf")
    print(f"\nThis demo is equivalent to demos/FiniteVolume/advection_2d.cpp")


if __name__ == "__main__":
    main()
