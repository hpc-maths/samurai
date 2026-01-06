#!/usr/bin/env python3
"""
Finite Volume example for the advection equation in 2D using multiresolution.

This demo demonstrates:
- 2D adaptive mesh refinement (AMR)
- Upwind operator for advection
- Mesh adaptation based on multiresolution analysis
- HDF5 output for Paraview visualization

The advection equation: du/dt + a·∇u = 0
with velocity a = (1, 1) and a circular initial condition.

Note: This is a simplified demo that showcases the available Python bindings.
For the full simulation with field initialization and time stepping, see the
C++ version at demos/FiniteVolume/advection_2d.cpp

Equivalents to: demos/FiniteVolume/advection_2d.cpp
"""

import sys
import os
from pathlib import Path

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam


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

    # Create mesh and field with initial value
    mesh = sam.MRMesh2D(box, config)
    u = sam.ScalarField2D("u", mesh, 1.0)  # Initialize with value 1.0

    # ============================================================
    # Apply boundary conditions
    # ============================================================

    # Dirichlet boundary condition with value 0
    sam.make_dirichlet_bc(u, 0.0)

    # ============================================================
    # Time step calculation
    # ============================================================

    # dt based on CFL condition
    # dt = cfl * min_cell_length / max_velocity
    min_cell_length = mesh.min_cell_length  # Property, not a method
    max_velocity = max(abs(v) for v in velocity)
    dt = cfl * min_cell_length / max_velocity

    print(f"Min cell length: {min_cell_length:.6e}")
    print(f"Time step: {dt:.6e}\n")

    # ============================================================
    # Mesh adaptation setup
    # ============================================================

    # Create adaptation object
    MRadaptation = sam.make_MRAdapt(u)

    # Configure adaptation parameters
    mra_config = sam.MRAConfig()
    mra_config.epsilon = 2e-4    # Tolerance for adaptation
    mra_config.regularity = 1.0  # Mesh gradation parameter

    # ============================================================
    # Initial adaptation and save
    # ============================================================

    print("Performing initial mesh adaptation...")
    MRadaptation(mra_config)
    sam.update_ghost_mr(u)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial state
    print(f"Saving initial state to {output_path}/{filename}_init.h5")
    sam.save(str(output_path), filename + "_init", u)
    sam.dump(str(output_path), filename + "_restart_init", u)

    # ============================================================
    # Demo: Upwind operator
    # ============================================================

    print("\nDemonstrating upwind operator...")
    upwind_result = sam.upwind(velocity, u)
    print(f"  Upwind operator applied: velocity = {velocity}")
    print(f"  Result field name: {upwind_result.name}")  # Property, not a method

    # Save upwind result
    sam.save(str(output_path), filename + "_upwind", u, upwind_result)
    print(f"  Saved upwind result to {output_path}/{filename}_upwind.h5")

    # ============================================================
    # Demo: Multiple adaptation iterations
    # ============================================================

    print("\nDemonstrating multiple adaptation iterations...")
    for i in range(3):
        # Adapt mesh (in real simulation, this would be done each time step)
        MRadaptation(mra_config)
        print(f"  Iteration {i+1}: mesh adapted")

        # Save each iteration
        sam.save(str(output_path), f"{filename}_adapt_{i+1}", u)

    # ============================================================
    # Demo: Exploring the mesh structure
    # ============================================================

    print("\nExploring mesh structure...")
    cell_count = [0]

    def count_cells(cell):
        cell_count[0] += 1

    sam.for_each_cell(mesh, count_cells)
    print(f"  Total cells in mesh: {cell_count[0]}")

    # Count cells by level
    level_counts = {}

    def count_by_level(cell):
        level = cell.level
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1

    sam.for_each_cell(mesh, count_by_level)
    print(f"  Cells by level: {dict(sorted(level_counts.items()))}")

    # Sample some cells
    print("\nSampling cells:")
    sample_count = [0]

    def sample_cells(cell):
        if sample_count[0] < 5:
            center = cell.center()
            print(f"  Cell {sample_count[0]}: level={cell.level}, "
                  f"center=({center[0]:.4f}, {center[1]:.4f}), "
                  f"length={cell.length:.6f}")
            sample_count[0] += 1

    sam.for_each_cell(mesh, sample_cells)

    # ============================================================
    # Summary
    # ============================================================

    print(f"\n=== Demo Complete ===")
    print(f"\nGenerated files in {output_path}:")
    print(f"  - {filename}_init.h5/.xdmf     (initial state)")
    print(f"  - {filename}_upwind.h5/.xdmf  (upwind operator result)")
    print(f"  - {filename}_adapt_*.h5/.xdmf (adaptation iterations)")
    print(f"  - {filename}_restart_*.h5     (checkpoint files)")
    print(f"\nTo visualize in Paraview:")
    print(f"  paraview {output_path}/{filename}_init.xdmf")
    print(f"\nNote: This demo shows the available Python bindings.")
    print(f"      For a complete simulation with field initialization")
    print(f"      and time stepping, additional bindings are needed.")
    print(f"      See demos/FiniteVolume/advection_2d.cpp for the full C++ version.")


if __name__ == "__main__":
    main()
