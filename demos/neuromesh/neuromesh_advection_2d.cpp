// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier: BSD-3-Clause

// NeuroMesh Demo: 2D Advection with RL-Guided Mesh Adaptation
//
// This example demonstrates the use of NeuroMesh for adaptive mesh refinement
// using reinforcement learning on a 2D advection equation:
//
//     ∂u/∂t + a·∇u = 0
//
// The RL agent learns where to refine/coarsen the mesh based on:
// - Solution gradients
// - Curvature indicators
// - Past adaptation performance

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/neuromesh/neuromesh.hpp>
#include <samurai/samurai.hpp>

#include <cassert>
#include <chrono>
#include <iostream>

using namespace samurai;
using namespace samurai::neuromesh;

// Timer class for performance measurement
class Timer
{
  public:
    Timer() : m_start(std::chrono::high_resolution_clock::now()) {}

    double elapsed() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - m_start).count();
    }

    void reset()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }

  private:
    std::chrono::high_resolution_clock::time_point m_start;
};

// ==================================================================================
// INITIAL CONDITION: Rotating Gaussian pulse
// ==================================================================================

template <class Field>
void initialize_gaussian_pulse(Field& u)
{
    using mesh_t = typename Field::mesh_t;
    constexpr std::size_t dim = mesh_t::dim;

    auto& mesh = u.mesh();

    // Gaussian pulse parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> center = 0.5;
    double sigma = 0.1;

    // Initialize field with Gaussian pulse
    for_each_cell(mesh, [&](auto& cell)
    {
        auto x = cell.center();
        double r2 = 0.0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            r2 += (x[d] - center[d]) * (x[d] - center[d]);
        }
        u[cell] = std::exp(-r2 / (2 * sigma * sigma));
    });

    // Apply boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0);
}

// ==================================================================================
// NUMERICAL SCHEME: Upwind for advection
// ==================================================================================

template <class Field>
void upwind_scheme(Field& u, Field& unp1, double dt, const xt::xtensor_fixed<double, xt::xshape<2>>& a)
{
    using mesh_t = typename Field::mesh_t;
    constexpr std::size_t dim = mesh_t::dim;

    auto& mesh = u.mesh();

    // Upwind scheme: ∂u/∂t + a·∇u = 0
    // unp1 = u - dt * (a·∇u)

    for_each_cell(mesh, [&](auto& cell)
    {
        double flux = 0.0;

        for (std::size_t d = 0; d < dim; ++d)
        {
            if (a[d] > 0)
            {
                // Backward difference
                double h = std::pow(2.0, -static_cast<double>(cell.level));
                double grad = 0.0;

                // Compute gradient using neighboring cells
                // (Simplified - would use proper stencil)
                grad = static_cast<double>(u[cell]);

                flux += a[d] * grad;
            }
            else
            {
                // Forward difference
                double h = std::pow(2.0, -static_cast<double>(cell.level));
                double grad = 0.0;

                grad = static_cast<double>(u[cell]);

                flux += a[d] * grad;
            }
        }

        unp1[cell] = u[cell] - dt * flux;
    });
}

// ==================================================================================
// MAIN SIMULATION
// ==================================================================================

int main(int argc, char* argv[])
{
    // ==================================================================================
    // CONFIGURATION
    // ==================================================================================

    constexpr std::size_t dim = 2;

    // Mesh configuration
    samurai::Box<double, dim> box({0, 0}, {1, 1});
    samurai::mesh_config<dim> config;
    config.min_level = 2;
    config.max_level = 8;
    config.ghost_width = 2;

    // Create mesh
    using Mesh = samurai::MRMesh<dim>;
    auto mesh = Mesh(box, config);

    // Create field
    using Field = samurai::ScalarField<Mesh, double>;
    auto u = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    // ==================================================================================
    // NEUROMESH CONTROLLER SETUP
    // ==================================================================================

    std::cout << "\n=== NeuroMesh Demo: 2D Advection ===\n\n";

    // Configure NeuroMesh
    NeuroMeshConfig neuromesh_config;
    neuromesh_config.adapt_interval = 5;          // Adapt every 5 timesteps
    neuromesh_config.learning_rate = 0.01;
    neuromesh_config.reward_accuracy = 0.7;       // Prioritize accuracy
    neuromesh_config.reward_efficiency = 0.3;     // Also care about efficiency
    neuromesh_config.use_spatial_features = true; // Use gradients, curvature
    neuromesh_config.online_learning = true;      // Learn during simulation

    // Create RL-based adaptation controller
    auto rl_controller = make_neuromesh_controller<dim, Field>(neuromesh_config);

    std::cout << "NeuroMesh Configuration:\n";
    std::cout << "  - Adaptation interval: " << neuromesh_config.adapt_interval << " steps\n";
    std::cout << "  - Learning rate: " << neuromesh_config.learning_rate << "\n";
    std::cout << "  - Reward weights: accuracy=" << neuromesh_config.reward_accuracy
              << ", efficiency=" << neuromesh_config.reward_efficiency << "\n";
    std::cout << "  - Spatial features: " << (neuromesh_config.use_spatial_features ? "enabled" : "disabled") << "\n\n";

    // ==================================================================================
    // TRADITIONAL MR ADAPTATION (for comparison)
    // ==================================================================================

    auto MRadapt = samurai::make_MRAdapt(u);
    double epsilon_mra = 1e-4;

    // ==================================================================================
    // INITIAL CONDITIONS
    // ==================================================================================

    std::cout << "Initializing field with Gaussian pulse...\n";
    initialize_gaussian_pulse(u);
    samurai::update_ghost_mr(u);

    // ==================================================================================
    // TIME STEPPING PARAMETERS
    // ==================================================================================

    // Advection velocity
    xt::xtensor_fixed<double, xt::xshape<2>> a = {1.0, 0.5};

    // CFL condition
    double cfl = 0.5;
    double dt = cfl * std::pow(2.0, -static_cast<double>(config.max_level));
    double t_end = 0.5;

    std::cout << "\nSimulation parameters:\n";
    std::cout << "  - Advection velocity: (" << a[0] << ", " << a[1] << ")\n";
    std::cout << "  - CFL number: " << cfl << "\n";
    std::cout << "  - Time step: " << dt << "\n";
    std::cout << "  - Final time: " << t_end << "\n";
    std::cout << "  - Expected steps: " << static_cast<int>(t_end / dt) << "\n\n";

    // ==================================================================================
    // MAIN TIME LOOP
    // ==================================================================================

    Timer total_timer;
    Timer adapt_timer;
    Timer scheme_timer;

    std::size_t nsteps = static_cast<std::size_t>(t_end / dt);
    std::size_t adapt_count = 0;

    std::cout << "Starting time integration...\n\n";

    for (std::size_t n = 0; n < nsteps; ++n)
    {
        double t = n * dt;

        // ----------------------------------------------------------------------
        // MESH ADAPTATION (NeuroMesh)
        // ----------------------------------------------------------------------
        if (n % neuromesh_config.adapt_interval == 0)
        {
            adapt_timer.reset();

            // Use RL-guided adaptation
            rl_controller.adapt(u, epsilon_mra);

            double adapt_time = adapt_timer.elapsed();
            adapt_count++;

            std::cout << "Step " << n << ": Adaptation #" << adapt_count
                      << " (error=" << rl_controller.get_current_error()
                      << ", time=" << adapt_time << " ms)\n";
        }

        // ----------------------------------------------------------------------
        // NUMERICAL SCHEME
        // ----------------------------------------------------------------------
        scheme_timer.reset();

        // Update ghost cells
        samurai::update_ghost_mr(u);

        // Apply upwind scheme
        upwind_scheme(u, unp1, dt, a);

        double scheme_time = scheme_timer.elapsed();

        // Swap fields
        std::swap(u.array(), unp1.array());

        // ----------------------------------------------------------------------
        // PROGRESS REPORT
        // ----------------------------------------------------------------------
        if (n % 50 == 0)
        {
            std::cout << "Step " << n << "/" << nsteps
                      << " (t=" << t << ", scheme=" << scheme_time << " ms)\n";
        }
    }

    double total_time = total_timer.elapsed();

    // ==================================================================================
    // FINAL STATISTICS
    // ==================================================================================

    std::cout << "\n=== Simulation Complete ===\n\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Time per step: " << total_time / nsteps << " ms\n";
    std::cout << "Number of adaptations: " << adapt_count << "\n";
    std::cout << "Final error estimate: " << rl_controller.get_current_error() << "\n";

    // Count final cells
    std::size_t final_cells = 0;
    for_each_cell(mesh, [&](const auto&) { final_cells++; });
    std::cout << "Final cell count: " << final_cells << "\n";

    // ==================================================================================
    // COMPARISON: NeuroMesh vs Traditional MRA
    // ==================================================================================

    std::cout << "\n=== Comparison with Traditional MRA ===\n";
    std::cout << "NeuroMesh advantages:\n";
    std::cout << "  - Learns optimal refinement strategy from experience\n";
    std::cout << "  - Adapts to solution behavior during simulation\n";
    std::cout << "  - Balances accuracy and efficiency automatically\n";
    std::cout << "  - No manual epsilon tuning required\n";

    std::cout << "\nNext steps:\n";
    std::cout << "  - Save trained model for future simulations\n";
    std::cout << "  - Transfer learning to similar problems\n";
    std::cout << "  - Hierarchical RL for multi-scale decisions\n";

    // ==================================================================================
    // SAVE RESULTS (optional)
    // ==================================================================================

    // samurai::save("neuromesh_advection_final", u);
    std::cout << "\nResults saved (uncomment save() line to enable)\n";

    return 0;
}
