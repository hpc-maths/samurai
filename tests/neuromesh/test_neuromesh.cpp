// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier: BSD-3-Clause

// Unit Tests for NeuroMesh

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/neuromesh/neuromesh.hpp>
#include <samurai/samurai.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace samurai;
using namespace samurai::neuromesh;

// Test utilities
#define TEST(name) void test_##name()
#define ASSERT_NEAR(a, b, tol) assert(std::abs((a) - (b)) < (tol))
#define ASSERT_TRUE(a) assert(a)
#define ASSERT_FALSE(a) assert(!(a))

// ==================================================================================
// TEST 1: Feature Extractor
// ==================================================================================

TEST(feature_extractor_basic)
{
    constexpr std::size_t dim = 2;
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<Mesh, double>;

    // Create simple mesh
    samurai::Box<double, dim> box({0, 0}, {1, 1});
    samurai::mesh_config<dim> config;
    config.min_level = 2;
    config.max_level = 4;

    Mesh mesh(box, config);
    Field u = make_scalar_field<double>("u", mesh);

    // Initialize field with simple function
    for_each_cell(mesh, [&](auto& cell)
    {
        auto x = cell.center();
        u[cell] = x[0] + 2 * x[1];
    });

    // Test feature extractor
    NeuroMeshConfig config_feat;
    FeatureExtractor<dim, Field> extractor(config_feat);

    std::size_t cell_count = 0;
    for_each_cell(mesh, [&](const auto& cell)
    {
        auto features = extractor.extract_features(u, cell);

        // Features should have correct size
        ASSERT_TRUE(features.size() >= 2);

        // Feature 0 is field value
        ASSERT_NEAR(features(0), static_cast<double>(u[cell]), 1e-10);

        // Feature 1 is mesh level
        ASSERT_NEAR(features(1), static_cast<double>(cell.level), 1e-10);

        cell_count++;
    });

    ASSERT_TRUE(cell_count > 0);

    std::cout << "✓ test_feature_extractor_basic passed\n";
}

// ==================================================================================
// TEST 2: Reward Engine
// ==================================================================================

TEST(reward_engine_basic)
{
    constexpr std::size_t dim = 1;
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<Mesh, double>;

    samurai::Box<double, dim> box({0}, {1});
    samurai::mesh_config<dim> config;
    config.min_level = 2;
    config.max_level = 4;

    Mesh mesh(box, config);
    Field u = make_scalar_field<double>("u", mesh);

    NeuroMeshConfig config_reward;
    RewardEngine<dim, Field> reward_engine(config_reward);

    // Initial state
    reward_engine.update_state(1.0, 1000);

    // Test reward computation
    double reward1 = reward_engine.compute_reward(u, 0.8, 800);
    ASSERT_TRUE(reward1 > 0);  // Should be positive (improvement)

    double reward2 = reward_engine.compute_reward(u, 1.2, 1200);
    ASSERT_TRUE(reward2 < 0);  // Should be negative (worsening)

    std::cout << "✓ test_reward_engine_basic passed\n";
}

// ==================================================================================
// TEST 3: RL Agent
// ==================================================================================

TEST(rl_agent_basic)
{
    constexpr std::size_t dim = 2;
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<Mesh, double>;

    NeuroMeshConfig config_agent;
    RLAgent<dim, Field> agent(config_agent, 5);  // 5 features

    // Test action selection
    xt::xarray<double> state = {0.5, 0.3, 2.0, 1.5, 0.8};

    for (int i = 0; i < 100; ++i)
    {
        CellAction action = agent.select_action(state);

        // Action should be valid
        ASSERT_TRUE(action == CellAction::Keep ||
                   action == CellAction::Refine ||
                   action == CellAction::Coarsen);
    }

    // Test experience storage
    xt::xarray<double> next_state = {0.6, 0.4, 2.1, 1.6, 0.9};
    agent.store_experience(state, CellAction::Refine, 1.0, next_state, false);

    ASSERT_TRUE(agent.m_training_step == 0);  // No training yet

    std::cout << "✓ test_rl_agent_basic passed\n";
}

// ==================================================================================
// TEST 4: Adaptation Controller
// ==================================================================================

TEST(adaptation_controller_basic)
{
    constexpr std::size_t dim = 2;
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<Mesh, double>;

    samurai::Box<double, dim> box({0, 0}, {1, 1});
    samurai::mesh_config<dim> config;
    config.min_level = 2;
    config.max_level = 5;

    Mesh mesh(box, config);
    Field u = make_scalar_field<double>("u", mesh);

    // Initialize field
    for_each_cell(mesh, [&](auto& cell)
    {
        u[cell] = 1.0;
    });

    // Create controller
    NeuroMeshConfig config_ctrl;
    config_ctrl.adapt_interval = 1;
    config_ctrl.online_learning = false;  // Disable learning for test

    AdaptationController<dim, Field> controller(config_ctrl);

    // Test adaptation (should not crash)
    controller.adapt(u, 1e-3);

    ASSERT_TRUE(controller.get_adaptation_count() == 1);

    std::cout << "✓ test_adaptation_controller_basic passed\n";
}

// ==================================================================================
// TEST 5: Integration Test
// ==================================================================================

TEST(integration_advection_1d)
{
    constexpr std::size_t dim = 1;
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<Mesh, double>;

    // Setup
    samurai::Box<double, dim> box({0}, {1});
    samurai::mesh_config<dim> config;
    config.min_level = 2;
    config.max_level = 6;

    Mesh mesh(box, config);
    Field u = make_scalar_field<double>("u", mesh);
    Field unp1 = make_scalar_field<double>("unp1", mesh);

    // Initial condition: Gaussian pulse
    for_each_cell(mesh, [&](auto& cell)
    {
        double x = cell.center()[0];
        double sigma = 0.1;
        u[cell] = std::exp(-std::pow(x - 0.5, 2) / (2 * sigma * sigma));
    });

    // Create RL controller
    NeuroMeshConfig config_rl;
    config_rl.adapt_interval = 5;
    config_rl.online_learning = false;

    AdaptationController<dim, Field> controller(config_rl);

    // Time stepping
    double a = 1.0;  // Advection velocity
    double cfl = 0.5;
    double dt = cfl * std::pow(2.0, -static_cast<double>(config.max_level));
    double t_end = 0.1;

    std::size_t nsteps = static_cast<std::size_t>(t_end / dt);

    for (std::size_t n = 0; n < nsteps; ++n)
    {
        // Adapt
        if (n % config_rl.adapt_interval == 0)
        {
            controller.adapt(u, 1e-3);
        }

        // Simple upwind
        for_each_cell(mesh, [&](auto& cell)
        {
            double h = std::pow(2.0, -static_cast<double>(cell.level));
            double flux = (a > 0) ? static_cast<double>(u[cell]) : 0.0;
            unp1[cell] = u[cell] - (dt / h) * flux;
        });

        std::swap(u.array(), unp1.array());
    }

    // Check final state
    ASSERT_TRUE(controller.get_adaptation_count() > 0);

    std::cout << "✓ test_integration_advection_1d passed\n";
}

// ==================================================================================
// TEST 6: Action String Conversion
// ==================================================================================

TEST(action_to_string)
{
    ASSERT_TRUE(action_to_string(CellAction::Keep) == "Keep");
    ASSERT_TRUE(action_to_string(CellAction::Refine) == "Refine");
    ASSERT_TRUE(action_to_string(CellAction::Coarsen) == "Coarsen");

    std::cout << "✓ test_action_to_string passed\n";
}

// ==================================================================================
// TEST 7: Pre-trained Models
// ==================================================================================

TEST(pretrained_models)
{
    constexpr std::size_t dim = 2;
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<Mesh, double>;

    // Test loading pre-trained models for different PDE types
    auto agent_advection = pretrained::load_model_for_pde<dim, Field>("advection");
    auto agent_diffusion = pretrained::load_model_for_pde<dim, Field>("diffusion");
    auto agent_ns = pretrained::load_model_for_pde<dim, Field>("navier_stokes");

    // Agents should be created successfully
    ASSERT_TRUE(agent_advection.m_epsilon_current < 0.1);  // Pre-trained should have low epsilon
    ASSERT_TRUE(agent_diffusion.m_epsilon_current < 0.1);
    ASSERT_TRUE(agent_ns.m_epsilon_current < 0.1);

    std::cout << "✓ test_pretrained_models passed\n";
}

// ==================================================================================
// MAIN TEST RUNNER
// ==================================================================================

int main(int argc, char* argv[])
{
    std::cout << "\n=== NeuroMesh Unit Tests ===\n\n";

    try
    {
        test_feature_extractor_basic();
        test_reward_engine_basic();
        test_rl_agent_basic();
        test_adaptation_controller_basic();
        test_integration_advection_1d();
        test_action_to_string();
        test_pretrained_models();

        std::cout << "\n=== All Tests Passed ✓ ===\n\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << "\n\n";
        return 1;
    }
}
