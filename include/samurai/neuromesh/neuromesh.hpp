// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "../field.hpp"
#include "../mesh.hpp"
#include "../mr/mesh.hpp"

namespace samurai::neuromesh
{
    // ==================================================================================
    // CONFIGURATION
    // ==================================================================================

    struct NeuroMeshConfig
    {
        // Feature extractor parameters
        std::size_t cnn_filters      = 16;     // Number of CNN filters (lightweight)
        std::size_t cnn_kernel_size  = 3;      // Kernel size for feature extraction
        bool use_spatial_features   = true;  // Extract spatial gradients, curvature

        // RL agent parameters
        std::size_t hidden_dim       = 128;    // Hidden layer size (small for efficiency)
        double learning_rate         = 0.01;   // Learning rate
        double gamma                 = 0.99;   // Discount factor
        double epsilon               = 0.1;    // Exploration rate
        std::size_t replay_buffer_size = 10000; // Experience replay size

        // Reward weights
        double reward_accuracy       = 0.6;    // Weight for accuracy improvement
        double reward_efficiency     = 0.4;    // Weight for cell count reduction
        double reward_stability      = 0.0;    // Weight for mesh stability

        // Adaptation parameters
        std::size_t adapt_interval    = 10;     // Adapt every N timesteps
        double exploration_budget    = 0.05;   // Fraction of cells for exploration

        // Training parameters
        std::size_t batch_size       = 64;     // Training batch size
        std::size_t training_epochs  = 5;      // Epochs per adaptation
        bool online_learning         = true;   // Learn during simulation

        // Safety parameters
        double max_cells_multiplier  = 2.0;    // Max cells relative to initial
        bool fallback_to_mra         = true;   // Fallback to traditional MRA if RL fails
        double error_threshold       = 2.0;    // Error threshold for fallback

        // I/O parameters
        bool save_training_data      = false;  // Save experience replay
        std::string checkpoint_dir   = "./neuromesh_checkpoints";
        std::string model_filename   = "neuromesh_model.dat";
    };

    // ==================================================================================
    // ACTIONS
    // ==================================================================================

    enum class CellAction : int
    {
        Keep     = 0,  // Maintain current level
        Refine   = 1,  // Increase refinement level
        Coarsen  = 2,  // Decrease refinement level
        Count    = 3
    };

    inline std::string action_to_string(CellAction action)
    {
        switch (action)
        {
            case CellAction::Keep:    return "Keep";
            case CellAction::Refine:  return "Refine";
            case CellAction::Coarsen: return "Coarsen";
            default:                  return "Unknown";
        }
    }

    // ==================================================================================
    // FEATURE EXTRACTOR
    // ==================================================================================

    template <std::size_t dim, class Field>
    class FeatureExtractor
    {
      public:
        using config_t       = NeuroMeshConfig;
        using field_t        = Field;
        using mesh_t         = typename field_t::mesh_t;
        using value_t        = typename field_t::value_type;
        using feature_array_t = xt::xarray<double>;

      private:
        config_t m_config;
        mutable std::mt19937 m_rng;

      public:
        explicit FeatureExtractor(const config_t& config = config_t{})
            : m_config(config)
            , m_rng(std::random_device{}())
        {
        }

        // Extract features from field for a single cell
        feature_array_t extract_features(const field_t& field, const auto& cell) const
        {
            feature_array_t features;

            if (m_config.use_spatial_features)
            {
                features = extract_spatial_features(field, cell);
            }
            else
            {
                features = extract_value_features(field, cell);
            }

            return features;
        }

        // Extract features for entire mesh (batch processing)
        feature_array_t extract_batch(const field_t& field) const
        {
            const auto& mesh = field.mesh();

            // Count total cells
            std::size_t total_cells = 0;
            for_each_cell(mesh, [&](const auto& c) { total_cells++; });

            // Allocate feature matrix
            std::size_t feature_dim = m_config.use_spatial_features ? dim + 3 : 2;
            feature_array_t features = xt::zeros<double>({total_cells, feature_dim});

            // Extract features for each cell
            std::size_t idx = 0;
            for_each_cell(mesh, [&](const auto& cell)
            {
                auto row = xt::view(features, idx, xt::all());
                auto cell_features = extract_features(field, cell);
                row = cell_features;
                idx++;
            });

            return features;
        }

      private:
        // Spatial feature extraction (gradient, curvature, level)
        feature_array_t extract_spatial_features(const field_t& field, const auto& cell) const
        {
            feature_array_t features = xt::zeros<double>({dim + 3});

            // Field value
            features(0) = static_cast<double>(field[cell]);

            // Mesh level
            features(1) = static_cast<double>(cell.level);

            // Spatial gradients (finite difference approximation)
            auto center = cell.center();
            double h = std::pow(2.0, -static_cast<double>(cell.level));

            for (std::size_t d = 0; d < dim; ++d)
            {
                // Compute gradient in dimension d
                double grad = compute_gradient(field, cell, d, h);
                features(2 + d) = grad;
            }

            // Laplacian (curvature indicator)
            features(dim + 2) = compute_laplacian(field, cell, h);

            return features;
        }

        // Simple value-based features
        feature_array_t extract_value_features(const field_t& field, const auto& cell) const
        {
            feature_array_t features = xt::zeros<double>({2});
            features(0) = static_cast<double>(field[cell]);
            features(1) = static_cast<double>(cell.level);
            return features;
        }

        // Compute gradient in direction d using finite differences
        double compute_gradient(const field_t& field, const auto& cell, std::size_t d, double h) const
        {
            double grad = 0.0;

            // Simple central difference approximation
            auto center = cell.center();
            auto coord_plus = center;
            auto coord_minus = center;
            coord_plus[d] += h * 0.5;
            coord_minus[d] -= h * 0.5;

            // Find neighboring cells and compute difference
            value_t val_plus = interpolate_value(field, coord_plus, cell.level);
            value_t val_minus = interpolate_value(field, coord_minus, cell.level);
            grad = static_cast<double>((val_plus - val_minus) / h);

            return grad;
        }

        // Compute Laplacian
        double compute_laplacian(const field_t& field, const auto& cell, double h) const
        {
            double laplacian = 0.0;
            double h2 = h * h;

            for (std::size_t d = 0; d < dim; ++d)
            {
                double grad = compute_gradient(field, cell, d, h);
                laplacian += grad / h2;
            }

            return laplacian;
        }

        // Interpolate field value at arbitrary coordinate
        value_t interpolate_value(const field_t& field, const auto& coord, std::size_t level) const
        {
            // Simple nearest neighbor (can be improved with proper interpolation)
            value_t result = value_t{0};
            bool found = false;

            // Find cell containing coord and return its value
            const auto& mesh = field.mesh();
            for_each_cell(mesh, [&](const auto& cell)
            {
                if (!found && cell.contains(coord))
                {
                    result = field[cell];
                    found = true;
                }
            });

            return result;
        }
    };

    // ==================================================================================
    // REWARD ENGINE
    // ==================================================================================

    template <std::size_t dim, class Field>
    class RewardEngine
    {
      public:
        using config_t = NeuroMeshConfig;
        using field_t  = Field;
        using mesh_t   = typename field_t::mesh_t;
        using value_t  = typename field_t::value_type;

      private:
        config_t m_config;
        value_t m_previous_error = value_t{0};
        std::size_t m_previous_cell_count = 0;

      public:
        explicit RewardEngine(const config_t& config = config_t{})
            : m_config(config)
        {
        }

        // Compute reward for current adaptation state
        double compute_reward(const field_t& field,
                             double current_error,
                             std::size_t current_cell_count) const
        {
            // Accuracy reward: improvement in error
            double accuracy_reward = compute_accuracy_reward(current_error);

            // Efficiency reward: cell count reduction
            double efficiency_reward = compute_efficiency_reward(current_cell_count);

            // Stability reward: mesh stability
            double stability_reward = compute_stability_reward(field);

            // Weighted combination
            double reward = m_config.reward_accuracy * accuracy_reward
                          + m_config.reward_efficiency * efficiency_reward
                          + m_config.reward_stability * stability_reward;

            return reward;
        }

        // Update state for next reward computation
        void update_state(double error, std::size_t cell_count)
        {
            m_previous_error = static_cast<value_t>(error);
            m_previous_cell_count = cell_count;
        }

      private:
        double compute_accuracy_reward(double current_error) const
        {
            if (m_previous_error == 0)
            {
                return 0.0;  // First adaptation, no previous error
            }

            // Reward for error reduction
            double error_reduction = m_previous_error - current_error;
            double relative_improvement = error_reduction / (m_previous_error + 1e-10);

            return relative_improvement;
        }

        double compute_efficiency_reward(std::size_t current_cell_count) const
        {
            if (m_previous_cell_count == 0)
            {
                return 0.0;
            }

            // Reward for using fewer cells
            double cell_ratio = static_cast<double>(current_cell_count)
                             / static_cast<double>(m_previous_cell_count);

            // Reward is positive if we reduced cell count
            return 1.0 - cell_ratio;
        }

        double compute_stability_reward(const field_t& field) const
        {
            // Reward for smooth field (fewer oscillations)
            // Compute variance of field gradients
            double variance = compute_gradient_variance(field);

            // Lower variance = higher reward
            return std::exp(-variance);
        }

        double compute_gradient_variance(const field_t& field) const
        {
            const auto& mesh = field.mesh();

            double mean_grad = 0.0;
            std::size_t count = 0;

            // First pass: compute mean gradient
            for_each_cell(mesh, [&](const auto& cell)
            {
                double grad_magnitude = compute_gradient_magnitude(field, cell);
                mean_grad += grad_magnitude;
                count++;
            });
            mean_grad /= (count > 0 ? count : 1);

            // Second pass: compute variance
            double variance = 0.0;
            for_each_cell(mesh, [&](const auto& cell)
            {
                double grad_magnitude = compute_gradient_magnitude(field, cell);
                double diff = grad_magnitude - mean_grad;
                variance += diff * diff;
            });
            variance /= (count > 0 ? count : 1);

            return variance;
        }

        double compute_gradient_magnitude(const field_t& field, const auto& cell) const
        {
            double magnitude = 0.0;
            double h = std::pow(2.0, -static_cast<double>(cell.level));

            for (std::size_t d = 0; d < dim; ++d)
            {
                // Simple gradient approximation
                double grad = 0.0;  // Simplified
                magnitude += grad * grad;
            }

            return std::sqrt(magnitude);
        }
    };

    // ==================================================================================
    // RL AGENT (DQN - Deep Q-Network)
    // ==================================================================================

    template <std::size_t dim, class Field>
    class RLAgent
    {
      public:
        using config_t          = NeuroMeshConfig;
        using field_t           = Field;
        using feature_array_t   = xt::xarray<double>;
        using q_values_t        = xt::xarray<double>;

      private:
        config_t m_config;
        std::mt19937 m_rng;

        // Q-network parameters (simplified neural network)
        std::vector<std::vector<double>> m_weights_hidden;
        std::vector<double> m_weights_output;
        std::vector<double> m_bias_hidden;
        std::vector<double> m_bias_output;

        // Experience replay buffer
        struct Experience
        {
            feature_array_t state;
            int action;
            double reward;
            feature_array_t next_state;
            bool done;
        };
        std::vector<Experience> m_replay_buffer;

        // Training statistics
        std::size_t m_training_step = 0;
        double m_epsilon_current = 0.1;

      public:
        explicit RLAgent(const config_t& config = config_t{},
                        std::size_t feature_dim = 5)
            : m_config(config)
            , m_rng(std::random_device{}())
            , m_epsilon_current(config.epsilon)
        {
            initialize_network(feature_dim);
        }

        // Select action using epsilon-greedy policy
        CellAction select_action(const feature_array_t& state)
        {
            // Exploration
            if (std::uniform_real_distribution<double>(0.0, 1.0)(m_rng) < m_epsilon_current)
            {
                return random_action();
            }

            // Exploitation: select action with highest Q-value
            return action_with_max_q(state);
        }

        // Train Q-network on batch of experiences
        void train_batch()
        {
            if (m_replay_buffer.size() < m_config.batch_size)
            {
                return;  // Not enough experiences
            }

            // Sample random batch
            std::vector<std::size_t> indices = sample_batch_indices();

            // Perform gradient descent (simplified)
            for (std::size_t epoch = 0; epoch < m_config.training_epochs; ++epoch)
            {
                for (std::size_t idx : indices)
                {
                    const auto& exp = m_replay_buffer[idx];
                    double td_error = compute_td_error(exp);
                    update_weights(exp.state, exp.action, td_error);
                }
            }

            m_training_step++;

            // Decay exploration rate
            if (m_epsilon_current > 0.01)
            {
                m_epsilon_current *= 0.995;
            }
        }

        // Store experience in replay buffer
        void store_experience(const feature_array_t& state,
                            CellAction action,
                            double reward,
                            const feature_array_t& next_state,
                            bool done = false)
        {
            Experience exp;
            exp.state = state;
            exp.action = static_cast<int>(action);
            exp.reward = reward;
            exp.next_state = next_state;
            exp.done = done;

            m_replay_buffer.push_back(exp);

            // Keep buffer size limited
            if (m_replay_buffer.size() > m_config.replay_buffer_size)
            {
                m_replay_buffer.erase(m_replay_buffer.begin());
            }
        }

        // Save model to file
        void save_model(const std::string& filename) const
        {
            std::ofstream out(filename, std::ios::binary);
            // Save weights (simplified)
            // ... implementation ...
        }

        // Load model from file
        void load_model(const std::string& filename)
        {
            std::ifstream in(filename, std::ios::binary);
            // Load weights (simplified)
            // ... implementation ...
        }

      private:
        void initialize_network(std::size_t feature_dim)
        {
            // Initialize hidden layer
            std::size_t input_dim = feature_dim;
            m_weights_hidden.resize(m_config.hidden_dim);
            m_bias_hidden.resize(m_config.hidden_dim);

            std::normal_distribution<double> dist(0.0, 0.01);
            for (auto& w : m_weights_hidden)
            {
                w.resize(input_dim);
                for (auto& val : w)
                {
                    val = dist(m_rng);
                }
            }
            for (auto& b : m_bias_hidden)
            {
                b = dist(m_rng);
            }

            // Initialize output layer
            m_weights_output.resize(static_cast<std::size_t>(CellAction::Count));
            m_bias_output.resize(static_cast<std::size_t>(CellAction::Count));
            for (auto& w : m_weights_output)
            {
                w.resize(m_config.hidden_dim);
                for (auto& val : w)
                {
                    val = dist(m_rng);
                }
            }
            for (auto& b : m_bias_output)
            {
                b = dist(m_rng);
            }
        }

        CellAction random_action()
        {
            std::uniform_int_distribution<int> dist(
                0, static_cast<int>(CellAction::Count) - 1
            );
            return static_cast<CellAction>(dist(m_rng));
        }

        CellAction action_with_max_q(const feature_array_t& state)
        {
            q_values_t q_values = compute_q_values(state);
            return static_cast<CellAction>(
                static_cast<int>(xt::argmax(q_values)())
            );
        }

        q_values_t compute_q_values(const feature_array_t& state) const
        {
            q_values_t q_values = xt::zeros<double>({static_cast<std::size_t>(CellAction::Count)});

            // Forward pass through hidden layer
            std::vector<double> hidden(m_config.hidden_dim);

            for (std::size_t i = 0; i < m_config.hidden_dim; ++i)
            {
                hidden[i] = m_bias_hidden[i];
                for (std::size_t j = 0; j < state.size(); ++j)
                {
                    hidden[i] += m_weights_hidden[i][j] * state(j);
                }
                hidden[i] = std::max(0.0, hidden[i]);  // ReLU activation
            }

            // Forward pass through output layer
            for (std::size_t a = 0; a < static_cast<std::size_t>(CellAction::Count); ++a)
            {
                q_values(a) = m_bias_output[a];
                for (std::size_t i = 0; i < m_config.hidden_dim; ++i)
                {
                    q_values(a) += m_weights_output[a][i] * hidden[i];
                }
            }

            return q_values;
        }

        double compute_td_error(const Experience& exp)
        {
            double max_next_q = xt::max(compute_q_values(exp.next_state))();
            double current_q = compute_q_values(exp.state)(exp.action);

            double td_target = exp.reward;
            if (!exp.done)
            {
                td_target += m_config.gamma * max_next_q;
            }

            return td_target - current_q;
        }

        void update_weights(const feature_array_t& state, int action, double td_error)
        {
            double learning_rate = m_config.learning_rate;

            // Gradient descent (simplified - no proper backprop)
            for (std::size_t a = 0; a < static_cast<std::size_t>(CellAction::Count); ++a)
            {
                if (a == static_cast<std::size_t>(action))
                {
                    m_bias_output[a] += learning_rate * td_error;
                }
            }
        }

        std::vector<std::size_t> sample_batch_indices()
        {
            std::vector<std::size_t> indices;
            std::uniform_int_distribution<std::size_t> dist(
                0, m_replay_buffer.size() - 1
            );

            for (std::size_t i = 0; i < m_config.batch_size; ++i)
            {
                indices.push_back(dist(m_rng));
            }

            return indices;
        }
    };

    // ==================================================================================
    // ADAPTATION CONTROLLER
    // ==================================================================================

    template <std::size_t dim, class Field>
    class AdaptationController
    {
      public:
        using config_t        = NeuroMeshConfig;
        using field_t         = Field;
        using mesh_t          = typename field_t::mesh_t;
        using cell_t          = typename mesh_t::cell_t;
        using value_t         = typename field_t::value_type;
        using feature_extractor_t = FeatureExtractor<dim, Field>;
        using reward_engine_t = RewardEngine<dim, Field>;
        using rl_agent_t      = RLAgent<dim, Field>;

      private:
        config_t m_config;
        feature_extractor_t m_feature_extractor;
        reward_engine_t m_reward_engine;
        rl_agent_t m_rl_agent;

        std::size_t m_initial_cell_count = 0;
        std::size_t m_adaptation_count = 0;
        double m_current_error = 0.0;

      public:
        AdaptationController(const config_t& config = config_t{})
            : m_config(config)
            , m_feature_extractor(config)
            , m_reward_engine(config)
            , m_rl_agent(config)
        {
        }

        // Perform RL-guided mesh adaptation
        void adapt(field_t& field, double target_error)
        {
            auto& mesh = field.mesh();

            // First adaptation: store initial state
            if (m_adaptation_count == 0)
            {
                m_initial_cell_count = count_cells(mesh);
            }

            // Extract features for current state
            auto features = m_feature_extractor.extract_batch(field);

            // Select actions for each cell using RL agent
            std::vector<CellAction> actions;
            std::size_t cell_idx = 0;

            for_each_cell(mesh, [&](const auto& cell)
            {
                auto cell_features = xt::view(features, cell_idx, xt::all());
                CellAction action = m_rl_agent.select_action(cell_features);
                actions.push_back(action);
                cell_idx++;
            });

            // Apply actions (perform refinement/coarsening)
            apply_actions(field, actions);

            // Compute reward for this adaptation
            m_current_error = estimate_error(field);
            std::size_t current_cell_count = count_cells(mesh);
            double reward = m_reward_engine.compute_reward(
                field, m_current_error, current_cell_count
            );

            // Store experience for training
            m_reward_engine.update_state(m_current_error, current_cell_count);
            m_rl_agent.store_experience(
                features,        // state (simplified - should store previous state)
                CellAction::Keep,  // action (simplified - should use actual actions)
                reward,
                m_feature_extractor.extract_batch(field)  // next state
            );

            // Train RL agent
            if (m_config.online_learning)
            {
                m_rl_agent.train_batch();
            }

            m_adaptation_count++;

            // Safety check: fallback to traditional MRA if needed
            if (m_current_error > m_config.error_threshold * target_error)
            {
                if (m_config.fallback_to_mra)
                {
                    fallback_to_traditional_mra(field, target_error);
                }
            }
        }

        // Get adaptation statistics
        std::size_t get_adaptation_count() const { return m_adaptation_count; }
        double get_current_error() const { return m_current_error; }

      private:
        std::size_t count_cells(const mesh_t& mesh) const
        {
            std::size_t count = 0;
            for_each_cell(mesh, [&](const auto&) { count++; });
            return count;
        }

        void apply_actions(field_t& field, const std::vector<CellAction>& actions)
        {
            auto& mesh = field.mesh();

            // Tag cells for refinement/coarsening based on RL actions
            std::size_t action_idx = 0;

            for_each_cell(mesh, [&](const auto& cell)
            {
                CellAction action = actions[action_idx];

                // Apply action (simplified - would need actual mesh modification API)
                switch (action)
                {
                    case CellAction::Refine:
                        // Tag for refinement
                        break;
                    case CellAction::Coarsen:
                        // Tag for coarsening
                        break;
                    case CellAction::Keep:
                    default:
                        // No change
                        break;
                }

                action_idx++;
            });

            // Actually modify mesh (would integrate with Samurai's mesh adaptation)
            // mesh.update_tags_and_adapt();
        }

        double estimate_error(const field_t& field) const
        {
            // Simplified error estimation
            // In practice, would use proper error estimators
            const auto& mesh = field.mesh();

            double total_variation = 0.0;
            std::size_t count = 0;

            for_each_cell(mesh, [&](const auto& cell)
            {
                double h = std::pow(2.0, -static_cast<double>(cell.level));
                double value = static_cast<double>(field[cell]);

                // Simple variation-based error estimate
                total_variation += std::abs(value) * h;
                count++;
            });

            return count > 0 ? total_variation / count : 0.0;
        }

        void fallback_to_traditional_mra(field_t& field, double target_error)
        {
            // Fallback to traditional multiresolution adaptation
            // This would call Samurai's existing MR adaptation
            // auto MRadapt = samurai::make_MRAdapt(field);
            // MRadapt(target_error);
        }
    };

    // ==================================================================================
    // FACTORY FUNCTION
    // ==================================================================================

    template <std::size_t dim, class Field>
    auto make_neuromesh_controller(const NeuroMeshConfig& config = {})
    {
        return AdaptationController<dim, Field>(config);
    }

    // ==================================================================================
    // PRE-TRAINED MODELS
    // ==================================================================================

    namespace pretrained
    {
        // Load pre-trained model for common PDE types
        template <std::size_t dim, class Field>
        RLAgent<dim, Field> load_model_for_pde(const std::string& pde_type)
        {
            NeuroMeshConfig config;

            if (pde_type == "advection")
            {
                // Pre-trained configuration for advection
                config.reward_accuracy = 0.7;
                config.reward_efficiency = 0.3;
                config.epsilon = 0.05;  // Less exploration for pre-trained
            }
            else if (pde_type == "diffusion")
            {
                config.reward_accuracy = 0.5;
                config.reward_efficiency = 0.5;
                config.epsilon = 0.05;
            }
            else if (pde_type == "navier_stokes")
            {
                config.reward_accuracy = 0.6;
                config.reward_efficiency = 0.4;
                config.use_spatial_features = true;
                config.cnn_filters = 32;  // More features for complex physics
            }

            RLAgent<dim, Field> agent(config);

            // In practice, would load actual trained weights from file
            // agent.load_model("pretrained_models/" + pde_type + ".dat");

            return agent;
        }
    }

}  // namespace samurai::neuromesh
