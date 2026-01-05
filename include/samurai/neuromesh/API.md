# NeuroMesh API Reference

Complete API documentation for NeuroMesh RL-based mesh adaptation system.

## Table of Contents

- [Configuration](#configuration)
- [Feature Extractor](#feature-extractor)
- [Reward Engine](#reward-engine)
- [RL Agent](#rl-agent)
- [Adaptation Controller](#adaptation-controller)
- [Factory Functions](#factory-functions)
- [Pre-trained Models](#pre-trained-models)

---

## Configuration

### `NeuroMeshConfig`

Main configuration struct for NeuroMesh system.

```cpp
struct NeuroMeshConfig
{
    // Feature extractor parameters
    std::size_t cnn_filters      = 16;     // Number of CNN filters
    std::size_t cnn_kernel_size  = 3;      // Kernel size for feature extraction
    bool use_spatial_features   = true;  // Extract spatial gradients, curvature

    // RL agent parameters
    std::size_t hidden_dim       = 128;    // Hidden layer size
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
    bool fallback_to_mra         = true;   // Fallback to traditional MRA
    double error_threshold       = 2.0;    // Error threshold for fallback

    // I/O parameters
    bool save_training_data      = false;  // Save experience replay
    std::string checkpoint_dir   = "./neuromesh_checkpoints";
    std::string model_filename   = "neuromesh_model.dat";
};
```

**Members:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cnn_filters` | `std::size_t` | `16` | Number of convolutional filters in feature extractor |
| `use_spatial_features` | `bool` | `true` | Enable gradient and curvature feature extraction |
| `hidden_dim` | `std::size_t` | `128` | Size of hidden layer in Q-network |
| `learning_rate` | `double` | `0.01` | Learning rate for Q-network updates |
| `gamma` | `double` | `0.99` | Discount factor for future rewards |
| `epsilon` | `double` | `0.1` | Initial exploration rate (ε-greedy) |
| `reward_accuracy` | `double` | `0.6` | Weight for accuracy improvement in reward |
| `reward_efficiency` | `double` | `0.4` | Weight for cell count reduction in reward |
| `adapt_interval` | `std::size_t` | `10` | Number of timesteps between adaptations |
| `online_learning` | `bool` | `true` | Enable learning during simulation |
| `fallback_to_mra` | `bool` | `true` | Enable fallback to traditional MRA on error |

---

## Feature Extractor

### `FeatureExtractor<dim, Field>`

Extracts features from field for RL decision making.

```cpp
template <std::size_t dim, class Field>
class FeatureExtractor
{
public:
    using config_t       = NeuroMeshConfig;
    using field_t        = Field;
    using feature_array_t = xt::xarray<double>;

    explicit FeatureExtractor(const config_t& config = config_t{});

    // Extract features for single cell
    feature_array_t extract_features(const field_t& field, const auto& cell) const;

    // Extract features for entire mesh (batch)
    feature_array_t extract_batch(const field_t& field) const;
};
```

**Methods:**

#### `extract_features(field, cell)`

Extract features for a single cell.

**Parameters:**
- `field`: The field to extract features from
- `cell`: The cell to extract features for

**Returns:**
- `feature_array_t`: Array of feature values
  - If `use_spatial_features = true`: `[value, level, gradient_0, ..., gradient_dim-1, laplacian]`
  - If `use_spatial_features = false`: `[value, level]`

**Example:**
```cpp
FeatureExtractor<2, Field> extractor(config);
auto features = extractor.extract_features(u, cell);

std::cout << "Value: " << features(0) << "\n";
std::cout << "Level: " << features(1) << "\n";
std::cout << "Gradient X: " << features(2) << "\n";
```

#### `extract_batch(field)`

Extract features for entire mesh at once.

**Parameters:**
- `field`: The field to extract features from

**Returns:**
- `feature_array_t`: Matrix of shape `(num_cells, num_features)`

**Example:**
```cpp
auto all_features = extractor.extract_batch(u);
std::cout << "Shape: " << all_features.shape() << "\n";  // (N_cells, N_features)
```

---

## Reward Engine

### `RewardEngine<dim, Field>`

Computes rewards for RL training based on adaptation performance.

```cpp
template <std::size_t dim, class Field>
class RewardEngine
{
public:
    using config_t = NeuroMeshConfig;
    using field_t  = Field;

    explicit RewardEngine(const config_t& config = config_t{});

    // Compute reward for current state
    double compute_reward(const field_t& field,
                         double current_error,
                         std::size_t current_cell_count) const;

    // Update state for next reward computation
    void update_state(double error, std::size_t cell_count);
};
```

**Methods:**

#### `compute_reward(field, current_error, current_cell_count)`

Compute reward for the current adaptation state.

**Parameters:**
- `field`: Current field state
- `current_error`: Current error estimate
- `current_cell_count`: Current number of cells

**Returns:**
- `double`: Computed reward value
  - Positive: improvement over previous state
  - Negative: worsening compared to previous state

**Reward Formula:**
```cpp
reward = w_accuracy * accuracy_reward
        + w_efficiency * efficiency_reward
        + w_stability * stability_reward
```

**Example:**
```cpp
RewardEngine<2, Field> reward_engine(config);
double reward = reward_engine.compute_reward(u, error, cell_count);

if (reward > 0) {
    std::cout << "Good adaptation!\n";
}
```

#### `update_state(error, cell_count)`

Update internal state for next reward computation.

**Parameters:**
- `error`: Current error estimate
- `cell_count`: Current cell count

**Example:**
```cpp
reward_engine.update_state(current_error, current_cells);
```

---

## RL Agent

### `RLAgent<dim, Field>`

Deep Q-Network agent for action selection.

```cpp
template <std::size_t dim, class Field>
class RLAgent
{
public:
    using config_t          = NeuroMeshConfig;
    using feature_array_t   = xt::xarray<double>;

    explicit RLAgent(const config_t& config = config_t{},
                    std::size_t feature_dim = 5);

    // Select action using epsilon-greedy policy
    CellAction select_action(const feature_array_t& state);

    // Train on batch of experiences
    void train_batch();

    // Store experience in replay buffer
    void store_experience(const feature_array_t& state,
                        CellAction action,
                        double reward,
                        const feature_array_t& next_state,
                        bool done = false);

    // Save/load model
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};
```

**Methods:**

#### `select_action(state)`

Select action for given state using ε-greedy policy.

**Parameters:**
- `state`: Feature array representing current state

**Returns:**
- `CellAction`: Selected action (`Keep`, `Refine`, or `Coarsen`)

**Example:**
```cpp
auto features = extractor.extract_features(u, cell);
CellAction action = agent.select_action(features);

switch (action) {
    case CellAction::Refine:
        std::cout << "Refining cell\n";
        break;
    case CellAction::Coarsen:
        std::cout << "Coarsening cell\n";
        break;
    default:
        std::cout << "Keeping cell\n";
}
```

#### `train_batch()`

Train Q-network on a batch of experiences from replay buffer.

**Prerequisites:**
- Replay buffer must have at least `batch_size` experiences

**Example:**
```cpp
if (replay_buffer.size() >= config.batch_size) {
    agent.train_batch();
}
```

#### `store_experience(state, action, reward, next_state, done)`

Store a transition in the experience replay buffer.

**Parameters:**
- `state`: Current state features
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state features
- `done`: Whether episode is done (default: `false`)

**Example:**
```cpp
agent.store_experience(
    current_state,
    CellAction::Refine,
    reward,
    next_state,
    false  // episode not done
);
```

---

## Adaptation Controller

### `AdaptationController<dim, Field>`

Main controller for RL-guided mesh adaptation.

```cpp
template <std::size_t dim, class Field>
class AdaptationController
{
public:
    using config_t = NeuroMeshConfig;

    explicit AdaptationController(const config_t& config = config_t{});

    // Perform RL-guided mesh adaptation
    void adapt(field_t& field, double target_error);

    // Get statistics
    std::size_t get_adaptation_count() const;
    double get_current_error() const;
};
```

**Methods:**

#### `adapt(field, target_error)`

Perform RL-guided mesh adaptation on the field.

**Parameters:**
- `field`: Field to adapt (in-out parameter)
- `target_error`: Target error tolerance

**Process:**
1. Extract features from current field
2. Select actions for each cell using RL agent
3. Apply refinement/coarsening actions
4. Compute reward for this adaptation
5. Store experience and train agent

**Example:**
```cpp
AdaptationController<2, Field> controller(config);

double target_error = 1e-4;
for (std::size_t n = 0; n < nsteps; ++n) {
    // ... numerical scheme ...

    // Adapt every 10 steps
    if (n % 10 == 0) {
        controller.adapt(u, target_error);
    }
}
```

#### `get_adaptation_count()`

Get number of adaptations performed.

**Returns:**
- `std::size_t`: Number of adaptations

#### `get_current_error()`

Get current error estimate.

**Returns:**
- `double`: Current error estimate

---

## Factory Functions

### `make_neuromesh_controller<dim, Field>(config)`

Factory function to create an adaptation controller.

**Parameters:**
- `config`: NeuroMesh configuration (default: `NeuroMeshConfig{}`)

**Returns:**
- `AdaptationController<dim, Field>`: Configured controller

**Example:**
```cpp
auto controller = make_neuromesh_controller<2, Field>(config);
```

---

## Pre-trained Models

### `load_model_for_pde<dim, Field>(pde_type)`

Load a pre-trained model for a specific PDE type.

**Parameters:**
- `pde_type`: String identifier for PDE type
  - `"advection"`: Advection-dominant problems
  - `"diffusion"`: Diffusion-dominant problems
  - `"navier_stokes"`: Fluid dynamics problems

**Returns:**
- `RLAgent<dim, Field>`: Pre-trained agent

**Example:**
```cpp
using namespace samurai::neuromesh::pretrained;

auto agent = load_model_for_pde<2, Field>("advection");
```

---

## Enums

### `CellAction`

Action type for cell refinement decisions.

```cpp
enum class CellAction : int
{
    Keep    = 0,  // Maintain current level
    Refine  = 1,  // Increase refinement level
    Coarsen = 2,  // Decrease refinement level
    Count    = 3
};
```

**Helper Function:**
```cpp
std::string action_to_string(CellAction action);
```

---

## Usage Examples

### Example 1: Basic Usage

```cpp
#include <samurai/neuromesh/neuromesh.hpp>

// Create field
auto u = samurai::make_scalar_field<double>("u", mesh);

// Create controller
auto controller = make_neuromesh_controller<2, decltype(u)>();

// Time loop
for (std::size_t n = 0; n < nsteps; ++n) {
    // Adapt mesh
    controller.adapt(u, 1e-4);

    // Numerical scheme
    update_ghost_mr(u);
    unp1 = u - dt * flux(u);
    std::swap(u.array(), unp1.array());
}
```

### Example 2: Custom Configuration

```cpp
NeuroMeshConfig config;
config.reward_accuracy = 0.7;
config.reward_efficiency = 0.3;
config.use_spatial_features = true;
config.adapt_interval = 5;

auto controller = make_neuromesh_controller<2, Field>(config);
```

### Example 3: Pre-trained Model

```cpp
using namespace samurai::neuromesh::pretrained;

auto agent = load_model_for_pde<2, Field>("advection");
// Use agent in custom controller or directly
```

---

## Performance Considerations

### Memory Usage

| Component | Memory (approx.) |
|-----------|------------------|
| Feature Extractor | ~1 MB |
| RL Agent (Q-network) | ~100 KB |
| Experience Replay (10k) | ~5 MB |
| **Total** | ~6 MB |

### Computational Cost

| Operation | Cost (relative) |
|------------|-----------------|
| Feature extraction | 1x |
| Action selection | 0.1x |
| Training (batch) | 5x |
| **Total overhead** | ~10% of simulation |

### Optimization Tips

1. **Reduce `hidden_dim`** for faster training
2. **Decrease `replay_buffer_size`** to save memory
3. **Disable `online_learning`** for inference-only
4. **Increase `adapt_interval`** to reduce overhead

---

## Thread Safety

**Current implementation is NOT thread-safe.**

For multi-threaded usage:
- Create separate controller per thread
- Or protect with mutex (not implemented)

---

## Future API Additions

- [ ] GPU-accelerated feature extraction
- [ ] Asynchronous training
- [ ] Distributed RL for multi-physics
- [ ] Real-time monitoring hooks
- [ ] JSON configuration file support
