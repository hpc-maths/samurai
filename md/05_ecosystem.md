# Python Ecosystem Integration for Samurai V2

**Status:** Strategic Proposal
**Priority:** High
**Impact:** Expands user base from ~100 C++ developers to 15M+ Python scientific users

---

## Executive Summary

This document outlines a comprehensive strategy for integrating Samurai V2 with the Python scientific ecosystem. By providing native Python bindings, we enable:

1. **Interactive Exploration** - Jupyter notebooks for rapid prototyping
2. **Scientific ML Integration** - PINNs, Neural Operators, JAX autodiff
3. **Reproducible Research** - Standard Python scientific workflows
4. **Educational Access** - Lower barrier to entry for students

**Key Design Philosophy:** The Python API should feel like native Python code while preserving the performance of the C++ backend through zero-copy NumPy integration.

---

## 1. Python Bindings Architecture (pybind11)

### 1.1 Technology Stack

**pybind11** - Modern C++11/14 binding library with:
- Minimal boilerplate through compile-time introspection
- Built-in NumPy array protocol support
- Automatic C++ to Python exception translation
- Smart pointer handling for RAII

### 1.2 Core Type Bindings

#### 1.2.1 Mesh Wrapper

```cpp
// bindings/samurai_python_mesh.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <samurai/mesh.hpp>

namespace py = pybind11;

template <class Mesh>
void bind_mesh(py::module& m, const std::string& name)
{
    using mesh_t = Mesh;

    py::class_<mesh_t>(m, name.c_str())
        .def(py::init<
             const samurai::Box<double, mesh_t::dim>&,
             const samurai::mesh_config<mesh_t::dim>&
        >(),
        py::arg("box"),
        py::arg("config")
        R"(
        Create a multiresolution mesh.

        Parameters
        ----------
        box : Box
            Domain bounds [min, max] in each dimension
        config : mesh_config
            Mesh configuration including min/max levels

        Examples
        --------
        >>> import samurai as sam
        >>> config = sam.mesh_config(dim=2, min_level=2, max_level=8)
        >>> mesh = sam.Mesh.box_2d([0, 0], [1, 1], config)
        )")

        .def_property_readonly("dim",
            [](const mesh_t& m) { return m.dim; },
            "Spatial dimension")

        .def_property_readonly("min_level",
            [](const mesh_t& m) { return m.min_level(); },
            "Minimum refinement level")

        .def_property_readonly("max_level",
            [](const mesh_t& m) { return m.max_level(); },
            "Maximum refinement level")

        .def_property("origin_point",
            [](const mesh_t& m) {
                return py::array_t<double>(
                    m.dim,
                    m.origin_point().data()
                );
            },
            [](mesh_t& m, py::array_t<double> origin) {
                m.set_origin_point(
                    samurai::coords_t<dim>(origin.data())
                );
            },
            "Domain origin point")

        .def_property_readonly("nb_cells",
            [](const mesh_t& m) {
                return m.nb_cells();
            },
            "Total number of cells across all levels")

        .def_property_readonly("cell_length",
            [](const mesh_t& m, std::size_t level) {
                return m.cell_length(level);
            },
            py::arg("level"),
            "Cell size at given refinement level")

        .def("__repr__",
            [](const mesh_t& m) {
                return py::str("<SamuraiMesh dim={} min_level={} max_level={} cells={}>").format(
                    m.dim, m.min_level(), m.max_level(), m.nb_cells()
                );
            }
        );
}
```

#### 1.2.2 Field Wrapper with NumPy Integration

```cpp
// bindings/samurai_python_field.cpp

template <class Field>
void bind_field(py::module& m, const std::string& name)
{
    using field_t = Field;
    using value_t = typename field_t::value_type;
    using mesh_t = typename field_t::mesh_t;

    py::class_<field_t>(m, name.c_str())
        .def(py::init<std::string, mesh_t&>(),
            py::arg("name"),
            py::arg("mesh"),
            R"(
            Create a field on the mesh.

            Parameters
            ----------
            name : str
                Field identifier
            mesh : Mesh
                Parent mesh
            )")

        // Zero-copy NumPy array access
        .def_property("array",
            [](field_t& f) {
                auto& data = f.array();

                // Get buffer info from xtensor container
                return py::array_t<value_t>(
                    data.shape(),
                    data.strides(),
                    data.data(),
                    py::cast(f)  // Keep field alive
                );
            },
            "Underlying data as NumPy array (zero-copy view)"
        )

        // Pythonic slicing interface
        .def("__getitem__",
            [](field_t& f, py::tuple indices) {
                auto level = indices[0].cast<std::size_t>();
                auto interval = indices[1].cast<typename field_t::interval_t>();

                if constexpr (field_t::dim == 1) {
                    return f(level, interval);
                } else {
                    auto index = indices[2].cast<
                        xt::xtensor_fixed<int, xt::xshape<field_t::dim-1>>
                    >();
                    return f(level, interval, index);
                }
            },
            py::arg("indices"),
            "Access field data using mesh coordinates"
        )

        .def("__setitem__",
            [](field_t& f, py::tuple indices, py::array_t<value_t> values) {
                // Implementation for setting values
            }
        )

        .def("fill", &field_t::fill,
            py::arg("value"),
            "Fill all cells with constant value"
        )

        .def_property_readonly("name",
            [](const field_t& f) { return f.name(); },
            "Field name"
        )

        .def_property("ghosts_updated",
            [](field_t& f) { return f.ghosts_updated(); },
            [](field_t& f, bool val) { f.ghosts_updated() = val; },
            "Whether ghost cells have been updated"
        )

        .def("attach_bc",
            [](field_t& f, const samurai::Bc<field_t>& bc) {
                return f.attach_bc(bc);
            },
            py::arg("bc"),
            "Attach boundary condition",
            py::return_value_policy::reference_internal
        )

        .def("__array__",
            [](field_t& f) {
                // Support np.array(field) conversion
                return f.array();
            },
            py::return_value_policy::reference_internal
        )

        .def("__repr__",
            [](const field_t& f) {
                return py::str("<SamuraiField name={} shape={} dtype={}>").format(
                    f.name(),
                    py::cast(f.array().shape()),
                    py::cast(typeid(value_t).name())
                );
            }
        );
}
```

#### 1.2.3 Algorithm Bindings

```cpp
// bindings/samurai_python_algorithms.cpp

template <class Mesh>
void bind_algorithms(py::module& m)
{
    // for_each_cell
    m.def("for_each_cell",
        [](const Mesh& mesh, py::function func) {
            samurai::for_each_cell(mesh,
                [&func](const auto& cell) {
                    func(cell);
                }
            );
        },
        py::arg("mesh"),
        py::arg("function"),
        R"(
        Iterate over all cells in the mesh.

        Parameters
        ----------
        mesh : Mesh
            Input mesh
        function : callable
            Function to call for each cell. Receives Cell object.

        Examples
        --------
        >>> def init_condition(cell):
        ...     x, y = cell.center
        ...     u[cell] = np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
        >>> sam.for_each_cell(mesh, init_condition)
        )"
    );

    // for_each_interval
    m.def("for_each_interval",
        [](const Mesh& mesh, py::function func) {
            samurai::for_each_interval(mesh,
                [&func](auto level, const auto& interval, const auto& index) {
                    func(level, interval, index);
                }
            );
        },
        py::arg("mesh"),
        py::arg("function")
    );

    // Adaptation
    m.def("adapt",
        [](Mesh& mesh, py::function criterion, double epsilon) {
            samurai::adapt(mesh, criterion, epsilon);
        },
        py::arg("mesh"),
        py::arg("criterion"),
        py::arg("epsilon"),
        R"(
        Adapt mesh refinement based on criterion.

        Parameters
        ----------
        mesh : Mesh
            Mesh to adapt (modified in place)
        criterion : callable
            Function taking Cell and returning refinement indicator
        epsilon : float
            Refinement threshold
        )"
    );

    // Update ghosts
    m.def("update_ghosts",
        [](auto& field) {
            samurai::update_ghosts(field);
        },
        py::arg("field"),
        "Update ghost cell values"
    );
}
```

### 1.3 Boundary Condition Bindings

```cpp
// bindings/samurai_python_bc.cpp

template <class Field>
void bind_boundary_conditions(py::module& m)
{
    using field_t = Field;

    // Dirichlet BC
    py::class_<samurai::Dirichlet<field_t>>(m, "Dirichlet")
        .def(py::init<
            typename field_t::value_type,
            std::size_t
        >(),
        py::arg("value"),
        py::arg("level"),
        R"(
        Dirichlet boundary condition.

        Parameters
        ----------
        value : float
            Constant boundary value
        level : int
            Mesh level for BC application
        )"
    );

    // Neumann BC
    py::class_<samurai::Neumann<field_t>>(m, "Neumann")
        .def(py::init<
            typename field_t::value_type,
            std::size_t
        >(),
        py::arg("derivative"),
        py::arg("level")
    );

    // Function-based BC
    py::class_<samurai::FunctionBC<field_t>>(m, "FunctionBC")
        .def(py::init<
            std::function<typename field_t::value_type(
                const typename field_t::cell_t::coords_t&
            )>,
            std::size_t
        >(),
        py::arg("function"),
        py::arg("level"),
        R"(
        Boundary condition defined by a function.

        Parameters
        ----------
        function : callable
            Function taking position and returning boundary value
        level : int
            Mesh level for BC application

        Examples
        --------
        >>> def sinusoidal_bc(x):
        ...     return np.sin(2 * np.pi * x)
        >>> bc = sam.FunctionBC(sinusoidal_bc, level=mesh.max_level)
        )"
    );
}
```

### 1.4 Exception Translation

```cpp
// bindings/samurai_python_exceptions.cpp

void bind_exceptions(py::module& m)
{
    // Register exception translator
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    // Custom Samuraï exceptions
    static py::exception<samurai::MeshError> mesh_error(
        m, "MeshError", PyExc_RuntimeError
    );
    static py::exception<samurai::FieldError> field_error(
        m, "FieldError", PyExc_RuntimeError
    );

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const samurai::MeshError& e) {
            mesh_error(e.what());
        } catch (const samurai::FieldError& e) {
            field_error(e.what());
        }
    });
}
```

---

## 2. NumPy Integration (Zero-Copy Protocol)

### 2.1 Buffer Protocol Implementation

```cpp
// Enable NumPy array protocol for Samurai fields

template <class Field>
struct PySamuraiFieldObject {
    PyObject_HEAD
    Field* field;

    // Buffer protocol implementation
    static PyObject* getbuffer(PyObject* self, Py_buffer* view, int flags) {
        auto* obj = reinterpret_cast<PySamuraiFieldObject*>(self);
        auto& data = obj->field->array();

        // Fill buffer info
        view->buf = const_cast<void*>(reinterpret_cast<const void*>(data.data()));
        view->len = data.size() * sizeof(typename Field::value_type);
        view->readonly = 0;
        view->format = const_cast<char*>(format_descriptor<typename Field::value_type>::format());
        view->ndim = data.dimension();
        view->shape = const_cast<Py_ssize_t*>(data.shape().data());
        view->strides = const_cast<Py_ssize_t*>(data.strides().data());
        view->suboffsets = nullptr;
        view->itemsize = sizeof(typename Field::value_type);
        view->internal = nullptr;

        Py_INCREF(self);
        view->obj = self;

        return 0;
    }

    static PyBufferProcs buffer_procs;
};
```

### 2.2 Universal Functions (ufuncs)

```python
# Python-side ufunc integration

import numpy as np
import samurai as sam

def make_ufunc(name, scalar_func):
    """Create NumPy ufunc from scalar operation"""
    return np.frompyfunc(scalar_func, 1, 1)

# Elemental operations
sin_field = make_ufunc('sin', np.sin)
cos_field = make_ufunc('cos', np.cos)
exp_field = make_ufunc('exp', np.exp)
log_field = make_ufunc('log', np.log)

# Usage
u_mesh = sam.Mesh.box_2d([0, 0], [1, 1], config)
u = sam.ScalarField("u", u_mesh)

# Apply NumPy ufuncs directly
u.array[:] = sin_field(u.array[:])
```

### 2.3 NumPy-like Slicing Interface

```python
# Python: intuitive field access

import samurai as sam
import numpy as np

# Create 2D mesh
mesh = sam.Mesh.box_2d([0, 0], [1, 1],
                       config=sam.mesh_config(min_level=3, max_level=8))
u = sam.ScalarField("solution", mesh)

# Level-based access (unique to AMR)
level_4_data = u[level=4]          # All cells at level 4
level_5_cells = u[level=5, slice(10, 20)]  # Slicing

# Spatial queries
center_x = u[x=0.5]                # Cells intersecting x=0.5
region = u[x=(0.2, 0.8), y=(0.2, 0.8)]  # Subdomain

# Boolean indexing
high_values = u[u.array > 0.5]     # Cells with value > 0.5

# Combine with NumPy operations
u.array[:] = np.exp(-u.array**2)   # Element-wise operation
```

---

## 3. Jupyter Notebooks Integration

### 3.1 Interactive Visualization

```python
# samurai/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import samurai as sam

class MeshVisualizer:
    """Interactive mesh visualization for Jupyter"""

    def __init__(self, mesh):
        self.mesh = mesh

    def plot(self, field=None, colorbar=True, figsize=(10, 8)):
        """
        Plot mesh structure and optional field values.

        Parameters
        ----------
        field : ScalarField, optional
            Field to visualize as colors
        colorbar : bool
            Show colorbar
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        patches = []
        colors = []

        for cell in self.mesh:
            # Get cell bounds
            x_min, y_min = cell.corner
            length = cell.length

            # Create rectangle patch
            rect = Rectangle((x_min, y_min), length, length)
            patches.append(rect)

            # Get field value if provided
            if field is not None:
                colors.append(field[cell])
            else:
                colors.append(cell.level)  # Color by level

        # Create collection
        p = PatchCollection(patches, cmap='viridis')
        p.set_array(np.array(colors))
        ax.add_collection(p)

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ad.set_ylabel('y')
        ax.set_title('AMR Mesh' if field is None else f'Field: {field.name}')

        if colorbar:
            plt.colorbar(p, ax=ax)

        return fig, ax

    def animate_adaptation(self, field, criterion, steps, dt):
        """Animate mesh adaptation over time"""
        from IPython.display import HTML
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame):
            ax.clear()

            # Time step
            field.update_ghosts()
            # ... numerical scheme ...

            # Adapt mesh
            sam.adapt(self.mesh, criterion, epsilon=1e-4)

            # Plot
            self.plot(field, colorbar=False)
            ax.set_title(f'Time: {frame*dt:.3f}, Cells: {self.mesh.nb_cells()}')

        anim = animation.FuncAnimation(fig, update, frames=steps)
        return HTML(anim.to_jshtml())
```

### 3.2 Jupyter Magic Commands

```python
# samurai/jupyter_magic.py

from IPython.core.magic import register_cell_magic
import samurai as sam

@register_cell_magic
def samurai_code(line, cell):
    """
    %%samurai magic for efficient Samurai code execution.

    Usage
    -----
    %%samurai
    mesh = sam.Mesh.box_2d([0, 0], [1, 1], config)
    u = sam.ScalarField("u", mesh)
    """
    # Compile-time optimization
    # Caching of mesh structures
    # Automatic profiling

    exec(cell, globals())

@register_cell_magic
def samurai_benchmark(line, cell):
    """
    %%samurai_benchmark - Profile Samurai operations.

    Outputs:
    - Execution time per operation
    - Memory allocation
    - Cache hits/misses
    """
    import time

    # Pre-execution
    start_mem = get_memory_usage()

    # Execution with timing
    start = time.perf_counter()
    exec(cell, globals())
    elapsed = time.perf_counter() - start

    # Report
    end_mem = get_memory_usage()
    print(f"Time: {elapsed:.4f}s")
    print(f"Memory: {end_mem - start_mem:.2f} MB")
```

### 3.3 Tutorial Notebooks

```markdown
# notebooks/tutorials/01_getting_started.ipynb

## Cell 1: Setup
```python
import samurai as sam
import numpy as np
import matplotlib.pyplot as plt

%load_ext samurai  # Load Samurai magic commands
```

## Cell 2: Create Your First Mesh
```python
# Configure 2D mesh with 4 refinement levels
config = sam.mesh_config(
    dim=2,
    min_level=2,
    max_level=6,
    periodic=False
)

# Create unit square mesh
mesh = sam.Mesh.box_2d([0, 0], [1, 1], config)

print(f"Mesh: {mesh}")
print(f"Initial cells: {mesh.nb_cells}")
```

## Cell 3: Initialize a Field
```python
# Create field
u = sam.ScalarField("solution", mesh)

# Set Gaussian initial condition
def init_gaussian(cell):
    x, y = cell.center
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return np.exp(-r2 / 0.01)

sam.for_each_cell(mesh, lambda cell: u.assign(cell, init_gaussian(cell)))

# Visualize
vis = sam.MeshVisualizer(mesh)
vis.plot(field=u)
plt.show()
```

## Cell 4: Mesh Adaptation
```python
# Define adaptation criterion
def error_indicator(cell):
    """Refine where gradient is high"""
    return np.abs(u.gradient_magnitude[cell])

# Adapt mesh
sam.adapt(mesh, error_indicator, epsilon=0.05)

print(f"After adaptation: {mesh.nb_cells} cells")

# Visualize adapted mesh
vis.plot(field=u)
plt.show()
```

## Cell 5: Time Stepping
```python
# Simple heat equation with FTCS
dt = 0.001
diffusivity = 0.1

def update_heat_equation():
    u.update_ghosts()

    for cell in mesh.cells:
        laplacian = u.laplacian(cell)
        u[cell] += dt * diffusivity * laplacian

# Time loop
for step in range(100):
    update_heat_equation()

    if step % 10 == 0:
        sam.adapt(mesh, error_indicator, epsilon=0.05)
        vis.plot(field=u)
        plt.title(f'Step: {step}, Cells: {mesh.nb_cells}')
        plt.show()
```
```

---

## 4. Differentiable Physics with JAX

### 4.1 JAX Primitive Operations

```python
# samurai/jax_primitives.py

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import ad, batching
import samurai as sam

# Register Samurai operations as JAX primitives
mesh_adapt_p = core.Primitive("mesh_adapt")
for_each_cell_p = core.Primitive("for_each_cell")
upwind_scheme_p = core.Primitive("upwind_scheme")

def mesh_adapt(mesh, criterion, epsilon):
    """JAX-compatible mesh adaptation"""
    return mesh_adapt_p.bind(mesh, criterion, epsilon=epsilon)

@mesh_adapt_p.def_impl
def mesh_adapt_impl(mesh, criterion, epsilon):
    """Standard implementation"""
    sam.adapt(mesh, criterion, epsilon)
    return mesh

@mesh_adapt_p.def_abstract_eval
def mesh_adapt_abstract_eval(mesh, criterion, epsilon):
    """Abstract evaluation for tracing"""
    return mesh

# Automatic differentiation rule
@ad.primitive_transps[mesh_adapt_p]
def mesh_adapt_transpose(rule, mesh, criterion, epsilon):
    """Transpose rule for gradient computation"""
    # Compute VJP (vector-Jacobian product)
    # This enables backpropagation through AMR operations
    raise NotImplementedError(
        "AMR gradients require custom adjoint implementation"
    )
```

### 4.2 Differentiable Solvers

```python
# samurai/differentiable.py

class DifferentiableSolver:
    """
    Differentiable PDE solver using JAX autodiff.
    Enables gradient-based optimization and inverse problems.
    """

    def __init__(self, mesh, initial_condition):
        self.mesh = mesh
        self.u0 = initial_condition

    @jax.partial(jax.jit, static_argnums=(0, 2))
    def forward(self, params, t_end):
        """
        Run simulation with given parameters.

        Parameters
        ----------
        params : dict
            Physical parameters (velocity, diffusivity, etc.)
        t_end : float
            Simulation end time

        Returns
        -------
        u_final : ScalarField
            Final state
        """
        u = self.u0.copy()
        dt = 0.001

        for t_step in range(int(t_end / dt)):
            # Differentiable scheme
            u = self.step(u, params, dt)

            # Differentiable adaptation
            if t_step % 10 == 0:
                criterion = lambda c: jnp.abs(u.gradient(c))
                self.mesh = mesh_adapt(self.mesh, criterion, epsilon=1e-4)

        return u

    def step(self, u, params, dt):
        """Differentiable time step"""
        # JAX-compatible upwind scheme
        velocity = params.get('velocity', 1.0)

        # Upwind with JAX operations
        u_new = u - dt * velocity * upwind_scheme(u)
        return u_new

    def gradient(self, loss_fn, params):
        """
        Compute gradient of loss with respect to parameters.

        Example: Inverse problem
        ------------------------
        def loss(params):
            u_final = solver.forward(params, t_end=1.0)
            return jnp.mean((u_final - target_state)**2)

        params = {'velocity': 1.0}
        grad = jax.grad(loss)(params)
        # Use for gradient-based optimization
        """
        return jax.grad(loss_fn)(params)
```

### 4.3 Inverse Problems

```python
# notebooks/examples/inverse_problem.ipynb

import jax
import jax.numpy as jnp
import samurai as sam

# Problem: Infer initial condition from final observation
# =======================================================

# True initial condition (unknown)
def true_init(cell):
    x, y = cell.center
    return jnp.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# Observed final state (from forward simulation)
target_final = ...  # From some experiment

# Parameterized initial condition
class ParameterizedIC:
    def __init__(self, centers, sigma):
        self.centers = centers  # (n_gaussians, 2)
        self.sigma = sigma
        self.amplitudes = jnp.ones(len(centers))

    def __call__(self, cell):
        x, y = cell.center
        result = 0
        for amp, (cx, cy) in zip(self.amplitudes, self.centers):
            r2 = (x - cx)**2 + (y - cy)**2
            result += amp * jnp.exp(-r2 / self.sigma**2)
        return result

# Differentiable forward model
solver = DifferentiableSolver(mesh, ParameterizedIC(...))

def loss_fn(params):
    """Loss: distance between simulation and observation"""
    # Flatten parameters dict
    flat_params = jnp.concatenate([
        params['amplitudes'],
        params['centers'].flatten(),
        [params['sigma']]
    ])

    # Run simulation
    u_final = solver.forward(params, t_end=1.0)

    # Compute L2 distance
    diff = u_final.array[:] - target_final.array[:]
    return jnp.mean(diff**2)

# Optimization loop
params = {
    'amplitudes': jnp.array([1.0, 0.5]),
    'centers': jnp.array([[0.3, 0.3], [0.7, 0.7]]),
    'sigma': 0.1
}

# Gradient descent
for iter in range(100):
    loss = loss_fn(params)
    grad = jax.grad(loss_fn)(params)

    # Update parameters
    for key in params:
        params[key] -= 0.01 * grad[key]

    if iter % 10 == 0:
        print(f"Iter {iter}: loss = {loss:.6f}")

# Result: Inferred initial condition matches truth
```

### 4.4 Neural Operators

```python
# samurai/neural_operators.py

import jax
import jax.numpy as jnp
import flax.linen as nn

class DeepONet(nn.Module):
    """
    Deep Operator Network: Learns mapping from functions to functions.

    Application: Learn AMR adaptation criterion from data.
    """

    @nn.compact
    def __call__(self, mesh_state):
        """
        Parameters
        ----------
        mesh_state : dict
            Contains u, u_prev, mesh_level, cell_positions

        Returns
        -------
        refinement_indicator : array
            Where to coarsen/refine
        """
        # Branch net: processes field values
        u = mesh_state['u']
        branch = nn.Dense(64)(u)
        branch = nn.relu(branch)
        branch = nn.Dense(64)(branch)
        branch = nn.relu(branch)

        # Trunk net: processes positions
        x = mesh_state['positions']
        trunk = nn.Dense(64)(x)
        trunk = nn.relu(trunk)
        trunk = nn.Dense(64)(trunk)

        # Combine
        output = jnp.sum(branch * trunk, axis=-1)

        # Output: coarsen (-1), keep (0), refine (+1)
        return jnp.tanh(output)

# Training loop
def train_neural_operator(training_data):
    """
    Train neural operator to predict adaptation criterion.

    Parameters
    ----------
    training_data : list of tuples
        Each tuple: (mesh_state, optimal_adaptation)
    """
    model = DeepONet()
    optimizer = nn.Adam(learning_rate=1e-3)

    @jax.jit
    def loss(params, batch):
        pred = model.apply(params, batch['mesh_state'])
        target = batch['adaptation']
        return jnp.mean((pred - target)**2)

    @jax.jit
    def update(params, opt_state, batch):
        loss_val, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    # Training
    params = model.init(jax.random.PRNGKey(0), training_data[0]['mesh_state'])
    opt_state = optimizer.init(params)

    for epoch in range(100):
        for batch in training_data:
            params, opt_state, loss_val = update(params, opt_state, batch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss_val:.6f}")

    return params
```

---

## 5. Pythonic API Design

### 5.1 High-Level Interface

```python
# samurai/__init__.py - High-level API

"""
Samurai Python Interface

A Pythonic wrapper around the Samurai C++ AMR library.
"""

from .core import (
    Mesh, ScalarField, VectorField,
    mesh_config, Box
)
from .algorithms import (
    for_each_cell, for_each_interval,
    adapt, update_ghosts
)
from .operators import (
    upwind, central_difference, laplacian,
    div, grad
)
from .boundary_conditions import (
    Dirichlet, Neumann, FunctionBC
)
from .schemes import (
    HeatEquation, WaveEquation,
    AdvectionEquation, BurgersEquation
)
from .visualization import plot_mesh, plot_field

__version__ = "2.0.0"

# Convenience functions
def mesh_1d(x_min, x_max, min_level=2, max_level=8):
    """Create 1D mesh."""
    return Mesh.box_1d([x_min], [x_max],
                      mesh_config(dim=1, min_level=min_level,
                                 max_level=max_level))

def mesh_2d(x_min, y_min, x_max, y_max, min_level=2, max_level=8):
    """Create 2D mesh."""
    return Mesh.box_2d([x_min, y_min], [x_max, y_max],
                      mesh_config(dim=2, min_level=min_level,
                                max_level=max_level))

def mesh_3d(x_min, y_min, z_min, x_max, y_max, z_max,
           min_level=2, max_level=8):
    """Create 3D mesh."""
    return Mesh.box_3d([x_min, y_min, z_min],
                      [x_max, y_max, z_max],
                      mesh_config(dim=3, min_level=min_level,
                                max_level=max_level))
```

### 5.2 Context Managers

```python
# samurai/context.py

from contextlib import contextmanager

@contextmanager
def timer(name):
    """Context manager for timing operations."""
    import time
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")

@contextmanager
def mesh_checkpoint(mesh, path):
    """Save/restore mesh state."""
    import pickle
    import os

    # Save initial state
    if os.path.exists(path):
        with open(path, 'rb') as f:
            initial_state = pickle.load(f)
    else:
        initial_state = None

    try:
        yield mesh
    finally:
        # Restore if needed
        if initial_state is not None:
            mesh.restore_state(initial_state)

@contextmanager
def mpi_session(communicator=None):
    """MPI context for parallel simulations."""
    try:
        from mpi4py import MPI
        comm = communicator or MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"MPI session: rank {rank}/{size}")
        yield comm
    except ImportError:
        print("MPI not available, running serial")
        yield None

# Usage
with timer("Simulation"):
    with mesh_checkpoint(mesh, "checkpoint.pkl"):
        with mpi_session() as comm:
            # Parallel simulation with checkpointing
            run_simulation(mesh, comm)
```

### 5.3 Example Workflows

```python
# examples/pythonic_api.py

import samurai as sam
import numpy as np

# Example 1: Advection equation with AMR
# =======================================

# Setup
mesh = sam.mesh_2d(0, 0, 1, 1, min_level=2, max_level=8)
u = sam.ScalarField("density", mesh)
velocity = [1.0, 0.5]

# Initial condition
def init_pulse(cell):
    x, y = cell.center
    r2 = (x - 0.2)**2 + (y - 0.5)**2
    return np.exp(-r2 / 0.01)

sam.for_each_cell(mesh, lambda c: u.assign(c, init_pulse(c)))

# Boundary conditions
u.attach_bc(sam.FunctionBC(lambda x: 0, level=mesh.max_level))

# Time loop with automatic adaptation
t_end = 1.0
dt = 0.001

for t in sam.TimeLoop(t_end=t_end, dt=dt):
    # Update ghost cells
    u.update_ghosts()

    # Numerical scheme
    u[:] = u - dt * sam.upwind(velocity=velocity, field=u)

    # Adapt mesh based on gradient
    def criterion(cell):
        grad_mag = np.sqrt(u.grad_x[cell]**2 + u.grad_y[cell]**2)
        return grad_mag

    mesh.adapt(criterion, epsilon=1e-3)

    # Visualization
    if t.step % 10 == 0:
        sam.plot_field(u, title=f"t = {t.time:.3f}")

# Example 2: Diffusion with multigrid
# ====================================

mesh = sam.mesh_2d(0, 0, 1, 1, min_level=1, max_level=6)
u = sam.ScalarField("temperature", mesh)
u.attach_bc(sam.Dirichlet(value=0, level=mesh.max_level))

# Initial: hot spot in center
sam.for_each_cell(mesh, lambda c: u.assign(c, 1.0 if np.linalg.norm(c.center - 0.5) < 0.1 else 0.0))

# Diffusion solver with multigrid preconditioner
solver = sam.HeatEquation(
    mesh=mesh,
    diffusivity=0.1,
    time_integrator='RK4',
    multigrid_levels=4
)

# Time integration
for t in sam.TimeLoop(t_end=0.1, dt=0.001):
    u = solver.step(u, dt)
    mesh.adapt(lambda c: np.abs(u.laplacian[c]), epsilon=1e-4)
```

### 5.4 Type Hints

```python
# samurai/types.py

from typing import TypeVar, Generic, Callable, Optional
import numpy as np
import jax.numpy as jnp

Dim = TypeVar('Dim', bound=int)
ValueType = TypeVar('ValueType', float, int, complex)

class ScalarField(Generic[Dim, ValueType]):
    """Typed scalar field."""

    def __init__(self, name: str, mesh: 'Mesh[Dim]'):
        self.name = name
        self.mesh = mesh

    def __getitem__(self, cell: 'Cell') -> ValueType:
        ...

    def __setitem__(self, cell: 'Cell', value: ValueType) -> None:
        ...

class VectorField(Generic[Dim, ValueType]):
    """Typed vector field."""
    ...

# Function signatures with type hints
AdaptationCriterion = Callable[['Cell'], float]
TimeIntegrator = Callable[[ScalarField, float], ScalarField]

def adapt(mesh: 'Mesh',
          criterion: AdaptationCriterion,
          epsilon: float) -> None:
    ...

def update_ghosts(field: ScalarField) -> None:
    ...
```

---

## 6. Scientific ML Integration

### 6.1 Physics-Informed Neural Networks (PINNs)

```python
# samurai/pinn.py

import jax
import jax.numpy as jnp
import flax.linen as nn

class PINNSolver:
    """
    Physics-Informed Neural Network for PDEs.

    Combines:
    - Neural network as function approximator
    - PDE residual as loss function
    - Samurai mesh for collocation points
    """

    def __init__(self, mesh, pde_residual, boundary_conditions):
        self.mesh = mesh
        self.pde_residual = pde_residual
        self.boundary_conditions = boundary_conditions

        # Neural network
        self.net = nn.Sequential([
            nn.Dense(64),
            nn.tanh,
            nn.Dense(64),
            nn.tanh,
            nn.Dense(64),
            nn.tanh,
            nn.Dense(1)  # Output: field value
        ])

    @jax.jit
    def predict(self, params, x, t):
        """Network prediction."""
        return self.net.apply(params, jnp.stack([x, t], axis=-1))

    @jax.jit
    def loss(self, params, collocation_points):
        """Physics-informed loss."""
        # Interior loss: PDE residual
        u_pred = self.predict(params, collocation_points[:, 0],
                             collocation_points[:, 1])

        # Compute derivatives using autodiff
        u_t = jax.grad(lambda t: self.predict(params, collocation_points[0, 0], t))
        u_x = jax.grad(lambda x: self.predict(params, x, collocation_points[0, 1]))
        u_xx = jax.grad(lambda x: u_x(x))

        residual = u_t - self.pde_residual(u_x, u_xx)
        interior_loss = jnp.mean(residual**2)

        # Boundary loss
        bc_loss = 0.0
        for bc in self.boundary_conditions:
            bc_pred = self.predict(params, bc.x, bc.t)
            bc_loss += jnp.mean((bc_pred - bc.value)**2)

        return interior_loss + bc_loss

    def train(self, n_epochs=1000):
        """Train PINN."""
        # Generate collocation points from mesh
        collocation_points = self._sample_collocation_points()

        # Initialize network
        params = self.net.init(jax.random.PRNGKey(0),
                              jnp.zeros((2,)))

        # Optimizer
        optimizer = nn.Adam(learning_rate=1e-3)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, points):
            loss_val, grads = jax.value_and_grad(self.loss)(params, points)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        # Training loop
        for epoch in range(n_epochs):
            params, opt_state, loss_val = step(
                params, opt_state, collocation_points
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss_val:.6e}")

        return params

    def _sample_collocation_points(self):
        """Sample collocation points from Samurai mesh."""
        points = []
        for cell in self.mesh:
            # Use cell centers as collocation points
            x, y = cell.center
            points.append([x, 0])  # Assuming time-independent for now
        return jnp.array(points)
```

### 6.2 Hybrid Physics-ML

```python
# samurai/hybrid.py

class HybridSolver:
    """
    Combine traditional numerical schemes with ML corrections.

    Strategy:
    - Base solver: Finite volume/difference scheme
    - ML correction: Learns error in scheme
    - Adaptation: ML guides refinement
    """

    def __init__(self, mesh, base_scheme):
        self.mesh = mesh
        self.base_scheme = base_scheme

        # Error correction network
        self.correction_net = nn.Sequential([
            nn.Dense(32),
            nn.relu,
            nn.Dense(32),
            nn.relu,
            nn.Dense(1)  # Error correction
        ])

    def step(self, u, dt, params):
        """Hybrid time step."""
        # Base scheme
        u_base = self.base_scheme(u, dt)

        # ML correction
        correction = self._compute_correction(u, params)
        u_corrected = u_base + dt * correction

        return u_corrected

    def _compute_correction(self, u, params):
        """Compute ML correction field."""
        corrections = []

        for cell in self.mesh:
            # Local features
            features = self._extract_features(u, cell)

            # Predict correction
            correction = self.correction_net.apply(params, features)
            corrections.append(correction)

        return jnp.array(corrections)

    def _extract_features(self, u, cell):
        """Extract local features for ML model."""
        # Field value
        u_val = u[cell]

        # Gradients
        grad_x = u.grad_x[cell]
        grad_y = u.grad_y[cell]

        # Level information
        level = cell.level

        # Position
        x, y = cell.center

        return jnp.array([u_val, grad_x, grad_y, x, y, level])

    def train_correction(self, high_res_solution):
        """
        Train correction network using high-resolution reference.

        Parameters
        ----------
        high_res_solution : ScalarField
            High-fidelity solution (e.g., from fine uniform mesh)
        """
        # Generate training data
        training_data = []

        for cell in self.mesh:
            # Base scheme prediction
            u_base = self.base_scheme.predict(cell)

            # High-res reference
            u_ref = high_res_solution[cell]

            # Error to learn
            error = u_ref - u_base

            features = self._extract_features(self.mesh, cell)
            training_data.append((features, error))

        # Train network
        # ... standard training loop ...
```

### 6.3 Neural Operators for Surrogate Modeling

```python
# samurai/neural_operator.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import equinox as eqx

class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator (FNO) for learning solution operators.

    Application: Learn mapping from initial conditions to final states,
    bypassing expensive time integration.
    """

    modes: tuple  # Fourier modes to keep
    width: int    # Hidden layer width

    @nn.compact
    def __call__(self, x):
        """
        Parameters
        ----------
        x : (batch, mesh_size, input_dim)
            Input field (e.g., initial condition)

        Returns
        -------
        y : (batch, mesh_size, output_dim)
            Output field (e.g., solution at t=T)
        """
        # Lift to higher dimension
        x = nn.Dense(self.width)(x)

        # Fourier layers
        for _ in range(4):
            # Fourier transform
            x_ft = jnp.fft.fft(x, axis=1)

            # Truncate modes
            x_ft = x_ft[:, :self.modes[0], :]

            # Spectral convolution
            x_ft = nn.Dense(self.width)(x_ft)

            # Inverse FFT
            x = jnp.fft.ifft(x_ft, axis=1).real

            # Nonlinearity
            x = nn.gelu(x)

        # Project to output
        y = nn.Dense(1)(x)
        return y.squeeze(-1)

class NeuralOperatorSolver:
    """
    Learn solution operators for AMR simulations.
    """

    def __init__(self, mesh, config):
        self.mesh = mesh
        self.config = config

        # Note: For AMR, need to handle variable mesh structure
        # Strategy: parameterize by mesh level and cell positions

        self.model = FourierNeuralOperator(
            modes=(16, 16),
            width=64
        )

    def train(self, dataset):
        """
        Train neural operator.

        Parameters
        ----------
        dataset : list of tuples
            Each tuple: (initial_condition, final_state, mesh_structure)
        """
        def loss(params, batch):
            u0, u_target, mesh_info = batch

            # Predict final state
            u_pred = self.model.apply(params, u0)

            # L2 loss
            return jnp.mean((u_pred - u_target)**2)

        # Training loop
        params = self.model.init(jax.random.PRNGKey(0), dataset[0][0])
        optimizer = nn.Adam(learning_rate=1e-3)

        for epoch in range(1000):
            for batch in dataset:
                loss_val, grads = jax.value_and_grad(loss)(params, batch)
                params = optax.apply_updates(params, optimizer.update(grads)[0])

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss_val:.6e}")

        return params

    def predict(self, params, u0, mesh_structure):
        """
        Predict solution given initial condition.

        Note: For AMR, need to handle mesh adaptation.
        """
        return self.model.apply(params, u0)
```

---

## 7. Implementation Plan

### 7.1 Phases

**Phase 1: Core Bindings (Months 1-3)**
- Mesh, Field, CellArray bindings
- Basic algorithms (for_each_cell, for_each_interval)
- NumPy zero-copy integration
- Error handling
- Target: 80% of core functionality

**Phase 2: Schemes & Operators (Months 4-5)**
- Finite volume schemes
- Boundary conditions
- Operators (gradient, divergence, laplacian)
- Time integrators

**Phase 3: JAX Integration (Months 6-7)**
- JAX primitive registration
- Differentiable schemes
- VJP rules for key operations

**Phase 4: High-Level API (Months 8-9)**
- Pythonic API design
- Context managers
- Convenience functions

**Phase 5: Scientific ML (Months 10-12)**
- PINN implementation
- Neural operators
- Hybrid solvers

### 7.2 Testing Strategy

```python
# tests/python/test_core.py

import pytest
import numpy as np
import samurai as sam

class TestMesh:
    """Test mesh operations."""

    def test_mesh_creation_1d(self):
        """Test 1D mesh creation."""
        mesh = sam.mesh_1d(0, 1, min_level=2, max_level=5)
        assert mesh.dim == 1
        assert mesh.min_level == 2
        assert mesh.max_level == 5
        assert mesh.nb_cells > 0

    def test_mesh_creation_2d(self):
        """Test 2D mesh creation."""
        mesh = sam.mesh_2d(0, 0, 1, 1, min_level=2, max_level=5)
        assert mesh.dim == 2

    def test_mesh_adaptation(self):
        """Test mesh adaptation."""
        mesh = sam.mesh_2d(0, 0, 1, 1)
        initial_cells = mesh.nb_cells

        def criterion(cell):
            return np.linalg.norm(cell.center - 0.5)

        sam.adapt(mesh, criterion, epsilon=0.1)
        assert mesh.nb_cells != initial_cells

class TestField:
    """Test field operations."""

    def test_field_creation(self):
        """Test field creation."""
        mesh = sam.mesh_2d(0, 0, 1, 1)
        u = sam.ScalarField("test", mesh)
        assert u.name == "test"
        assert u.array.shape[0] == mesh.nb_cells

    def test_field_fill(self):
        """Test field filling."""
        mesh = sam.mesh_2d(0, 0, 1, 1)
        u = sam.ScalarField("test", mesh)
        u.fill(1.0)
        assert np.allclose(u.array[:], 1.0)

    def test_field_numpy_integration(self):
        """Test zero-copy NumPy integration."""
        mesh = sam.mesh_2d(0, 0, 1, 1)
        u = sam.ScalarField("test", mesh)

        # Get NumPy view
        arr = u.array
        assert isinstance(arr, np.ndarray)

        # Modify through NumPy
        arr[:] = 5.0
        assert np.allclose(u.array[:], 5.0)

class TestAlgorithms:
    """Test algorithm bindings."""

    def test_for_each_cell(self):
        """Test cell iteration."""
        mesh = sam.mesh_2d(0, 0, 1, 1)
        u = sam.ScalarField("test", mesh)

        count = 0
        def count_cells(cell):
            nonlocal count
            count += 1
            u[cell] = cell.level

        sam.for_each_cell(mesh, count_cells)
        assert count == mesh.nb_cells
        assert np.all(u.array[:] > 0)

    def test_boundary_conditions(self):
        """Test boundary conditions."""
        mesh = sam.mesh_2d(0, 0, 1, 1)
        u = sam.ScalarField("test", mesh)

        # Dirichlet BC
        bc = sam.Dirichlet(value=0.0, level=mesh.max_level)
        u.attach_bc(bc)

        # Neumann BC
        bc = sam.Neumann(derivative=1.0, level=mesh.max_level)
        u.attach_bc(bc)

class TestJAX:
    """Test JAX integration."""

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
    def test_jax_compatibility(self):
        """Test JAX array compatibility."""
        import jax.numpy as jnp

        mesh = sam.mesh_2d(0, 0, 1, 1)
        u = sam.ScalarField("test", mesh)

        # Convert to JAX
        u_jax = jnp.array(u.array)
        assert u_jax.shape == u.array.shape

        # Operations
        result = jnp.sin(u_jax)
        assert result.shape == u_jax.shape

class BenchmarkPerformance:
    """Performance benchmarks."""

    def test_mesh_adaptation_performance(self):
        """Benchmark mesh adaptation."""
        import time

        mesh = sam.mesh_2d(0, 0, 1, 1, min_level=1, max_level=6)

        start = time.perf_counter()
        sam.adapt(mesh, lambda c: 1, epsilon=0.1)
        elapsed = time.perf_counter() - start

        # Should be < 1 second for this mesh
        assert elapsed < 1.0

    def test_field_access_performance(self):
        """Benchmark field access."""
        mesh = sam.mesh_2d(0, 0, 1, 1, min_level=3, max_level=5)
        u = sam.ScalarField("test", mesh)

        # Direct NumPy access
        start = time.perf_counter()
        for _ in range(100):
            arr = u.array
            arr[:] = arr * 2
        elapsed = time.perf_counter() - start

        # Should be fast
        assert elapsed < 0.1
```

### 7.3 Documentation

```rst
# docs/python_api.rst

Python API Reference
====================

Core Types
----------

.. autoclass:: samurai.Mesh
    :members:

.. autoclass:: samurai.ScalarField
    :members:

.. autoclass:: samurai.VectorField
    :members:

Algorithms
----------

.. autofunction:: samurai.for_each_cell

.. autofunction:: samurai.for_each_interval

.. autofunction:: samurai.adapt

Operators
---------

.. autofunction:: samurai.upwind

.. autofunction:: samurai.laplacian

.. autofunction:: samurai.grad

.. autofunction:: samurai.div

Boundary Conditions
-------------------

.. autoclass:: samurai.Dirichlet
    :members:

.. autoclass:: samurai.Neumann
    :members:

.. autoclass:: samurai.FunctionBC
    :members:
```

---

## 8. Performance Considerations

### 8.1 Zero-Copy Strategy

```cpp
// Ensure zero-copy between C++ and Python

// CORRECT: Zero-copy view
py::array_t<double> get_array(Field& field) {
    auto& data = field.array();
    return py::array_t<double>(
        data.shape(),      // Shape
        data.strides(),    // Strides
        data.data(),       // Pointer to existing data
        py::cast(field)    // Keep field alive
    );
}

// INCORRECT: Creates copy
py::array_t<double> get_array_copy(Field& field) {
    auto& data = field.array();
    std::vector<double> copy(data.begin(), data.end());
    return py::array_t<double>(copy.size(), copy.data());
}
```

### 8.2 GIL Release

```cpp
// Release GIL for long-running operations

void adapt_mesh(Mesh& mesh, Criterion& criterion, double epsilon) {
    // Release GIL
    py::gil_scoped_release release;

    // Expensive computation without GIL
    samurai::adapt(mesh, criterion, epsilon);

    // Reacquire GIL
    py::gil_scoped_acquire acquire;
}
```

### 8.3 Memory Management

```cpp
// Proper lifetime management

// Strategy: Keep C++ object alive while Python object exists
py::class_<Mesh>("Mesh", py::dynamic_attr())
    .def(py::init(...))
    // Keep internal C++ object alive
    .def("get_cells", [](Mesh& self) {
        // Returns view, not copy
        return py::array(..., self.cell_data(), py::cast(self));
    });
```

### 8.4 Compilation Cache

```python
# samurai/compile_cache.py

import functools
import hashlib
import pickle
import os

_CACHE_DIR = os.path.expanduser("~/.samurai/cache")

def cached_compile(func):
    """Cache compiled C++ extensions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from function + arguments
        key = hashlib.md5(
            pickle.dumps((func.__name__, args, kwargs))
        ).hexdigest()
        cache_file = os.path.join(_CACHE_DIR, f"{key}.so")

        if os.path.exists(cache_file):
            # Load from cache
            return load_compiled_extension(cache_file)
        else:
            # Compile and cache
            os.makedirs(_CACHE_DIR, exist_ok=True)
            result = func(*args, **kwargs)
            save_compiled_extension(cache_file, result)
            return result

    return wrapper
```

---

## 9. Security and Safety

### 9.1 Input Validation

```cpp
// Validate Python inputs before passing to C++

void validate_mesh_config(const py::dict& config) {
    if (!config.contains("min_level")) {
        throw std::invalid_argument("Missing 'min_level' in mesh config");
    }

    int min_level = config["min_level"].cast<int>();
    if (min_level < 0 || min_level > 20) {
        throw std::invalid_argument("min_level must be in [0, 20]");
    }

    // ... more validation ...
}
```

### 9.2 Resource Limits

```python
# samurai/resource_limits.py

import resource

def set_resource_limits():
    """Prevent runaway simulations."""
    # Max memory: 16GB
    resource.setrlimit(
        resource.RLIMIT_AS,
        (16 * 1024**3, 16 * 1024**3)
    )

    # Max CPU time: 1 hour
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (3600, 3600)
    )

# Call on import
set_resource_limits()
```

---

## 10. Conclusion

### 10.1 Key Deliverables

1. **Complete Python bindings** for 80% of Samurai C++ API
2. **Zero-copy NumPy integration** for efficient data exchange
3. **JAX autodiff support** for differentiable physics
4. **Scientific ML integration** (PINNs, neural operators)
5. **Jupyter notebooks** for education and research
6. **Comprehensive documentation** and tutorials

### 10.2 Impact Metrics

- **User base expansion:** 100 C++ developers → 15M+ Python users
- **Research visibility:** Enable reproducible notebooks
- **Educational reach:** Lower barrier for students
- **ML integration:** State-of-the-art scientific ML workflows

### 10.3 Future Directions

1. **GPU acceleration** through JAX/CuPy
2. **Distributed computing** with Dask/dask-mpi
3. **Real-time visualization** in Jupyter
4. **Cloud deployment** (Google Colab, Binder)
5. **Community contributions** through Python-first development

---

**Document Version:** 1.0
**Last Updated:** 2025-01-05
**Author:** Samurai V2 Development Team
**Status:** Ready for Implementation
