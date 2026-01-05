# Python Bindings for Samurai V2: Technical Report

**Author**: Samurai V2 Development Team
**Date**: 2025-01-05
**Version**: 1.0
**Framework**: pybind11

---

## Executive Summary

This document provides a comprehensive technical design for Python bindings to the Samurai V2 AMR library using pybind11. The header-only nature of Samurai makes it ideal for Python integration without binary compatibility concerns. The proposed architecture exposes mesh, fields, operators, and algorithms to the Scientific Python ecosystem (NumPy, SciPy, Matplotlib, Jupyter).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Structure](#2-module-structure)
3. [Core Class Wrappers](#3-core-class-wrappers)
4. [NumPy Array Interoperability](#4-numpy-array-interoperability)
5. [Type Conversions](#5-type-conversions)
6. [Error Handling](#6-error-handling)
7. [HDF5 I/O from Python](#7-hdf5-io-from-python)
8. [Implementation Details](#8-implementation-details)
9. [Build System Integration](#9-build-system-integration)
10. [Testing Strategy](#10-testing-strategy)
11. [Documentation](#11-documentation)
12. [Performance Considerations](#12-performance-considerations)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

- **Header-Only Advantage**: Samurai's header-only design eliminates ABI compatibility issues
- **Zero-Copy Access**: Direct memory access between C++ and Python via NumPy buffer protocol
- **Pythonic API**: Natural Python interface while preserving C++ performance
- **Selective Exposure**: Only expose necessary functionality, not entire C++ API

### 1.2 pybind11 as Framework Choice

**Rationale**:
- Lightweight, header-only library
- Excellent NumPy/xtensor interoperability
- Automatic type conversions for STL containers
- Built-in support for C++ exceptions translation
- Modern C++17/20 support
- Wide adoption in scientific computing community

### 1.3 High-Level Architecture

```
samurai/                    # Python package
├── __init__.py            # Package initialization
├── core/                   # Core mesh and field classes
│   ├── __init__.py
│   ├── mesh.pyi           # Type stubs
│   └── field.pyi
├── schemes/               # Numerical schemes
│   ├── __init__.py
│   └── finite_volume.pyi
├── operators/             # Differential operators
│   ├── __init__.py
│   └── fluxes.pyi
├── io/                    # HDF5 I/O
│   ├── __init__.py
│   └── hdf5.pyi
└── algorithms/            # Mesh adaptation, prediction
    ├── __init__.py
    └── amr.pyi
```

---

## 2. Module Structure

### 2.1 Main Module Definition

```cpp
// src/python_bindings/samurai_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for submodules
void init_core(py::module_& m);
void init_schemes(py::module_& m);
void init_operators(py::module_& m);
void init_io(py::module_& m);
void init_algorithms(py::module_& m);

PYBIND11_MODULE(samurai, m) {
    // Module docstring
    m.doc() = "Samurai V2: Adaptive Mesh Refinement library";

    // Version information
    m.attr("__version__") = SAMURAI_VERSION;

    // Define submodules
    auto core = m.def_submodule("core", "Core mesh and field classes");
    init_core(core);

    auto schemes = m.def_submodule("schemes", "Numerical schemes");
    init_schemes(schemes);

    auto operators = m.def_submodule("operators", "Differential operators");
    init_operators(operators);

    auto io = m.def_submodule("io", "HDF5 I/O operations");
    init_io(io);

    auto algorithms = m.def_submodule("algorithms", "AMR algorithms");
    init_algorithms(algorithms);

    // Enums
    py::enum_<samurai::Run>(m, "RunMode")
        .value("Sequential", samurai::Run::Sequential)
        .value("Parallel", samurai::Run::Parallel);

    py::enum_<samurai::BCVType>(m, "BoundaryConditionValueType")
        .value("Constant", samurai::BCVType::constant)
        .value("Function", samurai::BCVType::function);
}
```

### 2.2 Core Submodule

```cpp
// src/python_bindings/core_module.cpp

void init_core(py::module_& m) {
    // Mesh configuration
    py::class_<samurai::mesh_config<2>>(m, "MeshConfig2D")
        .def(py::init<>())
        .def("min_level", &samurai::mesh_config<2>::min_level,
             py::return_value_policy::reference)
        .def("max_level", &samurai::mesh_config<2>::max_level,
             py::return_value_policy::reference)
        .def("graduation_width", &samurai::mesh_config<2>::graduation_width,
             py::return_value_policy::reference)
        .def("start_level", &samurai::mesh_config<2>::start_level,
             py::return_value_policy::reference)
        .def("scaling_factor", &samurai::mesh_config<2>::scaling_factor,
             py::return_value_policy::reference)
        .def("periodic", py::overload_cast<bool(&)>(
                  &samurai::mesh_config<2>::periodic),
             py::return_value_policy::reference);

    // Box class
    py::class_<samurai::Box<double, 2>>(m, "Box2D")
        .def(py::init<const std::array<double, 2>&,
                      const std::array<double, 2>&>(),
             py::arg("min_corner"), py::arg("max_corner"))
        .def_property_readonly("min", &samurai::Box<double, 2>::min)
        .def_property_readonly("max", &samurai::Box<double, 2>::max)
        .def_property_readonly("length", &samurai::Box<double, 2>::length);

    // Mesh wrapper (template instantiation)
    bind_mesh<2>(m);
    bind_scalar_field<2>(m);
    bind_vector_field<2>(m);
}
```

---

## 3. Core Class Wrappers

### 3.1 Mesh Class Wrapper

```cpp
// src/python_bindings/mesh_bindings.hpp

template <std::size_t dim>
void bind_mesh(py::module_& m) {
    using Mesh = samurai::AMRMesh<dim>;
    using Config = typename Mesh::config;
    using interval_t = typename Mesh::interval_t;
    using value_t = typename interval_t::value_t;
    using index_t = typename interval_t::index_t;

    std::string mesh_name = "Mesh" + std::to_string(dim) + "D";

    py::class_<Mesh>(m, mesh_name.c_str(), py::buffer_protocol())
        // Constructors
        .def(py::init<const samurai::Box<double, dim>&,
                      const samurai::mesh_config<dim, int, 8, interval_t>&>(),
             py::arg("box"), py::arg("config"),
             "Create a mesh from a bounding box")

        // Properties
        .def_property_readonly("dim", [](const Mesh&) { return dim; })
        .def_property_readonly("min_level", &Mesh::min_level)
        .def_property_readonly("max_level", &Mesh::max_level)
        .def_property_readonly("graduation_width", &Mesh::graduation_width)
        .def_property_readonly("ghost_width", &Mesh::ghost_width)
        .def_property("origin_point",
             py::overload_cast<>(&Mesh::origin_point, py::const_),
             py::return_value_policy::reference)
        .def_property("scaling_factor",
             py::overload_cast<>(&Mesh::scaling_factor, py::const_),
             py::return_value_policy::reference)

        // Cell access
        .def("nb_cells", py::overload_cast<>(&Mesh::nb_cells, py::const_),
             "Total number of cells")
        .def("nb_cells", py::overload_cast<std::size_t>(
                  &Mesh::nb_cells, py::const_),
             py::arg("level"), "Number of cells at given level")

        // Cell iteration
        .def("for_each_cell", [](const Mesh& mesh, py::function func) {
            samurai::for_each_cell(mesh, [&](const auto& cell) {
                func(cell);
            });
        }, py::arg("func"), "Iterate over all cells with Python callable")

        .def("for_each_level", [](const Mesh& mesh, py::function func) {
            samurai::for_each_level(mesh, [&](std::size_t level) {
                func(level);
            });
        }, py::arg("func"), "Iterate over all refinement levels")

        // Mesh manipulation
        .def("update_mesh_neighbour", &Mesh::update_mesh_neighbour,
             "Update mesh neighbor information")

        // String representation
        .def("__repr__", [mesh_name](const Mesh& mesh) {
            std::ostringstream oss;
            oss << mesh_name << "(dim=" << dim
                << ", min_level=" << mesh.min_level()
                << ", max_level=" << mesh.max_level()
                << ", nb_cells=" << mesh.nb_cells() << ")";
            return oss.str();
        })

        // Pickle support
        .def(py::pickle(
            [](const Mesh& mesh) { // __getstate__
                return py::make_tuple(
                    mesh.min_level(), mesh.max_level(),
                    mesh.origin_point(), mesh.scaling_factor()
                );
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                samurai::Box<double, dim> box(
                    /* reconstruct from state */
                );
                samurai::mesh_config<dim, int, 8, interval_t> cfg;
                cfg.min_level(t[0].cast<std::size_t>());
                cfg.max_level(t[1].cast<std::size_t>());

                return Mesh(box, cfg);
            }
        ));
}
```

### 3.2 Field Class Wrapper with NumPy Integration

```cpp
// src/python_bindings/field_bindings.hpp

template <std::size_t dim>
void bind_scalar_field(py::module_& m) {
    using Mesh = samurai::AMRMesh<dim>;
    using Field = samurai::ScalarField<Mesh, double>;

    py::class_<Field>(m, "ScalarField", py::buffer_protocol())
        .def(py::init<std::string, Mesh&>(),
             py::arg("name"), py::arg("mesh"),
             "Create a scalar field")

        // Properties
        .def_property_readonly("name", &Field::name,
                              py::return_value_policy::reference)
        .def_property_readonly("mesh", &Field::mesh,
                              py::return_value_policy::reference)
        .def_property_readonly("size", &Field::size)

        // NumPy buffer protocol - ZERO-COPY ACCESS
        .def_buffer([](Field& f) -> py::buffer_info {
            using T = typename Field::value_type;
            return py::buffer_info(
                f.array().data(),                    // Pointer to data
                sizeof(T),                           // Size of one scalar
                py::format_descriptor<T>::format(),  // Python struct-style format
                1,                                   // Number of dimensions
                {f.array().size()},                 // Buffer dimensions
                {sizeof(T)}                          // Strides
            );
        })

        // Direct NumPy array view (zero-copy)
        .def("numpy_view", [](Field& f) {
            using T = typename Field::value_type;
            return py::array_t<T>(
                {f.array().size()},              // Shape
                {sizeof(T)},                    // Strides
                f.array().data(),               // Data pointer
                py::cast(f)                     // Keep alive
            );
        }, py::return_value_policy::take_ownership,
           "Returns a zero-copy NumPy view of the field data")

        // Element access
        .def("__getitem__", [](Field& f, const samurai::Cell<dim,
                                      typename Field::interval_t>& cell)
                              -> typename Field::value_type& {
            return f[cell];
        }, py::return_value_policy::reference)

        .def("__setitem__", [](Field& f,
                                const samurai::Cell<dim, typename Field::interval_t>& cell,
                                typename Field::value_type value) {
            f[cell] = value;
        })

        // Fill operations
        .def("fill", &Field::fill, py::arg("value"),
             "Fill field with constant value")

        // Boundary conditions
        .def("attach_bc", [](Field& f, py::function bc_func,
                             py::object direction) {
            // Python boundary condition wrapper
            auto bc_wrapper = [bc_func](const auto& dir,
                                        const auto& cell,
                                        const auto& coords) {
                return bc_func(dir, cell, coords).template
                       cast<typename Field::value_type>();
            };
            return f.attach_bc(
                samurai::FunctionBc<Field>(bc_wrapper)
            );
        }, py::arg("function"), py::arg("direction"),
           "Attach a Python function as boundary condition")

        // String representation
        .def("__repr__", [](const Field& f) {
            return "ScalarField(name='" + f.name() + "', size="
                   + std::to_string(f.array().size()) + ")";
        })

        // Arithmetic operators
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self *= double())
        .def(py::self * double());
}

template <std::size_t dim>
void bind_vector_field(py::module_& m) {
    using Mesh = samurai::AMRMesh<dim>;
    using Field = samurai::VectorField<Mesh, double, 2, false>;

    py::class_<Field>(m, "VectorField", py::buffer_protocol())
        .def(py::init<std::string, Mesh&>(),
             py::arg("name"), py::arg("mesh"))

        .def_property_readonly("n_components", &Field::n_comp)
        .def_property_readonly("name", &Field::name)

        // NumPy buffer protocol for SOA/AOS layout
        .def_buffer([](Field& f) -> py::buffer_info {
            using T = typename Field::value_type;
            std::vector<std::size_t> shape = {f.array().size(), f.n_comp};
            std::vector<std::size_t> strides = {f.n_comp * sizeof(T), sizeof(T)};
            return py::buffer_info(
                f.array().data(),
                sizeof(T),
                py::format_descriptor<T>::format(),
                2,
                shape,
                strides
            );
        })

        // Component access
        .def("get_component", [](Field& f, std::size_t i) {
            py::array_t<typename Field::value_type> arr =
                py::module_::import("numpy").attr("empty")(f.array().size());
            auto buf = arr.request();
            auto* ptr = static_cast<typename Field::value_type*>(buf.ptr);

            for (std::size_t j = 0; j < f.array().size(); ++j) {
                ptr[j] = f.array()(j, i);
            }
            return arr;
        }, py::arg("component_index"));
}
```

---

## 4. NumPy Array Interoperability

### 4.1 Zero-Copy Memory Access

The key innovation is direct memory sharing between Samurai fields and NumPy arrays:

```cpp
// Zero-copy view implementation
template <class Field>
py::array_t<typename Field::value_type>
numpy_zero_copy_view(Field& field) {
    using value_t = typename Field::value_type;

    // Get underlying xtensor container
    auto& xtensor = field.array();

    // Create py::array_t with existing data pointer
    // Keep field alive via keep_alive argument
    return py::array_t<value_t>(
        xtensor.shape(),                    // Shape
        xtensor.strides(),                  // Strides in bytes
        xtensor.data(),                     // Data pointer
        py::cast(field)                     // Keep field alive
    );
}
```

### 4.2 Python Usage

```python
import samurai
import numpy as np

# Create mesh and field
config = samurai.MeshConfig2D()
config.min_level = 2
config.max_level = 6

box = samurai.Box2D([0., 0.], [1., 1.])
mesh = samurai.Mesh2D(box, config)

u = samurai.ScalarField("solution", mesh)

# Zero-copy view - NO DATA COPY
u_arr = u.numpy_view()
print(f"Array address: {u_arr.__array_interface__['data'][0]}")
print(f"Field shares memory: {np.shares_memory(u_arr, u.numpy_view())}")  # True

# Direct modification - updates field in-place
u_arr[:] = 1.0
print(f"Field value at first cell: {u[0]}")  # 1.0

# Vectorized operations
u_arr[:] = np.sin(2 * np.pi * x_coords) * np.cos(2 * np.pi * y_coords)
```

### 4.3 Array Interface Protocol

```cpp
// Implement Python's array interface
template <class Field>
py::dict array_interface(Field& field) {
    using value_t = typename Field::value_type;
    auto& xt = field.array();

    py::dict interface;

    // Shape
    py::list shape;
    for (auto s : xt.shape()) {
        shape.append(s);
    }
    interface["shape"] = shape;

    // Typestr
    std::string typestr = "<f" + std::to_string(sizeof(value_t));
    interface["typestr"] = typestr;

    // Data pointer (read-write)
    py::tuple data(2);
    data[0] = reinterpret_cast<std::intptr_t>(xt.data());
    data[1] = false;  // Not read-only
    interface["data"] = data;

    // Version
    interface["version"] = 3;

    return interface;
}
```

---

## 5. Type Conversions

### 5.1 STL Container Conversions

```cpp
// Automatic std::vector <-> list conversion
py::implicitly_convertible<
    py::list,
    std::vector<typename Mesh::interval_t::value_t>
>();

// std::array <-> tuple
py::implicitly_convertible<
    py::tuple,
    std::array<double, 2>
>();

// Map conversion
py::class_<std::map<std::string, double>>(m, "StringDoubleMap")
    .def(py::init<>())
    .def("to_dict", [](const std::map<std::string, double>& m) {
        py::dict d;
        for (const auto& kv : m) {
            d[kv.first.c_str()] = kv.second;
        }
        return d;
    });
```

### 5.2 C++ Function to Python Callable

```cpp
// Wrapper for std::function
template <std::size_t dim>
void bind_function_wrappers(py::module_& m) {
    using coords_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

    // Function wrapper for initial conditions
    m.def("make_scalar_field", [](Mesh& mesh, py::function py_func) {
        // Convert Python callable to C++ std::function
        std::function<double(const coords_t&)> cpp_func =
            [py_func](const coords_t& coords) -> double {
                return py_func(coords).cast<double>();
            };

        return samurai::make_scalar_field<double>(
            "field", mesh, cpp_func
        );
    }, py::arg("mesh"), py::arg("function"),
       "Create field from Python callable");
}
```

### 5.3 Type Conversion Table

| C++ Type | Python Type | Conversion Method |
|-----------|-------------|-------------------|
| `double` | `float` | Automatic |
| `int` | `int` | Automatic |
| `std::string` | `str` | Automatic |
| `std::vector<T>` | `list` | Automatic |
| `std::array<T, N>` | `tuple` | Automatic |
| `std::map<K, V>` | `dict` | Custom wrapper |
| `xtensor<T>` | `numpy.ndarray` | Buffer protocol |
| `std::function` | `callable` | Custom wrapper |
| `samurai::Cell<dim>` | `samurai.Cell` | Class wrapper |
| `samurai::Interval` | `samurai.Interval` | Class wrapper |

### 5.4 Custom Type Casters

```cpp
// Custom type caster for xt::xtensor_fixed
namespace pybind11 { namespace detail {

template <class T, std::size_t N>
struct type_caster<xt::xtensor_fixed<T, xt::xshape<N>>> {
public:
    PYBIND11_TYPE_CASTER(xt::xtensor_fixed<T, xt::xshape<N>>,
                         const_name("xtensor_fixed"));

    bool load(handle src, bool) {
        py::array_t<T> buf = py::array_t<T>::ensure(src);
        if (!buf) return false;

        auto buf_info = buf.request();
        if (buf_info.ndim != 1) return false;
        if (buf_info.shape[0] != N) return false;

        value = xt::xtensor_fixed<T, xt::xshape<N>>::from_shape(
            xt::xshape<N>()
        );

        auto* ptr = static_cast<T*>(buf_info.ptr);
        std::copy(ptr, ptr + N, value.begin());

        return true;
    }

    static handle cast(
        const xt::xtensor_fixed<T, xt::xshape<N>>& src,
        return_value_policy, handle
    ) {
        py::array_t<T> array(N);
        auto buf = array.request();
        auto* ptr = static_cast<T*>(buf.ptr);
        std::copy(src.begin(), src.end(), ptr);
        return array.release();
    }
};

}}
```

---

## 6. Error Handling

### 6.1 Exception Translation

```cpp
// Custom exception translator
py::register_exception_translator([](std::exception_ptr p) {
    try {
        if (p) std::rethrow_exception(p);
    } catch (const samurai::AssertionError& e) {
        PyErr_SetString(PyExc_AssertionError, e.what());
    } catch (const samurai::MeshError& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const samurai::FieldError& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const std::out_of_range& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    }
});
```

### 6.2 Custom Exception Classes

```cpp
// Define Python exceptions
static py::exception<samurai::SamuraiError> samurai_error(m, "SamuraiError");
static py::exception<samurai::MeshError> mesh_error(m, "MeshError", samurai_error);
static py::exception<samurai::FieldError> field_error(m, "FieldError", samurai_error);

// Usage in Python
/*
try:
    mesh.adapt(epsilon=-1)
except samurai.SamuraiError as e:
    print(f"Error: {e}")
*/
```

### 6.3 Error Context Propagation

```cpp
// Wrapper with enhanced error messages
template <class Func>
auto wrap_with_context(Func&& func, const char* context) {
    return [func = std::forward<Func>(func), context](auto&&... args) {
        try {
            return func(std::forward<decltype(args)>(args)...);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string(context) + " failed: " + e.what()
            );
        }
    };
}

// Usage
.def("adapt", wrap_with_context(
    [](Mesh& mesh, double epsilon) { mesh.adapt(epsilon); },
    "Mesh adaptation"
), py::arg("epsilon"), "Adapt mesh based on error indicator")
```

### 6.4 Validation Layer

```cpp
// Input validation before calling C++
template <std::size_t dim>
void validated_adapt(Mesh& mesh, double epsilon) {
    if (epsilon < 0) {
        throw samurai::MeshError(
            "adapt(): epsilon must be positive, got " +
            std::to_string(epsilon)
        );
    }
    if (epsilon > 1) {
        PyErr_WarnEx(
            PyExc_RuntimeWarning,
            "Large epsilon value may cause excessive coarsening",
            1
        );
    }
    mesh.adapt(epsilon);
}
```

---

## 7. HDF5 I/O from Python

### 7.1 HDF5 Save/Load Wrapper

```cpp
// src/python_bindings/io_bindings.cpp

void init_io(py::module_& m) {
    using namespace samurai;

    // Save function
    m.def("save", [](const std::string& filename,
                     const py::args& fields,
                     bool by_level = false,
                     bool by_mesh_id = false) {
        if (fields.empty()) {
            throw std::runtime_error("save() requires at least one field");
        }

        // Extract mesh from first field
        auto& first_field = fields[0].cast<Field&>();
        auto& mesh = first_field.mesh();

        // Create options
        Hdf5Options<decltype(mesh)> options(by_level, by_mesh_id);

        // Call C++ save with variadic template expansion
        std::apply([&](const auto&... f) {
            save(fs::current_path(), filename, options, mesh, f...);
        }, extract_fields(fields));

    }, py::arg("filename"),
       py::arg("fields").noconvert(),  // Don't convert args
       py::kw_only(),  // Keyword-only arguments
       py::arg("by_level") = false,
       py::arg("by_mesh_id") = false,
       R"(
       Save mesh and fields to HDF5 file.

       Parameters
       ----------
       filename : str
           Output filename (without .h5 extension)
       *fields : Field
           Fields to save (variadic)
       by_level : bool, optional
           Group output by refinement level
       by_mesh_id : bool, optional
           Group output by mesh ID (cells, ghosts, etc.)

       Examples
       --------
       >>> samurai.save("solution", u, v, by_level=True)
       )");

    // Load function
    m.def("load", [](const std::string& filename) {
        auto file = HighFive::File(
            filename + ".h5",
            HighFive::File::ReadOnly
        );

        py::dict result;

        // Load metadata
        if (file.exist("mesh")) {
            auto mesh_group = file.getGroup("mesh");
            result["mesh"] = load_mesh(mesh_group);
        }

        // Load fields
        if (file.exist("fields")) {
            auto fields_group = file.getGroup("fields");
            for (const auto& name : fields_group.listObjectNames()) {
                result[name.c_str()] = load_field(fields_group, name);
            }
        }

        return result;
    }, py::arg("filename"));
}
```

### 7.2 h5py Integration

```cpp
// Expose HDF5 file handle to h5py
m.def("get_h5py_handle", [](const std::string& filename) {
    auto file = HighFive::File(
        filename + ".h5",
        HighFive::File::ReadOnly
    );

    // Get low-level HDF5 identifier
    hid_t fid = file.getId();

    // Import h5py
    auto h5py = py::module_::import("h5py");
    auto File = h5py.attr("File");

    // Create h5py.File from low-level id
    return File(py::int_(fid));
}, py::return_value_policy::take_ownership);
```

### 7.3 Field Extraction for HDF5

```cpp
// Helper to extract std::tuple of fields from py::args
template <typename... Fields>
auto extract_fields_as_tuple(const py::args& args) {
    return std::make_tuple(
        args[0].cast<Fields>()...
    );
}

// Usage with if constexpr for variadic fields
template <class Mesh, class... Fields>
void save_wrapper(const std::string& path,
                  const std::string& filename,
                  const Mesh& mesh,
                  const Fields&... fields) {
    save(path, filename, {}, mesh, fields...);
}
```

---

## 8. Implementation Details

### 8.1 Algorithm Wrappers

```cpp
// src/python_bindings/algorithm_bindings.cpp

void init_algorithms(py::module_& m) {
    // for_each_cell with Python callable
    m.def("for_each_cell", [](py::object mesh_obj, py::function func) {
        // Type dispatch for different mesh types
        if (py::isinstance<Mesh2D>(mesh_obj)) {
            auto mesh = mesh_obj.cast<Mesh2D&>();
            samurai::for_each_cell(mesh, [&](const auto& cell) {
                // Wrap cell in Python object
                func(cell);
            });
        }
    }, py::arg("mesh"), py::arg("function"));

    // Mesh adaptation
    m.def("adapt", [](py::object mesh_obj, double epsilon,
                      py::function criterion) {
        // Type erase the criterion
        auto cpp_criterion = [criterion](const auto& cell) -> double {
            return criterion(cell).cast<double>();
        };

        if (py::isinstance<Mesh2D>(mesh_obj)) {
            auto& mesh = mesh_obj.cast<Mesh2D&>();
            adapt(mesh, epsilon, cpp_criterion);
        }
    }, py::arg("mesh"), py::arg("epsilon"), py::arg("criterion"));

    // Make boundary condition
    m.def("make_bc", [](py::object field_obj, py::function bc_func,
                        py::object direction) {
        // Generic lambda for boundary condition
        auto bc_wrapper = [bc_func](const auto& dir,
                                    const auto& cell,
                                    const auto& coords) {
            return bc_func(dir, cell, coords).cast<double>();
        };

        return make_bc<Field>(bc_wrapper);
    }, py::arg("field"), py::arg("function"), py::arg("direction"));
}
```

### 8.2 Operator Wrappers

```cpp
// src/python_bindings/operator_bindings.cpp

void init_operators(py::module_& m) {
    using namespace samurai;

    // Gradient operator
    py::class_<GradientOperator<dim>>(m, "GradientOperator")
        .def(py::init<Field&>())
        .def("__call__", [](GradientOperator<dim>& op, const Field& f) {
            return op(f);
        });

    // Divergence operator
    py::class_<DivergenceOperator<dim>>(m, "DivergenceOperator")
        .def(py::init<VectorField<dim, double, dim>&>())
        .def("__call__", [](DivergenceOperator<dim>& op,
                           const VectorField<dim, double, dim>& f) {
            return op(f);
        });

    // Flux-based operators
    py::class_<FluxBasedOperator<dim>>(m, "FluxOperator")
        .def(py::init<Field&, py::function>())
        .def("set_flux", [](FluxBasedOperator<dim>& op,
                            py::function flux_func) {
            op.set_flux([flux_func](const auto&... args) {
                return flux_func(args...).cast<double>();
            });
        });
}
```

### 8.3 Cell Wrapper

```cpp
// Cell class wrapper
template <std::size_t dim>
void bind_cell(py::module_& m) {
    using interval_t = default_config::interval_t;
    using Cell = samurai::Cell<dim, interval_t>;

    py::class_<Cell>(m, ("Cell" + std::to_string(dim) + "D").c_str())
        .def_property_readonly("level", &Cell::level)
        .def_property_readonly("index", &Cell::index)
        .def_property_readonly("length", &Cell::length)
        .def_property_readonly("indices", &Cell::indices)
        .def_property_readonly("center", &Cell::center)
        .def_property_readonly("corner", &Cell::corner)

        .def("has_child", &Cell::has_child, "Check if cell has children")
        .def("is_boundary", &Cell::is_boundary, "Check if boundary cell")

        .def("__repr__", [](const Cell& c) {
            std::ostringstream oss;
            oss << "Cell(level=" << c.level
                << ", index=" << c.index << ")";
            return oss.str();
        })

        .def(py::self == py::self)
        .def(py::self != py::self);
}
```

---

## 9. Build System Integration

### 9.1 CMake Configuration

```cmake
# src/python_bindings/CMakeLists.txt

# Find Python and pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 2.10 REQUIRED)

# Create Python module
pybind11_add_module(samurai_python
    # Core bindings
    samurai_module.cpp
    core_module.cpp
    mesh_bindings.cpp
    field_bindings.cpp
    cell_bindings.cpp

    # Algorithm bindings
    algorithm_bindings.cpp

    # I/O bindings
    io_bindings.cpp

    # Operator bindings
    operator_bindings.cpp
)

# Link against Samurai
target_link_libraries(samurai_python
    PRIVATE
        samurai
        pybind11::module
)

# Include directories
target_include_directories(samurai_python
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${Python_INCLUDE_DIRS}
)

# C++ standard
target_compile_features(samurai_python PRIVATE cxx_std_20)

# Optimization
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(samurai_python PRIVATE -O3 -march=native)
endif()

# NumPy support
target_compile_definitions(samurai_python
    PRIVATE
        VERSION_INFO=${EXAMPLE_VERSION_INFO}
)

# Python type stubs
add_custom_command(TARGET samurai_python POST_BUILD
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/generate_stubs.py
        --output_dir ${CMAKE_BINARY_DIR}/python/stubs
    COMMENT "Generating Python type stubs"
)
```

### 9.2 Setup.py for pip Installation

```python
# setup.py

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Samurai extension module
ext_modules = [
    Pybind11Extension(
        "samurai",
        sources=[
            "src/python_bindings/samurai_module.cpp",
            "src/python_bindings/core_module.cpp",
            "src/python_bindings/mesh_bindings.cpp",
            # ... other sources
        ],
        extra_compile_args=["-O3", "-march=native"],
        include_dirs=[
            "include",
            pybind11.get_include(),
        ],
        cxx_std=20,
    ),
]

setup(
    name="samurai",
    version="0.28.0",
    author="Samurai Team",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.8",
)
```

### 9.3 Wheel Building

```bash
# Build command
python setup.py bdist_wheel

# Output: dist/samurai-0.28.0-cp310-cp310-linux_x86_64.whl

# Install
pip install dist/samurai-0.28.0-cp310-cp310-linux_x86_64.whl
```

---

## 10. Testing Strategy

### 10.1 pytest Test Suite

```python
# tests/python/test_mesh.py

import pytest
import samurai
import numpy as np

def test_mesh_construction():
    """Test basic mesh creation."""
    config = samurai.MeshConfig2D()
    config.min_level = 2
    config.max_level = 4

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)

    assert mesh.min_level == 2
    assert mesh.max_level == 4
    assert mesh.dim == 2

def test_field_creation():
    """Test field creation on mesh."""
    config = samurai.MeshConfig2D()
    config.min_level = 2
    config.max_level = 4

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)
    u = samurai.ScalarField("u", mesh)

    assert u.name == "u"
    assert u.mesh is mesh

def test_numpy_zero_copy():
    """Test zero-copy NumPy integration."""
    config = samurai.MeshConfig2D()
    config.min_level = 2
    config.max_level = 4

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)
    u = samurai.ScalarField("u", mesh)

    # Get zero-copy view
    u_arr = u.numpy_view()

    # Verify memory is shared
    u_arr[0] = 42.0
    assert u[0] == 42.0

def test_field_from_function():
    """Test creating field from Python callable."""
    config = samurai.MeshConfig2D()
    config.min_level = 2
    config.max_level = 4

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)

    def init_condition(x):
        return np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])

    u = samurai.make_scalar_field("u", mesh, init_condition)

    # Verify values
    for cell in mesh.for_each_cell():
        expected = init_condition(cell.center())
        assert abs(u[cell] - expected) < 1e-10

@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1])
def test_mesh_adaptation(epsilon):
    """Test mesh adaptation."""
    config = samurai.MeshConfig2D()
    config.min_level = 2
    config.max_level = 6

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)
    u = samurai.ScalarField("u", mesh)

    # Initialize field
    for cell in mesh.for_each_cell():
        u[cell] = np.sin(4 * np.pi * cell.center()[0])

    initial_cells = mesh.nb_cells()

    # Adapt mesh
    samurai.adapt(mesh, epsilon, lambda c: abs(u[c]))

    # Verify mesh changed
    assert mesh.nb_cells() != initial_cells
```

### 10.2 C++ Unit Tests

```cpp
// tests/python_bindings/test_field_buffer.cpp

#include <pybind11/embed.h>
#include <samurai/field.hpp>
#include <gtest/gtest.h>

class FieldBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Python interpreter
        py::scoped_interpreter guard{};

        // Import samurai module
        samurai = py::module_::import("samurai");
    }

    py::module_ samurai;
};

TEST_F(FieldBufferTest, ZeroCopyAccess) {
    using namespace samurai;

    // Create mesh and field in C++
    auto config = mesh_config<2>{};
    config.min_level = 2;
    config.max_level = 4;

    Box<double, 2> box({0., 0.}, {1., 1.});
    auto mesh = AMRMesh<2>(box, config);
    auto u = make_scalar_field<double>("u", mesh);

    // Get Python field
    auto py_u = samurai.attr("ScalarField")("u", mesh);
    auto py_arr = py_u.attr("numpy_view")();

    // Verify zero-copy
    py::array_t<double> arr(py_arr);
    auto* cpp_data = u.array().data();
    auto* py_data = static_cast<double*>(arr.request().ptr);

    EXPECT_EQ(cpp_data, py_data);
}
```

### 10.3 Benchmark Tests

```python
# tests/python/benchmark_performance.py

import pytest
import samurai
import numpy as np

def test_field_fill_performance(benchmark):
    """Benchmark field filling operation."""
    config = samurai.MeshConfig2D()
    config.min_level = 4
    config.max_level = 8

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)
    u = samurai.ScalarField("u", mesh)

    def fill_field():
        u.fill(1.0)

    benchmark(fill_field)

def test_numpy_vectorized_ops(benchmark):
    """Benchmark vectorized NumPy operations."""
    config = samurai.MeshConfig2D()
    config.min_level = 4
    config.max_level = 8

    box = samurai.Box2D([0., 0.], [1., 1.])
    mesh = samurai.Mesh2D(box, config)
    u = samurai.ScalarField("u", mesh)

    def vectorized_operation():
        u_arr = u.numpy_view()
        u_arr[:] = np.sin(u_arr) + np.cos(u_arr)

    benchmark(vectorized_operation)
```

---

## 11. Documentation

### 11.1 Sphinx Documentation

```rst
# docs/python_api.rst

Python API Reference
====================

Core Classes
------------

.. autoclass:: samurai.Mesh2D
   :members:

.. autoclass:: samurai.ScalarField
   :members:

.. autoclass:: samurai.VectorField
   :members:

I/O Operations
--------------

.. autofunction:: samurai.save

.. autofunction:: samurai.load

Algorithms
----------

.. autofunction:: samurai.for_each_cell

.. autofunction:: samurai.adapt

.. autofunction:: samurai.make_bc
```

### 11.2 Type Stubs (mypy Support)

```python
# samurai/core/__init__.pyi

from typing import Protocol, TypeVar, Callable
import numpy as np

class MeshConfig:
    min_level: int
    max_level: int
    graduation_width: int
    scaling_factor: float

class Mesh(Protocol):
    @property
    def dim(self) -> int: ...
    @property
    def min_level(self) -> int: ...
    @property
    def max_level(self) -> int: ...
    @property
    def nb_cells(self) -> int: ...

    def for_each_cell(self, func: Callable[['Cell'], None]) -> None: ...

class Cell(Protocol):
    @property
    def level(self) -> int: ...
    @property
    def center(self) -> np.ndarray: ...
    @property
    def length(self) -> float: ...

class ScalarField:
    name: str
    mesh: Mesh

    def __init__(self, name: str, mesh: Mesh) -> None: ...
    def numpy_view(self) -> np.ndarray: ...
    def fill(self, value: float) -> None: ...
    def __getitem__(self, cell: Cell) -> float: ...
    def __setitem__(self, cell: Cell, value: float) -> None: ...
```

### 11.3 Jupyter Notebook Tutorial

```python
# examples/python_tutorial.ipynb

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Samurai Python Tutorial\n",
    "\n",
    "This notebook demonstrates the Python bindings for Samurai V2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import samurai\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Create mesh configuration\n",
    "config = samurai.MeshConfig2D()\n",
    "config.min_level = 3\n",
    "config.max_level = 7\n",
    "\n",
    "# Define computational domain\n",
    "box = samurai.Box2D([0., 0.], [1., 1.])\n",
    "mesh = samurai.Mesh2D(box, config)\n",
    "\n",
    "print(f\"Mesh created: {mesh.nb_cells()} cells\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create field from Python function\n",
    "def init_condition(x):\n",
    "    \"\"\"Initial condition: Gaussian pulse.\"\"\"\n",
    "    x0, y0 = 0.5, 0.5\n",
    "    sigma = 0.1\n",
    "    r2 = (x[0]-x0)**2 + (x[1]-y0)**2\n",
    "    return np.exp(-r2 / (2*sigma**2))\n",
    "\n",
    "u = samurai.make_scalar_field(\"solution\", mesh, init_condition)\n",
    "\n",
    "# Plot field\n",
    "fig, ax = plt.subplots()\n",
    "for cell in mesh.for_each_cell():\n",
    "    x, y = cell.center()\n",
    "    ax.plot(x, y, 'o', markersize=cell.length*10)\n",
    "plt.show()"
   ]
  }
 ]
}
```

---

## 12. Performance Considerations

### 12.1 Memory Management

```cpp
// Keep parent object alive when returning views
.def("numpy_view", [](Field& f) {
    using T = typename Field::value_type;
    return py::array_t<T>(
        f.array().shape(),
        f.array().strides(),
        f.array().data(),
        py::cast(f)  // CRITICAL: keeps field alive
    );
}, py::return_value_policy::take_ownership)
```

### 12.2 Minimizing Python Overhead

```cpp
// Bad: Python loop overhead
mesh.for_each_cell(lambda cell: u[cell] = f(cell))

// Good: Vectorized operation via NumPy
u_arr = u.numpy_view()
u_arr[:] = f_vectorized(x_coords, y_coords)

// Even better: Pure C++ algorithm
samurai::apply_on_mesh(mesh, [&](auto& cell) {
    u[cell] = cpp_function(cell);
});
```

### 12.3 SIMD Vectorization

```cpp
// Ensure data alignment for SIMD
#pragma omp simd aligned(ptr: 64)
for (std::size_t i = 0; i < size; ++i) {
    ptr[i] = func(i);
}
```

### 12.4 Performance Best Practices

| Practice | Impact | Recommendation |
|----------|--------|---------------|
| Zero-copy NumPy views | High | Always use when possible |
| Vectorized NumPy ops | High | Prefer over Python loops |
| Compiled algorithms | Highest | Use C++ for core computations |
| Minimize pybind11 calls | Medium | Batch operations when possible |
| Memory alignment | Medium | Enable for SIMD |

---

## 13. Future Extensions

### 13.1 Numba JIT Integration

```python
# Future: Numba-compatible functions
import numba

@numba.jit
def numba_compatible_update(u_arr, dt):
    """JIT-compiled field update."""
    n = u_arr.shape[0]
    for i in range(n):
        u_arr[i] += dt * f(i)
    return u_arr
```

### 13.2 GPU Support

```cpp
#ifdef SAMURAI_WITH_CUDA
.def("to_gpu", [](Field& f) {
    return FieldGPU(f);  // Transfer to GPU
});
#endif
```

### 13.3 Multiprocessing Support

```python
# Future: Multiprocessing-friendly interface
from multiprocessing import Pool

def process_chunk(mesh_chunk):
    """Process mesh chunk in separate process."""
    u = samurai.ScalarField("u", mesh_chunk)
    # ... computation ...
    return u

with Pool() as p:
    results = p.map(process_chunk, mesh.partition(n_workers))
```

---

## 14. Summary and Next Steps

### 14.1 Implementation Roadmap

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1. Core bindings | Mesh, Field, Cell wrappers | 4 weeks |
| 2. NumPy integration | Zero-copy, buffer protocol | 2 weeks |
| 3. Algorithm bindings | for_each, adapt, BC | 3 weeks |
| 4. I/O operations | HDF5 save/load | 2 weeks |
| 5. Build system | CMake, setup.py, wheels | 2 weeks |
| 6. Testing | pytest, benchmarks | 3 weeks |
| 7. Documentation | Sphinx, tutorials | 2 weeks |
| **Total** | | **18 weeks** |

### 14.2 Key Success Metrics

- **Performance**: < 5% overhead vs. pure C++
- **Coverage**: > 90% of core API exposed
- **Documentation**: 100% public API documented
- **Testing**: > 80% code coverage
- **Usability**: < 10 minutes to first simulation

### 14.3 Recommended First Steps

1. **Proof of Concept**: Implement Mesh2D wrapper with basic operations
2. **NumPy Integration**: Demonstrate zero-copy field access
3. **Simple Simulation**: End-to-end 1D advection example
4. **Performance Validation**: Benchmark against C++ implementation
5. **Documentation**: Write tutorial notebook

---

## Appendix A: Complete Example

```python
# examples/python_heat_equation.py

"""
1D Heat equation using Samurai Python bindings.

Equation: ∂u/∂t = α ∂²u/∂x²
"""

import samurai
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0           # Domain length
T = 0.1           # Final time
alpha = 0.01      # Thermal diffusivity
nx = 100          # Number of cells (base level)

# Create mesh
config = samurai.MeshConfig1D()
config.min_level = 4
config.max_level = 8

box = samurai.Box1D([0.], [L])
mesh = samurai.Mesh1D(box, config)

# Create field
def initial_condition(x):
    """Gaussian initial temperature distribution."""
    return np.exp(-100 * (x[0] - 0.5)**2)

u = samurai.make_scalar_field("temperature", mesh, initial_condition)

# Time stepping
dx = L / nx
dt = 0.4 * dx**2 / alpha  # CFL condition
n_steps = int(T / dt)

# Boundary conditions (Dirichlet)
u.attach_bc(lambda *args: 0., samurai.Direction(0))  # Left
u.attach_bc(lambda *args: 0., samurai.Direction(1))  # Right

# Main loop
for n in range(n_steps):
    # Get zero-copy NumPy view
    u_arr = u.numpy_view()

    # Compute Laplacian (central difference)
    u_xx = np.zeros_like(u_arr)
    u_xx[1:-1] = (u_arr[2:] - 2*u_arr[1:-1] + u_arr[:-2]) / dx**2

    # Update field
    u_arr += alpha * dt * u_xx

    # Apply boundary conditions
    u_arr[0] = 0.
    u_arr[-1] = 0.

    # Adapt mesh every 10 steps
    if n % 10 == 0:
        def refine_criterion(cell):
            return abs(u[cell]) > 0.1

        samurai.adapt(mesh, 0.01, refine_criterion)

    # Visualization
    if n % 100 == 0:
        plt.figure()
        x_centers = [c.center()[0] for c in mesh.for_each_cell()]
        plt.plot(x_centers, u_arr, 'o-')
        plt.title(f"Temperature distribution at t={n*dt:.3f}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.grid(True)
        plt.show()

# Save final result
samurai.save("heat_equation_solution", u)
print(f"Simulation complete. Final mesh: {mesh.nb_cells()} cells")
```

---

## Appendix B: Type Reference

```cpp
// Key C++ types to expose

namespace samurai {

// Core types
template <std::size_t dim, class TInterval, std::size_t max_size>
class CellArray;

template <std::size_t dim, class TInterval>
class LevelCellArray;

template <class D, class Config>
class Mesh_base;

template <class mesh_t, class value_t>
class ScalarField;

template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
class VectorField;

// Algorithms
template <class Mesh, class Func>
void for_each_cell(const Mesh&, Func&&);

template <class Mesh, class Func>
void for_each_level(const Mesh&, Func&&);

template <std::size_t dim, class TInterval>
auto find(const LevelCellArray<dim, TInterval>&,
          const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>&);

// I/O
template <class mesh_t, class... T>
void save(const fs::path&, const std::string&,
          const Hdf5Options<mesh_t>&, const mesh_t&, const T&...);

// Boundary conditions
template <class Field>
class Bc;

template <class Field>
class ConstantBc;

template <class Field>
class FunctionBc;

}
```

---

**End of Document**

---

This technical report provides a complete blueprint for implementing Python bindings to Samurai V2 using pybind11. The design prioritizes performance through zero-copy NumPy integration while maintaining a Pythonic API that integrates seamlessly with the scientific Python ecosystem.
