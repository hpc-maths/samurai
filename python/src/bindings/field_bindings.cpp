// Samurai Python Bindings - ScalarField and VectorField classes
//
// Bindings for samurai::ScalarField and samurai::VectorField classes
// Uses NumPy buffer protocol for zero-copy interoperability

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/cell.hpp>
#include <samurai/algorithm.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

namespace py = pybind11;

// Type aliases matching MRMesh bindings
// Use the default interval type from Samurai
using default_interval = samurai::Interval<double, std::size_t>;

template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

template <std::size_t dim, std::size_t n_comp, bool SOA = false>
using VectorField = samurai::VectorField<MRMesh<dim>, double, n_comp, SOA>;

template <std::size_t dim>
using Cell = samurai::Cell<dim, default_interval>;

// ============================================================
// Field arithmetic operation helpers
// ============================================================

// Field - scalar operations (immediate evaluation, return new field)
template <std::size_t dim>
ScalarField<dim> field_sub_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_sub", mesh);
    result = field - scalar;
    return result;
}

template <std::size_t dim>
ScalarField<dim> scalar_sub_field(double scalar, const ScalarField<dim>& field)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>("scalar_sub", mesh);
    result = scalar - field;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_add_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_add", mesh);
    result = field + scalar;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_mul_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_mul", mesh);
    result = field * scalar;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_div_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_div", mesh);
    result = field / scalar;
    return result;
}

// Field - field operations
template <std::size_t dim>
ScalarField<dim> field_sub_field(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    auto result = samurai::make_scalar_field<double>(field1.name() + "_sub", mesh);
    result = field1 - field2;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_add_field(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    auto result = samurai::make_scalar_field<double>(field1.name() + "_add", mesh);
    result = field1 + field2;
    return result;
}

// Field utility operations
template <std::size_t dim>
ScalarField<dim> field_clone(const ScalarField<dim>& field)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_clone", mesh);
    result = field;  // Deep copy via assignment operator
    return result;
}

template <std::size_t dim>
void field_copy_to(ScalarField<dim>& dest, const ScalarField<dim>& src)
{
    dest = src;  // Uses copy assignment operator
}

// Helper to bind common Field methods
template <class Field, class Mesh, class... Options>
void bind_field_common_methods(py::class_<Field, Options...>& cls) {
    using value_t = typename Field::value_type;

    // Name property
    cls.def_property("name",
        [](const Field& f) -> const std::string& {
            return f.name();
        },
        [](Field& f, const std::string& name) -> std::string& {
            f.name() = name;
            return f.name();
        },
        "Field name (read/write)"
    );

    // Mesh property (reference_internal ensures correct lifetime)
    cls.def_property_readonly("mesh",
        [](const Field& f) -> const Mesh& {
            return f.mesh();
        },
        py::return_value_policy::reference_internal,
        "Mesh this field is defined on"
    );

    // Dimension property
    cls.def_property_readonly("dim",
        [](const Field&) { return Field::dim; },
        "Field dimension"
    );

    // Size (number of cells)
    cls.def_property_readonly("size",
        [](const Field& f) -> std::size_t {
            return f.mesh().nb_cells();
        },
        "Total number of cells in the field"
    );

    // Fill with constant value
    cls.def("fill",
        &Field::fill,
        py::arg("value"),
        "Fill all cells with a constant value"
    );

    // Ghost cells flag
    cls.def_property("ghosts_updated",
        [](const Field& f) -> bool {
            return f.ghosts_updated();
        },
        [](Field& f, bool value) -> bool& {
            f.ghosts_updated() = value;
            return f.ghosts_updated();
        },
        "Ghost cells update flag"
    );
}

// Template function to bind ScalarField for a specific dimension
template <std::size_t dim>
void bind_scalar_field(py::module_& m, const std::string& name) {
    using Mesh = MRMesh<dim>;
    using Field = ScalarField<dim>;
    using value_t = typename Field::value_type;

    // Create class with docstring
    auto cls = py::class_<Field>(m, name.c_str(), R"pbdoc(
        Scalar Field on adaptive mesh

        A scalar field defined on an adaptive mesh refinement grid.
        Provides zero-copy NumPy integration for efficient data access.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to define the field on
        name : str
            Field identifier
        init_value : float, optional
            Initial value for all cells (default: 0.0)

        Examples
        --------
        >>> import samurai as sam
        >>> box = sam.Box2D([0., 0.], [1., 1.])
        >>> config = sam.MeshConfig2D()
        >>> config.min_level = 0
        >>> config.max_level = 2
        >>> mesh = sam.MRMesh2D(box, config)
        >>> field = sam.ScalarField2D("u", mesh)
        >>> field.fill(1.0)

        NumPy Integration
        -----------------
        >>> import numpy as np
        >>> arr = field.numpy_view()  # Zero-copy view
        >>> arr[:] = np.sin(x) * np.cos(y)
        >>> # Modifying arr modifies field in-place

        Attributes
        ----------
        name : str
            Field name
        mesh : MRMesh
            Underlying mesh
        size : int
            Number of cells
    )pbdoc");

    // Constructor using factory function
    cls.def(py::init([](const std::string& field_name, Mesh& mesh, value_t init_value) {
        auto field = samurai::make_scalar_field<value_t>(field_name, mesh, init_value);
        return field;
    }),
        py::arg("name"),
        py::arg("mesh"),
        py::arg("init_value") = 0.0,
        py::keep_alive<1, 2>(),  // Field keeps Mesh alive
        "Create scalar field"
    );

    // Bind common methods
    bind_field_common_methods<Field, Mesh>(cls);

    // Explicit numpy_view method (zero-copy NumPy integration)
    cls.def("numpy_view",
        [](Field& f) -> py::array_t<value_t> {
            auto& xt = f.array();
            return py::array_t<value_t>(
                {xt.size()},             // Shape
                {sizeof(value_t)},       // Strides
                xt.data(),               // Data pointer
                py::cast(f)              // Keep field alive
            );
        },
        py::return_value_policy::take_ownership,
        "Returns zero-copy NumPy view of field data"
    );

    // Integer-based indexing
    // Note: For CellWrapper-based indexing from for_each_cell, use field[cell.index]
    cls.def("__getitem__",
        [](Field& f, std::size_t i) -> value_t {
            return f[i];
        },
        py::arg("index"),
        "Get field value by flat index"
    );

    cls.def("__setitem__",
        [](Field& f, std::size_t i, value_t value) {
            f[i] = value;
        },
        py::arg("index"),
        py::arg("value"),
        "Set field value by flat index"
    );

    // Arithmetic operators: field +/-/* scalar
    cls.def("__sub__", &field_sub_scalar<dim>,
        py::arg("scalar"),
        "Subtract scalar from field (returns new field)"
    );

    cls.def("__rsub__", &scalar_sub_field<dim>,
        py::arg("scalar"),
        "Subtract field from scalar (returns new field)"
    );

    cls.def("__add__", &field_add_scalar<dim>,
        py::arg("scalar"),
        "Add scalar to field (returns new field)"
    );

    cls.def("__radd__", &field_add_scalar<dim>,
        py::arg("scalar"),
        "Add scalar to field (right-hand version)"
    );

    cls.def("__mul__", &field_mul_scalar<dim>,
        py::arg("scalar"),
        "Multiply field by scalar (returns new field)"
    );

    cls.def("__rmul__", &field_mul_scalar<dim>,
        py::arg("scalar"),
        "Multiply field by scalar (right-hand version)"
    );

    cls.def("__truediv__", &field_div_scalar<dim>,
        py::arg("scalar"),
        "Divide field by scalar (returns new field)"
    );

    // Field-to-field operators
    cls.def("__sub__", &field_sub_field<dim>,
        py::arg("other"),
        "Subtract another field (returns new field)"
    );

    cls.def("__add__", &field_add_field<dim>,
        py::arg("other"),
        "Add another field (returns new field)"
    );

    // Utility methods
    cls.def("clone", &field_clone<dim>,
        "Create a deep copy of this field"
    );

    cls.def("copy_to", &field_copy_to<dim>,
        py::arg("dest"),
        "Copy this field's data to destination field"
    );

    // String representation
    cls.def("__repr__",
        [](const Field& f) {
            std::ostringstream oss;
            oss << "ScalarField" << Field::dim << "D(";
            oss << "name='" << f.name() << "', ";
            oss << "size=" << f.mesh().nb_cells();
            oss << ")";
            return oss.str();
        });

    cls.def("__str__",
        [](const Field& f) {
            std::ostringstream oss;
            oss << "ScalarField" << Field::dim << "D";
            oss << " '" << f.name() << "'";
            oss << " [" << f.mesh().nb_cells() << " cells]";
            return oss.str();
        });
}

// Helper to bind VectorField-specific methods
template <class Field, class Mesh, class... Options>
void bind_vectorfield_methods(py::class_<Field, Options...>& cls) {
    using value_t = typename Field::value_type;
    static constexpr std::size_t n_comp = Field::n_comp;

    // Number of components property
    cls.def_property_readonly("n_components",
        [](const Field&) { return n_comp; },
        "Number of components"
    );

    // Is SOA layout property
    cls.def_property_readonly("is_soa",
        [](const Field&) { return Field::is_soa; },
        "True if Structure of Arrays layout, False if Array of Structures"
    );

    // Fill with scalar value (use the default fill method)
    cls.def("fill",
        &Field::fill,
        py::arg("value"),
        "Fill all components and cells with a constant value"
    );

    // Fill with per-component values
    cls.def("fill",
        [](Field& f, py::list values) {
            if (values.size() != n_comp) {
                throw std::runtime_error("Expected " + std::to_string(n_comp) + " values, got " + std::to_string(values.size()));
            }
            std::vector<value_t> vals;
            for (std::size_t i = 0; i < n_comp; ++i) {
                vals.push_back(values[i].cast<value_t>());
            }

            auto& xt = f.array();
            if constexpr (Field::is_soa) {
                // SOA: fill each component
                for (std::size_t comp = 0; comp < n_comp; ++comp) {
                    std::size_t n_cells = xt.shape()[1];
                    for (std::size_t i = 0; i < n_cells; ++i) {
                        xt(comp, i) = vals[comp];
                    }
                }
            } else {
                // AOS: fill each cell's components
                std::size_t n_cells = xt.shape()[0];
                for (std::size_t i = 0; i < n_cells; ++i) {
                    for (std::size_t comp = 0; comp < n_comp; ++comp) {
                        xt(i, comp) = vals[comp];
                    }
                }
            }
        },
        py::arg("values"),
        "Fill all cells with per-component values"
    );

    // String representation
    cls.def("__repr__",
        [](const Field& f) {
            std::ostringstream oss;
            oss << "VectorField" << Field::dim << "D(";
            oss << "name='" << f.name() << "', ";
            oss << "n_comp=" << n_comp << ", ";
            oss << "size=" << f.mesh().nb_cells();
            oss << ")";
            return oss.str();
        });

    cls.def("__str__",
        [](const Field& f) {
            std::ostringstream oss;
            oss << "VectorField" << Field::dim << "D";
            oss << " '" << f.name() << "'";
            oss << " [" << n_comp << " components]";
            oss << " [" << f.mesh().nb_cells() << " cells]";
            return oss.str();
        });

    // Explicit numpy_view method
    cls.def("numpy_view",
        [](Field& f) -> py::array_t<value_t> {
            auto& xt = f.array();

            if constexpr (Field::is_soa) {
                // SOA: (n_components, n_cells)
                return py::array_t<value_t>(
                    {n_comp, static_cast<std::size_t>(xt.shape()[1])},
                    {static_cast<std::size_t>(xt.shape()[1]) * sizeof(value_t), sizeof(value_t)},
                    xt.data(),
                    py::cast(f)
                );
            } else {
                // AOS: (n_cells, n_components)
                return py::array_t<value_t>(
                    {static_cast<std::size_t>(xt.shape()[0]), n_comp},
                    {n_comp * sizeof(value_t), sizeof(value_t)},
                    xt.data(),
                    py::cast(f)
                );
            }
        },
        py::return_value_policy::take_ownership,
        "Returns zero-copy NumPy view of vector field data"
    );

    // Get component as scalar field (returns new field with extracted component)
    cls.def("get_component",
        [](Field& f, std::size_t comp) {
            // Create a new scalar field with the same mesh
            auto result = samurai::make_scalar_field<value_t>(f.name() + "_comp" + std::to_string(comp), const_cast<Mesh&>(f.mesh()));

            // Copy component data
            auto& src = f.array();
            auto& dst = result.array();

            if constexpr (Field::is_soa) {
                // SOA: component is contiguous
                std::size_t n_cells = src.shape()[1];
                for (std::size_t i = 0; i < n_cells; ++i) {
                    dst[i] = src(comp, i);
                }
            } else {
                // AOS: component is strided
                std::size_t n_cells = src.shape()[0];
                for (std::size_t i = 0; i < n_cells; ++i) {
                    dst[i] = src(i, comp);
                }
            }

            return result;
        },
        py::arg("component"),
        "Extract a single component as a new ScalarField"
    );
}

// Template function to bind VectorField for a specific dimension and component count
template <std::size_t dim, std::size_t n_comp, bool SOA>
void bind_vector_field(py::module_& m, const std::string& name) {
    using Mesh = MRMesh<dim>;
    using Field = VectorField<dim, n_comp, SOA>;
    using value_t = typename Field::value_type;

    // Create class with docstring
    auto cls = py::class_<Field>(m, name.c_str(), R"pbdoc(
        Vector Field on adaptive mesh

        A multi-component field defined on an adaptive mesh refinement grid.
        Provides zero-copy NumPy integration for efficient data access.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to define the field on
        name : str
            Field identifier
        init_value : float, optional
            Initial value for all components and cells

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> velocity = sam.VectorField2D_2("vel", mesh, 2)
        >>> velocity.fill([1.0, 0.0])  # Set (u, v) components

        NumPy Integration
        -----------------
        >>> arr = velocity.numpy_view()  # Shape: (n_cells, n_components)
        >>> arr[:, 0] = u_values  # Set first component
        >>> arr[:, 1] = v_values  # Set second component

        Attributes
        ----------
        name : str
            Field name
        mesh : MRMesh
            Underlying mesh
        n_components : int
            Number of components
        is_soa : bool
            Memory layout (True=Structure of Arrays, False=Array of Structures)
    )pbdoc");

    // Constructor using factory function
    cls.def(py::init([](const std::string& field_name, Mesh& mesh, value_t init_value) {
        auto field = samurai::make_vector_field<value_t, n_comp, SOA>(field_name, mesh, init_value);
        return field;
    }),
        py::arg("name"),
        py::arg("mesh"),
        py::arg("init_value") = 0.0,
        py::keep_alive<1, 2>(),  // Field keeps Mesh alive
        "Create vector field"
    );

    // Bind common methods (name, mesh, size, etc.)
    bind_field_common_methods<Field, Mesh>(cls);

    // Bind vector-specific methods
    bind_vectorfield_methods<Field, Mesh>(cls);
}

// Module initialization function for Field bindings
void init_field_bindings(py::module_& m) {
    // Bind ScalarField classes for dimensions 1, 2, and 3
    bind_scalar_field<1>(m, "ScalarField1D");
    bind_scalar_field<2>(m, "ScalarField2D");
    bind_scalar_field<3>(m, "ScalarField3D");

    // Bind VectorField classes for 1D (AOS layout)
    bind_vector_field<1, 2, false>(m, "VectorField1D_2");
    bind_vector_field<1, 3, false>(m, "VectorField1D_3");

    // Bind VectorField classes for 2 components (AOS layout)
    bind_vector_field<2, 2, false>(m, "VectorField2D_2");
    bind_vector_field<2, 3, false>(m, "VectorField2D_3");

    // Bind VectorField classes for 3D (AOS layout)
    bind_vector_field<3, 2, false>(m, "VectorField3D_2");
    bind_vector_field<3, 3, false>(m, "VectorField3D_3");

    // Factory functions for convenience - using overloaded functions
    // 1D scalar field factory
    m.def("make_scalar_field",
        [](MRMesh<1>& mesh, const std::string& field_name, double init_value) {
            return samurai::make_scalar_field<double>(field_name, mesh, init_value);
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("init_value") = 0.0,
        "Create a 1D scalar field"
    );

    // 2D scalar field factory
    m.def("make_scalar_field",
        [](MRMesh<2>& mesh, const std::string& field_name, double init_value) {
            return samurai::make_scalar_field<double>(field_name, mesh, init_value);
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("init_value") = 0.0,
        "Create a 2D scalar field"
    );

    // 3D scalar field factory
    m.def("make_scalar_field",
        [](MRMesh<3>& mesh, const std::string& field_name, double init_value) {
            return samurai::make_scalar_field<double>(field_name, mesh, init_value);
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("init_value") = 0.0,
        "Create a 3D scalar field"
    );

    // 1D VectorField factory function
    m.def("make_vector_field",
        [](MRMesh<1>& mesh, const std::string& field_name, std::size_t n_components, double init_value) -> py::object {
            if (n_components == 2) {
                auto field = samurai::make_vector_field<double, 2, false>(field_name, mesh, init_value);
                return py::cast(std::move(field));
            } else if (n_components == 3) {
                auto field = samurai::make_vector_field<double, 3, false>(field_name, mesh, init_value);
                return py::cast(std::move(field));
            } else {
                throw std::runtime_error("Unsupported n_components: " + std::to_string(n_components));
            }
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("n_components"),
        py::arg("init_value") = 0.0,
        "Create a 1D vector field with specified number of components"
    );

    // VectorField factory function - dispatch based on n_components
    m.def("make_vector_field",
        [](MRMesh<2>& mesh, const std::string& field_name, std::size_t n_components, double init_value) -> py::object {
            if (n_components == 2) {
                auto field = samurai::make_vector_field<double, 2, false>(field_name, mesh, init_value);
                return py::cast(std::move(field));
            } else if (n_components == 3) {
                auto field = samurai::make_vector_field<double, 3, false>(field_name, mesh, init_value);
                return py::cast(std::move(field));
            } else {
                throw std::runtime_error("Unsupported n_components: " + std::to_string(n_components));
            }
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("n_components"),
        py::arg("init_value") = 0.0,
        "Create a 2D vector field with specified number of components"
    );

    // 3D VectorField factory function
    m.def("make_vector_field",
        [](MRMesh<3>& mesh, const std::string& field_name, std::size_t n_components, double init_value) -> py::object {
            if (n_components == 2) {
                auto field = samurai::make_vector_field<double, 2, false>(field_name, mesh, init_value);
                return py::cast(std::move(field));
            } else if (n_components == 3) {
                auto field = samurai::make_vector_field<double, 3, false>(field_name, mesh, init_value);
                return py::cast(std::move(field));
            } else {
                throw std::runtime_error("Unsupported n_components: " + std::to_string(n_components));
            }
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("n_components"),
        py::arg("init_value") = 0.0,
        "Create a 3D vector field with specified number of components"
    );

    // Also expose them in a submodule for better organization
    py::module_ field = m.def_submodule("field", "Field classes");
    field.attr("ScalarField1D") = m.attr("ScalarField1D");
    field.attr("ScalarField2D") = m.attr("ScalarField2D");
    field.attr("ScalarField3D") = m.attr("ScalarField3D");
    field.attr("VectorField1D_2") = m.attr("VectorField1D_2");
    field.attr("VectorField1D_3") = m.attr("VectorField1D_3");
    field.attr("VectorField2D_2") = m.attr("VectorField2D_2");
    field.attr("VectorField2D_3") = m.attr("VectorField2D_3");
    field.attr("VectorField3D_2") = m.attr("VectorField3D_2");
    field.attr("VectorField3D_3") = m.attr("VectorField3D_3");

    // ============================================================
    // Time-stepping helper functions
    // ============================================================
    // Note: CellWrapper classes (Cell1D, Cell2D, Cell3D) are already bound in algorithm_bindings.cpp
    // ============================================================
    // Euler update: unp1 = u - dt * du
    m.def("euler_update_1d",
        [](ScalarField<1>& unp1, const ScalarField<1>& u, double dt, const ScalarField<1>& du) {
            unp1 = u - dt * du;
        },
        py::arg("unp1"),
        py::arg("u"),
        py::arg("dt"),
        py::arg("du"),
        "Euler time step update (1D): unp1 = u - dt * du"
    );

    m.def("euler_update_2d",
        [](ScalarField<2>& unp1, const ScalarField<2>& u, double dt, const ScalarField<2>& du) {
            unp1 = u - dt * du;
        },
        py::arg("unp1"),
        py::arg("u"),
        py::arg("dt"),
        py::arg("du"),
        "Euler time step update (2D): unp1 = u - dt * du"
    );

    m.def("euler_update_3d",
        [](ScalarField<3>& unp1, const ScalarField<3>& u, double dt, const ScalarField<3>& du) {
            unp1 = u - dt * du;
        },
        py::arg("unp1"),
        py::arg("u"),
        py::arg("dt"),
        py::arg("du"),
        "Euler time step update (3D): unp1 = u - dt * du"
    );

    // RK3 stage 2: u2 = 3/4*u + 1/4*(u1 - dt*du1)
    m.def("rk3_stage2_1d",
        [](ScalarField<1>& u2, const ScalarField<1>& u, const ScalarField<1>& u1, double dt, const ScalarField<1>& du1) {
            u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * du1);
        },
        py::arg("u2"),
        py::arg("u"),
        py::arg("u1"),
        py::arg("dt"),
        py::arg("du1"),
        "RK3 stage 2 update (1D): u2 = 3/4*u + 1/4*(u1 - dt*du1)"
    );

    m.def("rk3_stage2_2d",
        [](ScalarField<2>& u2, const ScalarField<2>& u, const ScalarField<2>& u1, double dt, const ScalarField<2>& du1) {
            u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * du1);
        },
        py::arg("u2"),
        py::arg("u"),
        py::arg("u1"),
        py::arg("dt"),
        py::arg("du1"),
        "RK3 stage 2 update (2D): u2 = 3/4*u + 1/4*(u1 - dt*du1)"
    );

    m.def("rk3_stage2_3d",
        [](ScalarField<3>& u2, const ScalarField<3>& u, const ScalarField<3>& u1, double dt, const ScalarField<3>& du1) {
            u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * du1);
        },
        py::arg("u2"),
        py::arg("u"),
        py::arg("u1"),
        py::arg("dt"),
        py::arg("du1"),
        "RK3 stage 2 update (3D): u2 = 3/4*u + 1/4*(u1 - dt*du1)"
    );

    // RK3 stage 3: unp1 = 1/3*u + 2/3*(u2 - dt*du2)
    m.def("rk3_stage3_1d",
        [](ScalarField<1>& unp1, const ScalarField<1>& u, const ScalarField<1>& u2, double dt, const ScalarField<1>& du2) {
            unp1 = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * du2);
        },
        py::arg("unp1"),
        py::arg("u"),
        py::arg("u2"),
        py::arg("dt"),
        py::arg("du2"),
        "RK3 stage 3 update (1D): unp1 = 1/3*u + 2/3*(u2 - dt*du2)"
    );

    m.def("rk3_stage3_2d",
        [](ScalarField<2>& unp1, const ScalarField<2>& u, const ScalarField<2>& u2, double dt, const ScalarField<2>& du2) {
            unp1 = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * du2);
        },
        py::arg("unp1"),
        py::arg("u"),
        py::arg("u2"),
        py::arg("dt"),
        py::arg("du2"),
        "RK3 stage 3 update (2D): unp1 = 1/3*u + 2/3*(u2 - dt*du2)"
    );

    m.def("rk3_stage3_3d",
        [](ScalarField<3>& unp1, const ScalarField<3>& u, const ScalarField<3>& u2, double dt, const ScalarField<3>& du2) {
            unp1 = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * du2);
        },
        py::arg("unp1"),
        py::arg("u"),
        py::arg("u2"),
        py::arg("dt"),
        py::arg("du2"),
        "RK3 stage 3 update (3D): unp1 = 1/3*u + 2/3*(u2 - dt*du2)"
    );

    // Field swap for efficient time stepping
    m.def("swap_field_arrays_1d",
        [](ScalarField<1>& f1, ScalarField<1>& f2) {
            std::swap(f1.array(), f2.array());
            // Also swap ghost update flags
            bool f1_ghosts = f1.ghosts_updated();
            f1.ghosts_updated() = f2.ghosts_updated();
            f2.ghosts_updated() = f1_ghosts;
        },
        py::arg("f1"),
        py::arg("f2"),
        "Swap underlying data arrays of two 1D fields (efficient for time stepping)"
    );

    m.def("swap_field_arrays_2d",
        [](ScalarField<2>& f1, ScalarField<2>& f2) {
            std::swap(f1.array(), f2.array());
            bool f1_ghosts = f1.ghosts_updated();
            f1.ghosts_updated() = f2.ghosts_updated();
            f2.ghosts_updated() = f1_ghosts;
        },
        py::arg("f1"),
        py::arg("f2"),
        "Swap underlying data arrays of two 2D fields (efficient for time stepping)"
    );

    m.def("swap_field_arrays_3d",
        [](ScalarField<3>& f1, ScalarField<3>& f2) {
            std::swap(f1.array(), f2.array());
            bool f1_ghosts = f1.ghosts_updated();
            f1.ghosts_updated() = f2.ghosts_updated();
            f2.ghosts_updated() = f1_ghosts;
        },
        py::arg("f1"),
        py::arg("f2"),
        "Swap underlying data arrays of two 3D fields (efficient for time stepping)"
    );
}
