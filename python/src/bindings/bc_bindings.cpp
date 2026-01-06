// Samurai Python Bindings - Boundary Conditions
//
// Bindings for make_bc and boundary condition types

#include <samurai/bc/bc.hpp>
#include <samurai/bc/dirichlet.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/field.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Type aliases matching field_bindings.cpp
template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

// ============================================================
// Dirichlet boundary condition bindings
// ============================================================

// Helper to attach Dirichlet BC for a specific dimension and order
// Returns void because the BC is attached to the field internally
template <std::size_t dim, std::size_t order>
void make_dirichlet_bc_scalar(ScalarField<dim>& field, double value)
{
    using DirichletOrder = samurai::Dirichlet<order>;
    samurai::make_bc<DirichletOrder>(field, value);
    // BC is now attached to the field - no need to return anything
}

// Wrapper function to dispatch based on order parameter
template <std::size_t dim>
void make_dirichlet_bc_dispatch(ScalarField<dim>& field, double value, std::size_t order)
{
    switch (order)
    {
    case 1:
        return make_dirichlet_bc_scalar<dim, 1>(field, value);
    case 2:
        return make_dirichlet_bc_scalar<dim, 2>(field, value);
    case 3:
        return make_dirichlet_bc_scalar<dim, 3>(field, value);
    case 4:
        return make_dirichlet_bc_scalar<dim, 4>(field, value);
    default:
        throw std::runtime_error("Dirichlet BC order must be between 1 and 4, got " + std::to_string(order));
    }
}

// 1D wrapper
void make_dirichlet_bc_1d(ScalarField<1>& field, double value, std::size_t order)
{
    make_dirichlet_bc_dispatch<1>(field, value, order);
}

// 2D wrapper
void make_dirichlet_bc_2d(ScalarField<2>& field, double value, std::size_t order)
{
    make_dirichlet_bc_dispatch<2>(field, value, order);
}

// 3D wrapper
void make_dirichlet_bc_3d(ScalarField<3>& field, double value, std::size_t order)
{
    make_dirichlet_bc_dispatch<3>(field, value, order);
}

// Module initialization function for BC bindings
void init_bc_bindings(py::module_& m)
{
    // ============================================================
    // Bind make_dirichlet_bc function for each dimension
    // ============================================================

    // 1D version
    m.def("make_dirichlet_bc",
        &make_dirichlet_bc_1d,
        py::arg("field"),
        py::arg("value"),
        py::arg("order") = 1,
        "Create and attach Dirichlet boundary condition to a 1D scalar field.\n\n"
        "Args:\n"
        "    field: ScalarField1D to apply BC to\n"
        "    value: Constant boundary value\n"
        "    order: Approximation order (1-4, default=1)\n\n"
        "Note:\n"
        "    The BC is attached to the field automatically. No return value."
    );

    // 2D version
    m.def("make_dirichlet_bc",
        &make_dirichlet_bc_2d,
        py::arg("field"),
        py::arg("value"),
        py::arg("order") = 1,
        "Create and attach Dirichlet boundary condition to a 2D scalar field.\n\n"
        "Args:\n"
        "    field: ScalarField2D to apply BC to\n"
        "    value: Constant boundary value\n"
        "    order: Approximation order (1-4, default=1)\n\n"
        "Note:\n"
        "    The BC is attached to the field automatically. No return value."
    );

    // 3D version
    m.def("make_dirichlet_bc",
        &make_dirichlet_bc_3d,
        py::arg("field"),
        py::arg("value"),
        py::arg("order") = 1,
        "Create and attach Dirichlet boundary condition to a 3D scalar field.\n\n"
        "Args:\n"
        "    field: ScalarField3D to apply BC to\n"
        "    value: Constant boundary value\n"
        "    order: Approximation order (1-4, default=1)\n\n"
        "Note:\n"
        "    The BC is attached to the field automatically. No return value."
    );
}
