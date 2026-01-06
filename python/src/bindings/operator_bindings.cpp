// Samurai Python Bindings - Operator functions
//
// Bindings for finite volume operators like upwind

#include <samurai/stencil_field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/field.hpp>
#include <samurai/algorithm.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Type aliases matching mesh_bindings.cpp
using default_interval = samurai::Interval<int, long long int>;

using Config1D = samurai::mesh_config<1>;
using CompleteConfig1D = samurai::complete_mesh_config<Config1D, samurai::MRMeshId>;
using Mesh1D = samurai::MRMesh<CompleteConfig1D>;

using Config2D = samurai::mesh_config<2>;
using CompleteConfig2D = samurai::complete_mesh_config<Config2D, samurai::MRMeshId>;
using Mesh2D = samurai::MRMesh<CompleteConfig2D>;

using Config3D = samurai::mesh_config<3>;
using CompleteConfig3D = samurai::complete_mesh_config<Config3D, samurai::MRMeshId>;
using Mesh3D = samurai::MRMesh<CompleteConfig3D>;

// Field type aliases
template <std::size_t dim>
using ScalarField = samurai::ScalarField<samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>, double>;

// 1D upwind operator - immediate evaluation version
py::object upwind_1d(double velocity, ScalarField<1>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
        [&result, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
        {
            result(level, interval, index) = upwind_expr(level, interval, index);
        }
    );

    return py::cast(result);
}

// 2D upwind operator - immediate evaluation version
py::object upwind_2d(const std::array<double, 2>& velocity, ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
        [&result, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
        {
            result(level, interval, index) = upwind_expr(level, interval, index);
        }
    );

    return py::cast(result);
}

// 3D upwind operator - immediate evaluation version
py::object upwind_3d(const std::array<double, 3>& velocity, ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
        [&result, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
        {
            result(level, interval, index) = upwind_expr(level, interval, index);
        }
    );

    return py::cast(result);
}

// Convenience wrapper accepting Python list/tuple for velocity (2D)
py::object upwind_2d_py(py::sequence velocity_seq, ScalarField<2>& field)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return upwind_2d(velocity, field);
}

// Convenience wrapper accepting Python list/tuple for velocity (3D)
py::object upwind_3d_py(py::sequence velocity_seq, ScalarField<3>& field)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return upwind_3d(velocity, field);
}

// Module initialization function for operator bindings
void init_operator_bindings(py::module_& m)
{
    // Bind 1D upwind operator
    m.def("upwind",
        &upwind_1d,
        py::arg("velocity"),
        py::arg("field"),
        R"pbdoc(
        Upwind operator for 1D advection.

        Computes the upwind flux for a scalar field in 1D.

        Parameters
        ----------
        velocity : float
            Advection velocity (scalar for 1D)
        field : ScalarField1D
            Input scalar field

        Returns
        -------
        ScalarField1D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.upwind(1.0, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc"
    );

    // Bind 2D upwind operator - std::array version
    m.def("upwind",
        &upwind_2d,
        py::arg("velocity"),
        py::arg("field"),
        R"pbdoc(
        Upwind operator for 2D advection (std::array version).

        Parameters
        ----------
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing upwind flux values
        )pbdoc"
    );

    // Bind 2D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
        &upwind_2d_py,
        py::arg("velocity"),
        py::arg("field"),
        R"pbdoc(
        Upwind operator for 2D advection.

        Computes the upwind flux for a scalar field in 2D.

        Parameters
        ----------
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> velocity = [1.0, 1.0]  # [vx, vy]
        >>> flux = sam.upwind(velocity, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc"
    );

    // Bind 3D upwind operator - std::array version
    m.def("upwind",
        &upwind_3d,
        py::arg("velocity"),
        py::arg("field"),
        R"pbdoc(
        Upwind operator for 3D advection (std::array version).

        Parameters
        ----------
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing upwind flux values
        )pbdoc"
    );

    // Bind 3D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
        &upwind_3d_py,
        py::arg("velocity"),
        py::arg("field"),
        R"pbdoc(
        Upwind operator for 3D advection.

        Computes the upwind flux for a scalar field in 3D.

        Parameters
        ----------
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> velocity = [1.0, 1.0, 0.0]  # [vx, vy, vz]
        >>> flux = sam.upwind(velocity, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc"
    );
}
