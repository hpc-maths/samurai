// Samurai Python Bindings - Algorithm functions
//
// Bindings for iteration primitives like for_each_interval

#include <samurai/algorithm.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/interval.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Use the same interval type as in interval_bindings.cpp
using default_interval = samurai::Interval<int, long long int>;

// Helper function to convert xtensor_fixed index to Python tuple
template <std::size_t dim, class IndexArray>
py::tuple convert_index_to_tuple(const IndexArray& index)
{
    if constexpr (dim == 1)
    {
        return py::tuple();
    }
    else if constexpr (dim == 2)
    {
        return py::make_tuple(index[0]);
    }
    else if constexpr (dim == 3)
    {
        return py::make_tuple(index[0], index[1]);
    }
    return py::tuple();
}

// Define exact type aliases matching mesh_bindings.cpp
// Note: Use samurai::mra::make_mesh return type directly
using Config1D = samurai::mesh_config<1>;
using CompleteConfig1D = samurai::complete_mesh_config<Config1D, samurai::MRMeshId>;
using Mesh1D = samurai::MRMesh<CompleteConfig1D>;

using Config2D = samurai::mesh_config<2>;
using CompleteConfig2D = samurai::complete_mesh_config<Config2D, samurai::MRMeshId>;
using Mesh2D = samurai::MRMesh<CompleteConfig2D>;

using Config3D = samurai::mesh_config<3>;
using CompleteConfig3D = samurai::complete_mesh_config<Config3D, samurai::MRMeshId>;
using Mesh3D = samurai::MRMesh<CompleteConfig3D>;

// Verify types are valid at compile time
static_assert(std::is_class_v<Mesh1D>, "Mesh1D must be a class type");
static_assert(std::is_class_v<Mesh2D>, "Mesh2D must be a class type");
static_assert(std::is_class_v<Mesh3D>, "Mesh3D must be a class type");

// Wrapper functions for each dimension
void for_each_interval_1d(const Mesh1D& mesh, py::function func)
{
    samurai::for_each_interval(mesh,
        [&func](std::size_t level, const default_interval& interval, const auto& index) {
            auto index_tuple = convert_index_to_tuple<1>(index);
            func(level, interval, index_tuple);
        }
    );
}

void for_each_interval_2d(const Mesh2D& mesh, py::function func)
{
    samurai::for_each_interval(mesh,
        [&func](std::size_t level, const default_interval& interval, const auto& index) {
            auto index_tuple = convert_index_to_tuple<2>(index);
            func(level, interval, index_tuple);
        }
    );
}

void for_each_interval_3d(const Mesh3D& mesh, py::function func)
{
    samurai::for_each_interval(mesh,
        [&func](std::size_t level, const default_interval& interval, const auto& index) {
            auto index_tuple = convert_index_to_tuple<3>(index);
            func(level, interval, index_tuple);
        }
    );
}

// Module initialization function for algorithm bindings
void init_algorithm_bindings(py::module_& m)
{
    // Bind 1D overload
    m.def("for_each_interval", &for_each_interval_1d,
        py::arg("mesh"),
        py::arg("function"),
        "Iterate over all intervals in the 1D mesh."
    );

    // Bind 2D overload
    m.def("for_each_interval", &for_each_interval_2d,
        py::arg("mesh"),
        py::arg("function"),
        "Iterate over all intervals in the 2D mesh."
    );

    // Bind 3D overload
    m.def("for_each_interval", &for_each_interval_3d,
        py::arg("mesh"),
        py::arg("function"),
        "Iterate over all intervals in the 3D mesh."
    );
}
