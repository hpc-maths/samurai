// Samurai Python Bindings - Algorithm functions
//
// Bindings for iteration primitives like for_each_interval and for_each_cell

#include <samurai/algorithm.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/interval.hpp>
#include <samurai/cell.hpp>
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

// ============================================================
// for_each_cell bindings
// ============================================================

// Cell type definition - use same interval type as for_each_interval
// The mesh uses Interval<int, long long int> internally
using cell_interval = samurai::Interval<int, long long int>;

template <std::size_t dim>
using Cell = samurai::Cell<dim, cell_interval>;

// CellWrapper: Lightweight wrapper for exposing Cell to Python
// Stores a copy of Cell data to avoid lifetime issues
template <std::size_t dim>
struct CellWrapper
{
    std::size_t level;
    std::size_t index;  // Linear index for field indexing
    double length;
    xt::xtensor_fixed<double, xt::xshape<dim>> center;
    xt::xtensor_fixed<double, xt::xshape<dim>> corner;

    // Constructor from C++ Cell
    explicit CellWrapper(const Cell<dim>& cell)
        : level(cell.level)
        , index(static_cast<std::size_t>(cell.index))
        , length(cell.length)
        , center(cell.center())
        , corner(cell.corner())
    {}
};

// Helper to bind CellWrapper class for a specific dimension
template <std::size_t dim>
void bind_cell_wrapper(py::module_& m, const std::string& name)
{
    using Wrapper = CellWrapper<dim>;

    py::class_<Wrapper>(m, name.c_str(), R"pbdoc(Cell wrapper for for_each_cell iteration.)pbdoc")
        .def_property_readonly("level",
            [](const Wrapper& w) { return w.level; },
            "Refinement level of the cell"
        )
        .def_property_readonly("index",
            [](const Wrapper& w) { return w.index; },
            "Linear index in field data array (for field[index] access)"
        )
        .def_property_readonly("length",
            [](const Wrapper& w) { return w.length; },
            "Physical size of the cell"
        )
        .def("center",
            [](const Wrapper& w) -> py::tuple {
                if constexpr (dim == 1) {
                    return py::make_tuple(w.center[0]);
                } else if constexpr (dim == 2) {
                    return py::make_tuple(w.center[0], w.center[1]);
                } else if constexpr (dim == 3) {
                    return py::make_tuple(w.center[0], w.center[1], w.center[2]);
                }
                return py::tuple();
            },
            "Returns cell center as (x, y, z) tuple"
        )
        .def("corner",
            [](const Wrapper& w) -> py::tuple {
                if constexpr (dim == 1) {
                    return py::make_tuple(w.corner[0]);
                } else if constexpr (dim == 2) {
                    return py::make_tuple(w.corner[0], w.corner[1]);
                } else if constexpr (dim == 3) {
                    return py::make_tuple(w.corner[0], w.corner[1], w.corner[2]);
                }
                return py::tuple();
            },
            "Returns cell corner (min point) as (x, y, z) tuple"
        )
        .def("__repr__",
            [name](const Wrapper& w) {
                std::ostringstream oss;
                oss << name << "(level=" << w.level << ", index=" << w.index << ")";
                return oss.str();
            }
        );
}

// Wrapper functions for for_each_cell for each dimension
void for_each_cell_1d(const Mesh1D& mesh, py::function func)
{
    samurai::for_each_cell(mesh,
        [&func](const auto& cell) {
            CellWrapper<1> wrapper(cell);
            func(wrapper);
        }
    );
}

void for_each_cell_2d(const Mesh2D& mesh, py::function func)
{
    samurai::for_each_cell(mesh,
        [&func](const auto& cell) {
            CellWrapper<2> wrapper(cell);
            func(wrapper);
        }
    );
}

void for_each_cell_3d(const Mesh3D& mesh, py::function func)
{
    samurai::for_each_cell(mesh,
        [&func](const auto& cell) {
            CellWrapper<3> wrapper(cell);
            func(wrapper);
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

    // ============================================================
    // Bind CellWrapper classes for for_each_cell
    // ============================================================
    bind_cell_wrapper<1>(m, "Cell1D");
    bind_cell_wrapper<2>(m, "Cell2D");
    bind_cell_wrapper<3>(m, "Cell3D");

    // ============================================================
    // Bind for_each_cell functions
    // ============================================================
    // Bind 1D overload
    m.def("for_each_cell", &for_each_cell_1d,
        py::arg("mesh"),
        py::arg("function"),
        "Iterate over all cells in the 1D mesh."
    );

    // Bind 2D overload
    m.def("for_each_cell", &for_each_cell_2d,
        py::arg("mesh"),
        py::arg("function"),
        "Iterate over all cells in the 2D mesh."
    );

    // Bind 3D overload
    m.def("for_each_cell", &for_each_cell_3d,
        py::arg("mesh"),
        py::arg("function"),
        "Iterate over all cells in the 3D mesh."
    );
}
