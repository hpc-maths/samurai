// Samurai Python Bindings - Multiresolution Adaptation
//
// Bindings for make_MRAdapt and update_ghost_mr functions

#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/field.hpp>
#include <samurai/algorithm/update.hpp>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;

// ============================================================
// Type aliases matching field_bindings.cpp pattern
// ============================================================

using default_interval = samurai::Interval<double, std::size_t>;

template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

// ============================================================
// Python-callable wrapper for Adapt objects
// ============================================================

// Base class for type erasure
class PyAdaptBase {
public:
    virtual ~PyAdaptBase() = default;
    virtual void call(samurai::mra_config& config) = 0;
};

// Template derived class for specific dimension
template <class AdaptType>
class PyAdaptImpl : public PyAdaptBase {
public:
    explicit PyAdaptImpl(AdaptType&& adapt) : m_adapt(std::move(adapt)) {}

    void call(samurai::mra_config& config) override {
        m_adapt(config);
    }

private:
    AdaptType m_adapt;
};

// Python-exposed wrapper class
class PyAdapt {
public:
    template <class AdaptType>
    explicit PyAdapt(AdaptType&& adapt)
        : m_impl(std::make_unique<PyAdaptImpl<AdaptType>>(std::move(adapt))) {}

    void operator()(samurai::mra_config& config) {
        m_impl->call(config);
    }

private:
    std::unique_ptr<PyAdaptBase> m_impl;
};

// ============================================================
// Dimension-specific factory functions
// Following pattern from operator_bindings.cpp (upwind_1d, upwind_2d, upwind_3d)
// ============================================================

// 1D update_ghost_mr wrapper
void update_ghost_mr_1d(ScalarField<1>& field)
{
    samurai::update_ghost_mr(field);
}

// 2D update_ghost_mr wrapper
void update_ghost_mr_2d(ScalarField<2>& field)
{
    samurai::update_ghost_mr(field);
}

// 3D update_ghost_mr wrapper
void update_ghost_mr_3d(ScalarField<3>& field)
{
    samurai::update_ghost_mr(field);
}

// 1D make_MRAdapt wrapper
PyAdapt make_mr_adapt_1d(ScalarField<1>& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdapt(std::move(adapt_obj));
}

// 2D make_MRAdapt wrapper
PyAdapt make_mr_adapt_2d(ScalarField<2>& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdapt(std::move(adapt_obj));
}

// 3D make_MRAdapt wrapper
PyAdapt make_mr_adapt_3d(ScalarField<3>& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdapt(std::move(adapt_obj));
}

// ============================================================
// Module initialization
// ============================================================

void init_adapt_bindings(py::module_& m)
{
    // Bind Adapt wrapper class
    py::class_<PyAdapt>(m, "MRAdapt", R"pbdoc(
        Multiresolution mesh adaptation callable.

        Created by make_MRAdapt(), this object performs adaptive mesh refinement
        based on the Harten multiresolution analysis algorithm.

        Examples
        --------
        >>> import samurai_python as sam
        >>> config = sam.MRAConfig()
        >>> config.epsilon = 2e-4
        >>> config.regularity = 2.0
        >>> MRadaptation = sam.make_MRAdapt(field)
        >>> MRadaptation(config)  # Perform adaptation

        Notes
        -----
        Create the adaptation object once and reuse it throughout your simulation.
        The same configuration can also be reused across multiple adaptation calls.
    )pbdoc")
        .def("__call__",
            [](PyAdapt& self, samurai::mra_config& config) {
                self(config);
            },
            py::arg("config"),
            "Perform mesh adaptation with the given configuration."
        );

    // Bind update_ghost_mr for all dimensions
    // Following pattern from operator_bindings.cpp where multiple functions
    // are bound with the same Python name
    m.def("update_ghost_mr",
        &update_ghost_mr_1d,
        py::arg("field"),
        "Update ghost cells for multiresolution analysis (1D)"
    );

    m.def("update_ghost_mr",
        &update_ghost_mr_2d,
        py::arg("field"),
        "Update ghost cells for multiresolution analysis (2D)"
    );

    m.def("update_ghost_mr",
        &update_ghost_mr_3d,
        py::arg("field"),
        "Update ghost cells for multiresolution analysis (3D)"
    );

    // Bind make_MRAdapt for all dimensions
    m.def("make_MRAdapt",
        &make_mr_adapt_1d,
        py::arg("field"),
        "Create multiresolution adaptation object (1D)"
    );

    m.def("make_MRAdapt",
        &make_mr_adapt_2d,
        py::arg("field"),
        "Create multiresolution adaptation object (2D)"
    );

    m.def("make_MRAdapt",
        &make_mr_adapt_3d,
        py::arg("field"),
        "Create multiresolution adaptation object (3D)"
    );
}
