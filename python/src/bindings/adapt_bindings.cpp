// Samurai Python Bindings - Multiresolution Adaptation
//
// Bindings for make_MRAdapt and update_ghost_mr functions

#include <samurai/mr/adapt.hpp>
#include <samurai/algorithm/update.hpp>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;

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

    // Note: The actual make_MRAdapt and update_ghost_mr bindings are complex
    // due to template type resolution issues with pybind11.
    // They will need to be implemented using a different approach,
    // possibly by exposing them through the ScalarField classes directly.
    // For now, these are placeholder functions.

    (void)m; // Suppress unused warning
}
