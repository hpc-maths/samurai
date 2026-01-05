// Samurai Python Bindings - Main Module
//
// This file serves as the entry point for the Python bindings.
// Bindings will be added progressively following the phased approach:
// - Phase 1: Core types (Box, Mesh, Field, Cell)
// - Phase 2: Algorithms (for_each_cell, adapt, BC)
// - Phase 3: Operators (diffusion, upwind, etc.)
// - Phase 4: I/O (HDF5 save/load)

#include <pybind11/pybind11.h>

// Samurai includes (will be added progressively as bindings are implemented)
// #include <samurai/box.hpp>
// #include <samurai/mesh.hpp>
// #include <samurai/field.hpp>
// #include <samurai/algorithm.hpp>

namespace py = pybind11;

// Version information (will be read from version.txt in production)
#define SAMURAI_PYTHON_VERSION "0.28.0-dev"

PYBIND11_MODULE(samurai_python, m) {
    // Module documentation
    m.doc() = R"pbdoc(
        Samurai Python Bindings
        -----------------------

        Adaptive Mesh Refinement (AMR) and Multiresolution Analysis library

        .. currentmodule:: samurai_python

        .. autosummary::
           :toctree: _generate

           Box
           Mesh
           Field
    )pbdoc";

    // Version attribute
    m.attr("__version__") = SAMURAI_PYTHON_VERSION;

    // TODO: Add submodule initializers as they are implemented
    // init_core(m);
    // init_algorithms(m);
    // init_operators(m);
    // init_io(m);

    // Placeholder: Basic test function
    m.def("test_function", []() {
        return "Samurai Python bindings are working!";
    }, R"pbdoc(
        Test function to verify bindings are loaded correctly.

        Returns:
            str: Success message
    )pbdoc");

    // Python module metadata
    #ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = SAMURAI_PYTHON_VERSION;
    #endif
}
