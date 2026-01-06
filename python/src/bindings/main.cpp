// Samurai Python Bindings - Main Module
//
// This file serves as the entry point for the Python bindings.
// Bindings will be added progressively following the phased approach:
// - Phase 1: Core types (Box, Mesh, Field, Cell)
// - Phase 2: Algorithms (for_each_cell, adapt, BC)
// - Phase 3: Operators (diffusion, upwind, etc.)
// - Phase 4: I/O (HDF5 save/load)

#include <pybind11/pybind11.h>

// Binding initialization headers
#include "box_bindings.hpp"
#include "mesh_config_bindings.hpp"
#include "mesh_bindings.hpp"
#include "field_bindings.hpp"
#include "interval_bindings.hpp"
#include "algorithm_bindings.hpp"
#include "operator_bindings.hpp"
#include "bc_bindings.hpp"
#include "mra_config_bindings.hpp"
#include "adapt_bindings.hpp"
#include "io_bindings.hpp"

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

           Box1D
           Box2D
           Box3D
           MeshConfig1D
           MeshConfig2D
           MeshConfig3D
           MRMesh1D
           MRMesh2D
           MRMesh3D
           ScalarField1D
           ScalarField2D
           ScalarField3D
           VectorField1D_2
           VectorField1D_3
           VectorField2D_2
           VectorField2D_3
           VectorField3D_2
           VectorField3D_3
           Interval
           MRAConfig
           MRAdapt
           make_MRAdapt
           update_ghost_mr
           upwind
           save
           dump
           load
    )pbdoc";

    // Version attribute
    m.attr("__version__") = SAMURAI_PYTHON_VERSION;

    // Initialize bindings
    init_box_bindings(m);
    init_mesh_config_bindings(m);
    init_mesh_bindings(m);
    init_field_bindings(m);
    init_interval_bindings(m);
    init_algorithm_bindings(m);
    init_operator_bindings(m);
    init_bc_bindings(m);
    init_mra_config_bindings(m);
    init_adapt_bindings(m);
    init_io_bindings(m);

    // TODO: Add more submodule initializers as they are implemented
    // init_fv_bindings(m);  // Finite volume schemes
    // init_lbm_bindings(m); // Lattice Boltzmann methods

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
