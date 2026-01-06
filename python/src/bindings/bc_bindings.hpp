// Samurai Python Bindings - Boundary Conditions header
//
// Declares the initialization function for boundary condition bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize boundary condition bindings (make_bc, make_dirichlet_bc, etc.)
void init_bc_bindings(py::module_& m);
