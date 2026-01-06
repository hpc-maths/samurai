// Samurai Python Bindings - Operator bindings header
//
// Declares the initialization function for operator bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize operator bindings (upwind, etc.)
void init_operator_bindings(py::module_& m);
