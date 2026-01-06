// Samurai Python Bindings - Adaptation header
//
// Declares the initialization function for MR adaptation bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize MR adaptation bindings
void init_adapt_bindings(py::module_& m);
