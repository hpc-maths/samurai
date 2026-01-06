// Samurai Python Bindings - MRA Configuration header
//
// Declares the initialization function for MRA configuration bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize MRA configuration bindings
void init_mra_config_bindings(py::module_& m);
