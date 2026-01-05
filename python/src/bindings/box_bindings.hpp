// Samurai Python Bindings - Box class header
//
// Declares the initialization function for Box bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize Box class bindings for 1D, 2D, and 3D
void init_box_bindings(py::module_& m);
