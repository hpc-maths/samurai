// Samurai Python Bindings - Interval class header
//
// Declares the initialization function for samurai::Interval bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize Interval class bindings
void init_interval_bindings(py::module_& m);
