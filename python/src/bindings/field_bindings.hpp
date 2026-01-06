// Samurai Python Bindings - Field classes header
//
// Declares the initialization function for ScalarField and VectorField bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize ScalarField and VectorField class bindings
void init_field_bindings(py::module_& m);
