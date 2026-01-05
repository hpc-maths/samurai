// Samurai Python Bindings - MRMesh class header
//
// Declares the initialization function for MRMesh bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize MRMesh class bindings for 1D, 2D, and 3D
void init_mesh_bindings(py::module_& m);
