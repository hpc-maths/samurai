// Samurai Python Bindings - MeshConfig class header
//
// Declares the initialization function for MeshConfig bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize MeshConfig class bindings for 1D, 2D, and 3D
void init_mesh_config_bindings(py::module_& m);
