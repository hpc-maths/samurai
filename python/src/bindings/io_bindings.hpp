// Samurai Python Bindings - HDF5 I/O header
//
// Declares the initialization function for HDF5 I/O bindings
// including save(), dump(), and load() functions for fields and meshes

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize HDF5 I/O bindings
void init_io_bindings(py::module_& m);
