// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * @file pch.hpp
 * @brief Precompiled header for Samurai library
 *
 * This header includes the most frequently used Samurai components
 * to accelerate compilation when PCH is enabled. Speeds up header
 * parsing across all compilation units using the library.
 *
 * Enable via CMake: -DSAMURAI_ENABLE_PCH=ON
 */

// Core Samurai headers (alphabetical order)
#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/interval.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

// I/O headers
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>

// External dependencies that are heavy
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/io/xio.hpp>

// Standard library (frequently used)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
