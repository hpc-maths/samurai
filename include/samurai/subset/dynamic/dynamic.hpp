// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Runtime (dynamic) counterpart of the static set algebra in `subset/`.
// Use it when the structure of a set expression is only known at runtime, or
// to expose set operations to bindings such as Python.
#include "any_traverser.hpp"
#include "dynamic_set.hpp"
#include "node.hpp"
