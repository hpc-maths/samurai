// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Helper compiled as its own translation unit: it checks that a program can
// include samurai/statistics.hpp from several translation units and that
// `samurai::statistics` is one recorder for the whole program (see
// test_statistics.cpp).

#include <samurai/statistics.hpp>
// Included twice on purpose: the header must be safe to include again.
#include <samurai/statistics.hpp> // NOLINT(readability-duplicate-include)

namespace samurai_test
{
    const samurai::Statistics* statistics_in_other_translation_unit()
    {
        return &samurai::statistics;
    }
} // namespace samurai_test
