// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Helper compiled as its own translation unit: it checks that `times::timers`
// is one registry for the whole program (see test_timers.cpp).

#include <string>

#include <samurai/timers.hpp>

namespace samurai_test
{
    bool time_in_other_translation_unit(const std::string& name)
    {
        samurai::times::timers.start(name);
        samurai::times::timers.stop(name);
        return samurai::times::timers.is_enabled();
    }
} // namespace samurai_test
