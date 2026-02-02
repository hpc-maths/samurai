// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>

#include "concepts.hpp"

namespace samurai
{
    template <class Field1, class Field2>
        requires(field_like<Field1> && field_like<Field2>)
    SAMURAI_INLINE void swap(Field1& u1, Field2& u2)
    {
        std::swap(u1.array(), u2.array());

        std::swap(u1.ghosts_updated(), u2.ghosts_updated());
    }
}
