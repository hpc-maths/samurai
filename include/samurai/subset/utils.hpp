// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{

    template <typename T>
    constexpr auto compute_min(const T& first_value, const T&... other_values)
    {
        if constexpr (sizeof...(other_values) == 0u) // Single argument case!
        {
            return first_value;
        }
        else // For the Ts...
        {
            return std::min(first_value, compute_min(other_values...);
        }
    }

    template <typename T>
    constexpr auto compute_max(const T& first_value, const T&... other_values)
    {
        if constexpr (sizeof...(other_values) == 0u) // Single argument case!
        {
            return first_value;
        }
        else // For the Ts...
        {
            return std::max(first_value, compute_max(other_values...);
        }
    }
}
