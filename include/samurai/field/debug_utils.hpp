// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <iostream>
#include <string_view>

#ifdef SAMURAI_CHECK_NAN
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xoperation.hpp>
#endif

namespace samurai::detail
{
    /**
     * @brief Check for NaN values in field data (debug utility)
     *
     * This function is only active when SAMURAI_CHECK_NAN is defined.
     * It logs detailed information about where NaNs are detected.
     *
     * @tparam Data Data container type
     * @param data The data to check
     * @param context Description of where the check occurred
     */
    template <class Data>
    SAMURAI_INLINE void check_nan([[maybe_unused]] const Data& data, [[maybe_unused]] std::string_view context)
    {
#ifdef SAMURAI_CHECK_NAN
        if (xt::any(xt::isnan(data)))
        {
            std::cerr << "⚠ NaN detected in " << context << "\n";
            // Extended logging could be added here if needed
        }
#endif
    }

} // namespace samurai::detail
