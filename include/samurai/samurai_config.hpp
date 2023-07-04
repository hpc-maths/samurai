// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <cstddef>
#include <xtl/xmeta_utils.hpp>

// NOLINTBEGIN(cppcoreguidelines-macro-usage,modernize-macro-to-enum)
#define SAMURAI_VERSION_MAJOR 0
#define SAMURAI_VERSION_MINOR 1
#define SAMURAI_VERSION_PATCH 0

// NOLINTEND(cppcoreguidelines-macro-usage,modernize-macro-to-enum)

namespace samurai
{
    template <class TValue, class TIndex>
    struct Interval;

    namespace default_config
    {
        static constexpr std::size_t max_level        = 20;
        static constexpr std::size_t ghost_width      = 1;
        static constexpr std::size_t graduation_width = 1;
        static constexpr std::size_t prediction_order = 1;

        using index_t    = signed long long int;
        using value_t    = int;
        using interval_t = Interval<value_t, index_t>;
    }

    template <class Field>
    struct Dirichlet;

    template <class Field>
    struct Neumann;

    struct bc_types
    {
        template <class Field>
        using types = xtl::mpl::vector<Dirichlet<Field>, Neumann<Field>>;
    };

#ifndef BC_TYPES
#define BC_TYPES bc_types
#endif

    template <class Field>
    class Bc;

#define ADD_BC(name)                                   \
    template <class Field>                             \
    struct name : public samurai::Bc<Field>            \
    {                                                  \
        using base_t = samurai::Bc<Field>;             \
        using samurai::Bc<Field>::Bc;                  \
                                                       \
        std::unique_ptr<base_t> clone() const override \
        {                                              \
            return std::make_unique<name>(*this);      \
        }                                              \
    };

}
