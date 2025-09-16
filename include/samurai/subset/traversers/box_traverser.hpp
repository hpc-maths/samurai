// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include "../../box.hpp"
#include "../../interval.hpp"
#include "set_traverser_base.hpp"

#pragma once

namespace samurai
{
    template <typename T>
    concept Box_concept = std::same_as<Box<typename T::point_t::value_type, T::dim>, T>;

    template <Box_concept B>
    class BoxTraverser;

    template <Box_concept B>
    struct SetTraverserTraits<BoxTraverser<B>>
    {
        using interval_t         = Interval<typename B::point_t::value_type>;
        using current_interval_t = const interval_t&;
    };

    template <Box_concept B>
    class BoxTraverser : public SetTraverserBase<BoxTraverser<B>>
    {
        using Self = BoxTraverser<B>;
        using Base = SetTraverserBase<Self>;

      public:

        using interval_t         = typename Base::interval_t;
        using current_interval_t = typename Base::current_interval_t;
        using value_t            = typename Base::value_t;

        BoxTraverser(const value_t& start, const value_t& end)
            : m_current_interval{start, end}
            , m_empty(false)
        {
        }

        inline bool is_empty() const
        {
            return m_empty;
        }

        inline void next_interval()
        {
            assert(!is_empty());
            m_empty = true;
        }

        inline current_interval_t current_interval() const
        {
            return m_current_interval;
        }

      private:

        interval_t m_current_interval;
        bool m_empty;
    };
}
