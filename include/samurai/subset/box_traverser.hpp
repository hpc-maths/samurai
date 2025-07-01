// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

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
        using interval_t = Interval<typename B::point_t::value_type>;

        static constexpr std::size_t dim = B::dim;
    };

    template <Box_concept B>
    class BoxTraverser : public SetTraverserBase<BoxTraverser<B>>
    {
        using Self       = BoxTraverser<B>;
        using Base       = SetTraverserBase<Self>;
        using interval_t = typename SetTraverserTraits<Self>::interval_t;
        using value_t    = typename interval_t::value_t;

      public:

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
            m_empty = true;
        }

        inline const interval_t& current_interval() const
        {
            return m_current_interval;
        }

      private:

        interval_t m_current_interval;
        bool m_empty;
    };
}
