// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

#include "../traversers/set_traverser_base.hpp"

namespace samurai
{

    template <std::size_t Dim, class TInterval>
    class LevelCellArray;

    template <class LCA>
    class LCATraverserRangeItem;

    template <class LCA>
    struct SetTraverserTraits<LCATraverserRangeItem<LCA>>
    {
        static_assert(std::same_as<LevelCellArray<LCA::dim, typename LCA::interval_t>, LCA>);

        using interval_t         = typename LCA::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class LCA>
    class LCATraverserRangeItem : public SetTraverserBase<LCATraverserRangeItem<LCA>>
    {
        using Self = LCATraverserRangeItem<LCA>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        using interval_iterator = typename std::vector<interval_t>::const_iterator;
        using offset_iterator   = typename std::vector<std::ptrdiff_t>::iterator;

        LCATraverserRangeItem(const interval_iterator first_interval, const offset_iterator offset)
            : m_first_interval(first_interval)
            , m_offset(*offset)
            , m_offset_bound(*(offset + 1))
        {
        }

        inline bool is_empty_impl() const
        {
            return m_offset == m_offset_bound;
        }

        inline void next_interval_impl()
        {
            ++m_offset;
        }

        inline current_interval_t current_interval_impl() const
        {
            return *(m_first_interval + m_offset);
        }

      private:

        interval_iterator m_first_interval;
        std::ptrdiff_t& m_offset;
        std::ptrdiff_t m_offset_bound;
    };

} // namespace samurai
