// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"
#include <concepts>

namespace samurai
{
    template <std::size_t Dim, class TInterval>
    class LevelCellArray;

    template <class LCA>
    class LCATraverser;

    template <class LCA>
    struct SetTraverserTraits<LCATraverser<LCA>>
    {
        static_assert(std::same_as<LevelCellArray<LCA::dim, typename LCA::interval_t>, LCA>);

        using interval_t         = typename LCA::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class LCA>
    class LCATraverser : public SetTraverserBase<LCATraverser<LCA>>
    {
        using Self = LCATraverser<LCA>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS
        using interval_iterator = typename std::vector<interval_t>::const_iterator;

        LCATraverser(const interval_iterator first, const interval_iterator end)
            : m_first_interval(first)
            , m_end_interval(end)
        {
        }

        inline bool is_empty_impl() const
        {
            return m_first_interval == m_end_interval;
        }

        inline void next_interval_impl()
        {
            ++m_first_interval;
        }

        inline current_interval_t current_interval_impl() const
        {
            return *m_first_interval;
        }

      private:

        interval_iterator m_first_interval;
        interval_iterator m_end_interval;
    };

} // namespace samurai
