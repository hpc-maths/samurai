// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include "set_traverser_base.hpp"

#pragma once

namespace samurai
{
    template <typename T>
    concept LCA_concept = std::same_as<LevelCellArray<T::dim, typename T::interval_t>, T>;

    template <LCA_concept LCA>
    class LCATraverser;

    template <LCA_concept LCA>
    struct SetTraverserTraits<LCATraverser<LCA>>
    {
        using interval_t = typename LCA::interval_t;

        static constexpr std::size_t dim = LCA::dim;
    };

    template <LCA_concept LCA>
    class LCATraverser : public SetTraverserBase<LCATraverser<LCA>>
    {
        using Self              = LCATraverser<LCA>;
        using Base              = SetTraverserBase<Self>;
        using interval_t        = typename SetTraverserTraits<Self>::interval_t;
        using interval_iterator = typename std::vector<interval_t>::const_iterator;

      public:

        LCATraverser(const interval_iterator first, const interval_iterator end)
            : m_first_interval(first)
            , m_end_interval(end)
        {
        }

        inline bool is_empty() const
        {
            return m_first_interval == m_end_interval;
        }

        inline void next_interval()
        {
            ++m_first_interval;
        }

        inline const interval_t& current_interval() const
        {
            return *m_first_interval;
        }

      private:

        interval_iterator m_first_interval;
        interval_iterator m_end_interval;
    };
}
