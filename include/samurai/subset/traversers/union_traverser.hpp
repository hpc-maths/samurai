// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{

    template <class... SetTraversers>
    class UnionTraverser;

    template <class... SetTraversers>
    struct SetTraverserTraits<UnionTraverser<SetTraversers...>>
    {
        static_assert((IsSetTraverser<SetTraversers>::value and ...));

        using FirstSetTraverser  = std::tuple_element_t<0, std::tuple<SetTraversers...>>;
        using interval_t         = typename FirstSetTraverser::interval_t;
        using current_interval_t = const interval_t&;
    };

    template <class... SetTraversers>
    class UnionTraverser : public SetTraverserBase<UnionTraverser<SetTraversers...>>
    {
        using Self = UnionTraverser<SetTraversers...>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS
        using Childrens = std::tuple<SetTraversers...>;

        static constexpr std::size_t nIntervals = std::tuple_size<Childrens>::value;

        UnionTraverser(const std::array<std::size_t, nIntervals>& shifts, const SetTraversers&... set_traversers)
            : m_set_traversers(set_traversers...)
            , m_shifts(shifts)
        {
            next_interval_impl();
        }

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_current_interval.start == std::numeric_limits<value_t>::max();
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            const auto startFunc = [shifts = m_shifts](const std::size_t i, const value_t start) -> value_t
            {
                return traverser_utils::refine_start(start, shifts[i]);
            };
            const auto endFunc = [shifts = m_shifts](const std::size_t i, const value_t end) -> value_t
            {
                return traverser_utils::refine_end(end, shifts[i]);
            };

            m_current_interval = traverser_utils::transform_and_union(m_set_traversers, startFunc, endFunc);
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        interval_t m_current_interval;
        Childrens m_set_traversers;
        const std::array<std::size_t, nIntervals>& m_shifts;
    };

} // namespace samurai
