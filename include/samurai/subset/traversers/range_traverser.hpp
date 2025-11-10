// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../interval.hpp"
#include "set_traverser_base.hpp"

namespace samurai
{

    template <std::forward_iterator Iterator>
    class RangeTraverser;

    template <std::forward_iterator Iterator>
    struct SetTraverserTraits<RangeTraverser<Iterator>>
    {
        using interval_t         = typename Iterator::value_type;
        using current_interval_t = const interval_t&;

        static_assert(IsInterval<interval_t>::value);
    };

    template <std::forward_iterator Iterator>
    class RangeTraverser : public SetTraverserBase<RangeTraverser<Iterator>>
    {
        using Self = RangeTraverser<Iterator>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        RangeTraverser(const Iterator current_interval, const Iterator bound_interval)
            : m_current_interval(current_interval)
            , m_bound_interval(bound_interval)
        {
        }

        inline bool is_empty_impl() const
        {
            return m_current_interval == m_bound_interval;
        }

        inline void next_interval_impl()
        {
            ++m_current_interval;
        }

        inline current_interval_t current_interval_impl() const
        {
            return *m_current_interval;
        }

      private:

        Iterator m_current_interval;
        Iterator m_bound_interval;
    };

} // namespace samurai
