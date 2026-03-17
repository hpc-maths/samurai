// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../concepts.hpp"
#include "set_traverser_base.hpp"
#include <iterator>

namespace samurai
{

    template <std::forward_iterator Iterator>
    class RangeTraverser;

    template <std::forward_iterator Iterator>
    struct SetTraverserTraits<RangeTraverser<Iterator>>
    {
        using interval_t         = std::iter_value_t<Iterator>;
        using current_interval_t = const interval_t&;

        static_assert(interval_like<interval_t>);
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

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_current_interval == m_bound_interval;
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            ++m_current_interval;
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return *m_current_interval;
        }

      private:

        Iterator m_current_interval;
        Iterator m_bound_interval;
    };

} // namespace samurai
