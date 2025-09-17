// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../box.hpp"
#include "../../interval.hpp"
#include "set_traverser_base.hpp"
#include <concepts>

namespace samurai
{
    template <class B>
    class BoxTraverser;

    template <class B>
    struct SetTraverserTraits<BoxTraverser<B>>
    {
        static_assert(std::same_as<Box<typename B::point_t::value_type, B::dim>, B>);

        using interval_t         = Interval<typename B::point_t::value_type>;
        using current_interval_t = const interval_t&;
    };

    template <class B>
    class BoxTraverser : public SetTraverserBase<BoxTraverser<B>>
    {
        using Self = BoxTraverser<B>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        BoxTraverser(const value_t& start, const value_t& end)
            : m_current_interval{start, end}
            , m_empty(false)
        {
        }

        inline bool is_empty_impl() const
        {
            return m_empty;
        }

        inline void next_interval_impl()
        {
            m_empty = true;
        }

        inline current_interval_t current_interval_impl() const
        {
            return m_current_interval;
        }

      private:

        interval_t m_current_interval;
        bool m_empty;
    };

} // namespace samurai
