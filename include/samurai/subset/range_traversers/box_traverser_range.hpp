// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../interval.hpp"
#include "../traversers/box_traverser.hpp"
#include "set_traverser_range_base.hpp"

namespace samurai
{
    template <class value_t, std::size_t dim_>
    class Box;

    template <class B>
    class BoxTraverserRange;

    template <class B>
    struct SetTraverserRangeTraits<BoxTraverserRange<B>>
    {
        static_assert(std::same_as<Box<typename B::point_t::value_type, B::dim>, B>);

        class Iterator
        {
          public:

            using index_t = typename interval_t::value_t;

            using iterator_category = std::forward_iterator_tag;
            using difference_type   = index_t;
            using value_type        = BoxTraverser<B>;
            using reference         = BoxTraverser<B>;

            Iterator(const interval_t& interval, const index_t& index)
                : m_interval(interval)
                , m_index(first_index)
            {
            }

            reference operator*() const
            {
                return reference(interval);
            }

            Iterator& operator++()
            {
                ++m_index;
                return *this;
            }

            Iterator operator++(int)
            {
                Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const Iterator& a, const Iterator& b)
            {
                return a.m_index == b.m_index;
            };

            friend bool operator!=(const Iterator& a, const Iterator& b)
            {
                return a.m_index != b.m_index;
            };

          private:

            interval_t m_interval;
            index_t m_index;
        };
    };

    template <class B>
    class BoxTraverserRange : public SetTraverserRangeBase<BoxTraverserRange<B>>
    {
        using Self = BoxTraverserRange<B>;

      public:

        SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS

        using interval_t = Interval<typename B::point_t::value_type>;
        using index_t    = typename interval_t::value_t;

        BoxTraverserRange(const interval_t& interval, const index_t& first_index, const index_t& last_index)
            : m_interval(interval)
            , m_first_index(first_index)
            , m_last_index(last_index)
        {
        }

        Iterator begin_impl()
        {
            return Iterator(m_interval, m_first_index);
        }

        Iterator end_impl()
        {
            return Iterator(m_interval, m_last_index);
        }

      private:

        interval_t m_interval;
        index_t m_first_index;
        index_t m_last_index;
    };
} // namespace samurai
