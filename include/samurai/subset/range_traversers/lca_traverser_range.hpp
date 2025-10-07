// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

#include "lca_traverser_range_item.hpp"
#include "set_traverser_range_base.hpp"

namespace samurai
{

    template <std::size_t Dim, class TInterval>
    class LevelCellArray;

    template <class LCA>
    class LCATraverserRange;

    template <class LCA>
    struct SetTraverserRangeTraits<LCATraverserRange<LCA>>
    {
        static_assert(std::same_as<LevelCellArray<LCA::dim, typename LCA::interval_t>, LCA>);

        using interval_iterator = typename std::vector<interval_t>::const_iterator;
        using offset_iterator   = typename std::vector<std::size_t>::iterator;

        template <bool isConst>
        class _Iterator
        {
          public:

            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = LCATraverserRangeItem<isConst, LCA>;
            using reference         = LCATraverserRangeItem<isConst, LCA>;

            _Iterator(const interval_iterator first_interval, const offset_iterator offset)
                : m_first_interval(first_interval)
                , m_offset(offset)
            {
            }

            reference operator*() const
            {
                return reference(m_first_interval, m_offset);
            }

            _Iterator& operator++()
            {
                ++m_offset;
                return *this;
            }

            _Iterator operator++(int)
            {
                _Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const _Iterator& a, const _Iterator& b)
            {
                return a.m_first_interval == b.m_first_interval and a.m_offset == b.m_offset;
            };

            friend bool operator!=(const _Iterator& a, const _Iterator& b)
            {
                return a.m_ptr != b.m_ptr or a.m_offset != b.m_offset;
            };

          private:

            interval_iterator m_first_interval;
            offset_iterator m_offset;
        };

        using Iterator       = _Iterator<false>;
        using const_Iterator = _Iterator<true>;
    };

    template <class LCA>
    class LCATraverserRange : public SetTraverserRangeBase<LCATraverserRange<LCA>>
    {
        using Self = LCATraverserRange<LCA>;

      public:

        SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS
        using interval_iterator = typename std::vector<interval_t>::const_iterator;
        using offset_iterator   = typename std::vector<std::size_t>::iterator;

        LCATraverserRange(const interval_iterator first_interval, const offset_iterator first_offsets, const offset_iterator last_offsets)
            : m_first_interval(first_interval)
            , m_first_offsets(first_offsets)
            , m_last_offsets(last_offsets)
        {
        }

        Iterator begin_impl()
        {
            return Iterator(m_first_interval, m_first_offsets);
        }

        Iterator end_impl()
        {
            return Iterator(m_first_interval, std::prev(m_last_offsets));
        }

        const_Iterator begin_impl() const
        {
            return const_Iterator(m_first_interval, m_first_offsets);
        }

        const_Iterator end_impl() const
        {
            return const_Iterator(m_first_interval, std::prev(m_last_offsets));
        }

      private:

        interval_iterator m_first_interval;
        offset_iterator m_first_offsets;
        offset_iterator m_last_offsets;
    };

} // namespace samurai
