// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <forward_list>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include "interval.hpp"
#include "samurai_config.hpp"

namespace samurai
{
    namespace detail
    {
        template <class InputIt, class UnaryPredicate>
        SAMURAI_INLINE constexpr std::pair<InputIt, InputIt> forward_find_if(InputIt first, InputIt last, UnaryPredicate p)
        {
            auto previous = first++;
            for (; first != last; ++first, ++previous)
            {
                if (p(*first))
                {
                    break;
                }
            }
            return {previous, first};
        }
    } // namespace detail

    ////////////////////////////////
    // ListOfIntervals definition //
    ////////////////////////////////

    /** @class ListOfIntervals
     *  @brief Forward list of intervals.
     *
     * The intervals are kept sorted and disjoint. An iterator on the last
     * interval touched is cached and used as a starting hint for the next
     * insertion. Intervals are almost always added in increasing order, so this
     * turns the insertion into an O(1) operation instead of a traversal of the
     * whole list from its beginning. The hint is only an optimization: it never
     * changes the content of the list.
     *
     * @tparam TValue  The coordinate type (must be signed).
     * @tparam TIndex  The index type (must be signed).
     */
    template <typename TValue, typename TIndex = default_config::index_t>
    struct ListOfIntervals : private std::forward_list<Interval<TValue, TIndex>>
    {
        using value_t    = TValue;
        using index_t    = TIndex;
        using interval_t = Interval<value_t, index_t>;

        using list_t = std::forward_list<interval_t>;
        using list_t::before_begin;
        using list_t::begin;
        using list_t::cbegin;
        using list_t::cend;
        using list_t::empty;
        using list_t::end;

        using const_iterator = typename list_t::const_iterator;
        using iterator       = typename list_t::iterator;
        using value_type     = typename list_t::value_type;

        ListOfIntervals();
        ListOfIntervals(const ListOfIntervals& other);
        ListOfIntervals& operator=(const ListOfIntervals& other);
        ListOfIntervals(ListOfIntervals&& other) noexcept;
        ListOfIntervals& operator=(ListOfIntervals&& other) noexcept;
        ~ListOfIntervals() = default;

        std::size_t size() const;

        void clear();
        void add_point(value_t point);
        void add_interval(const interval_t& interval);

      private:

        /// Iterator on the last interval touched, or before_begin() if there is none.
        iterator m_hint;

        /// Merge into *it all the following intervals it now overlaps.
        void merge_following(iterator it, value_t interval_end);
    };

    ////////////////////////////////////
    // ListOfIntervals implementation //
    ////////////////////////////////////

    template <typename TValue, typename TIndex>
    SAMURAI_INLINE ListOfIntervals<TValue, TIndex>::ListOfIntervals()
        : m_hint{before_begin()}
    {
    }

    // The hint is tied to the nodes of a given container, so it is never taken
    // over on copy nor on move. Dropping it is harmless: it only costs one full
    // traversal on the next insertion.

    template <typename TValue, typename TIndex>
    SAMURAI_INLINE ListOfIntervals<TValue, TIndex>::ListOfIntervals(const ListOfIntervals& other)
        : list_t{other}
        , m_hint{before_begin()}
    {
    }

    template <typename TValue, typename TIndex>
    SAMURAI_INLINE auto ListOfIntervals<TValue, TIndex>::operator=(const ListOfIntervals& other) -> ListOfIntervals&
    {
        if (this != &other)
        {
            list_t::operator=(other);
            m_hint = before_begin();
        }
        return *this;
    }

    template <typename TValue, typename TIndex>
    SAMURAI_INLINE ListOfIntervals<TValue, TIndex>::ListOfIntervals(ListOfIntervals&& other) noexcept
        : list_t{std::move(static_cast<list_t&>(other))}
        , m_hint{before_begin()}
    {
        other.m_hint = other.before_begin();
    }

    template <typename TValue, typename TIndex>
    SAMURAI_INLINE auto ListOfIntervals<TValue, TIndex>::operator=(ListOfIntervals&& other) noexcept -> ListOfIntervals&
    {
        if (this != &other)
        {
            list_t::operator=(std::move(static_cast<list_t&>(other)));
            m_hint       = before_begin();
            other.m_hint = other.before_begin();
        }
        return *this;
    }

    /// Number of intervals stored in the list.
    template <typename TValue, typename TIndex>
    SAMURAI_INLINE std::size_t ListOfIntervals<TValue, TIndex>::size() const
    {
        return static_cast<std::size_t>(std::distance(begin(), end()));
    }

    /// Remove all the intervals.
    template <typename TValue, typename TIndex>
    SAMURAI_INLINE void ListOfIntervals<TValue, TIndex>::clear()
    {
        list_t::clear();
        m_hint = before_begin();
    }

    /// Add a point inside the list.
    template <typename TValue, typename TIndex>
    SAMURAI_INLINE void ListOfIntervals<TValue, TIndex>::add_point(value_t point)
    {
        add_interval({point, point + 1});
    }

    /// Add an interval inside the list.
    template <typename TValue, typename TIndex>
    SAMURAI_INLINE void ListOfIntervals<TValue, TIndex>::add_interval(const interval_t& interval)
    {
        if (!interval.is_valid())
        {
            return;
        }

        // The intervals stored in the list are sorted and disjoint, so an
        // interval starting after the hinted one can only be inserted after it:
        // the search is then started from the hint rather than from the
        // beginning of the list. Without this, each insertion walks the whole
        // list and filling a list of n intervals costs O(n^2).
        auto search_from = before_begin();
        if (m_hint != before_begin() && interval.start >= m_hint->start)
        {
            if (interval.start <= m_hint->end)
            {
                // overlaps or touches the hinted interval: extend it
                m_hint->end = std::max(m_hint->end, interval.end);
                merge_following(m_hint, interval.end);
                return;
            }
            search_from = m_hint;
        }

        auto predicate = [interval](const auto& value)
        {
            return interval.start <= value.end;
        };
        auto it = detail::forward_find_if(search_from, end(), predicate);

        // if we are at the end just append the new interval or
        // if we are between two intervals, insert it
        if (it.second == end() || interval.end < it.second->start)
        {
            m_hint = this->insert_after(it.first, interval);
            return;
        }

        // else there is an overlap
        it.second->start = std::min(it.second->start, interval.start);
        it.second->end   = std::max(it.second->end, interval.end);
        merge_following(it.second, interval.end);
        m_hint = it.second;
    }

    /// Merge into *it all the following intervals it now overlaps.
    template <typename TValue, typename TIndex>
    SAMURAI_INLINE void ListOfIntervals<TValue, TIndex>::merge_following(iterator it, value_t interval_end)
    {
        auto it_next = std::next(it);
        while (it_next != end() && interval_end >= it_next->start)
        {
            it->end = std::max(it_next->end, interval_end);
            it_next = this->erase_after(it);
        }
    }

    template <typename value_t, typename index_t>
    SAMURAI_INLINE std::ostream& operator<<(std::ostream& out, const ListOfIntervals<value_t, index_t>& interval_list)
    {
        for (const auto& interval : interval_list)
        {
            out << interval << " ";
        }
        return out;
    }

    ///////////////////////////////////////
    // FlatListOfIntervals definition    //
    ///////////////////////////////////////

    /** @class FlatListOfIntervals
     *  @brief Vector-backed sorted list of disjoint intervals.
     *
     * Scratch buffer used by the subset workspaces. Same semantics as
     * ListOfIntervals (the buffer always holds the union of the added
     * intervals, sorted and disjoint) but on contiguous storage: insertion
     * uses a binary search plus a memmove of a handful of PODs instead of
     * walking a node-based list, and clear() keeps the capacity, so a buffer
     * reused across rows stops allocating after the first few insertions.
     */
    template <typename TValue, typename TIndex = default_config::index_t>
    struct FlatListOfIntervals
    {
        using value_t    = TValue;
        using index_t    = TIndex;
        using interval_t = Interval<value_t, index_t>;

        using storage_t      = std::vector<interval_t>;
        using const_iterator = typename storage_t::const_iterator;
        using iterator       = typename storage_t::iterator;
        using value_type     = interval_t;

        SAMURAI_INLINE void clear()
        {
            m_intervals.clear();
            m_hint = 0;
        }

        SAMURAI_INLINE bool empty() const
        {
            return m_intervals.empty();
        }

        SAMURAI_INLINE std::size_t size() const
        {
            return m_intervals.size();
        }

        SAMURAI_INLINE void add_interval(const interval_t& interval)
        {
            if (!interval.is_valid())
            {
                return;
            }

            if (m_intervals.empty())
            {
                m_intervals.push_back(interval);
                m_hint = 0;
                return;
            }

            // Same hint strategy as ListOfIntervals: intervals are almost
            // always added in increasing order, starting the search from the
            // last interval touched makes the insertion amortized O(1).
            std::size_t i = 0;
            if (interval.start >= m_intervals[m_hint].start)
            {
                if (interval.start <= m_intervals[m_hint].end)
                {
                    // overlaps or touches the hinted interval: extend it
                    if (interval.end > m_intervals[m_hint].end)
                    {
                        m_intervals[m_hint].end = interval.end;
                        absorb_following(m_hint);
                    }
                    return;
                }
                i = m_hint + 1;
            }

            // first interval that ends at or after the new one starts
            while (i != m_intervals.size() && m_intervals[i].end < interval.start)
            {
                ++i;
            }

            if (i == m_intervals.size())
            {
                m_intervals.push_back(interval);
            }
            else if (interval.end < m_intervals[i].start)
            {
                // disjoint: insert in place
                m_intervals.insert(m_intervals.begin() + std::ptrdiff_t(i), interval);
            }
            else
            {
                // overlap: extend *it, then absorb the intervals it now covers
                m_intervals[i].start = std::min(m_intervals[i].start, interval.start);
                if (interval.end > m_intervals[i].end)
                {
                    m_intervals[i].end = interval.end;
                    absorb_following(i);
                }
            }
            m_hint = i;
        }

        SAMURAI_INLINE void add_point(value_t point)
        {
            add_interval({point, point + 1});
        }

        SAMURAI_INLINE const_iterator cbegin() const
        {
            return m_intervals.cbegin();
        }

        SAMURAI_INLINE const_iterator cend() const
        {
            return m_intervals.cend();
        }

        SAMURAI_INLINE const_iterator begin() const
        {
            return m_intervals.begin();
        }

        SAMURAI_INLINE const_iterator end() const
        {
            return m_intervals.end();
        }

      private:

        /// Merge into m_intervals[i] all the following intervals it now overlaps.
        void absorb_following(std::size_t i)
        {
            std::size_t j = i + 1;
            while (j != m_intervals.size() && m_intervals[j].start <= m_intervals[i].end)
            {
                m_intervals[i].end = std::max(m_intervals[i].end, m_intervals[j].end);
                ++j;
            }
            m_intervals.erase(m_intervals.begin() + std::ptrdiff_t(i) + 1, m_intervals.begin() + std::ptrdiff_t(j));
        }

        storage_t m_intervals;

        /// Index of the last interval touched.
        std::size_t m_hint = 0;
    };

    template <typename value_t, typename index_t>
    SAMURAI_INLINE std::ostream& operator<<(std::ostream& out, const FlatListOfIntervals<value_t, index_t>& interval_list)
    {
        for (const auto& interval : interval_list)
        {
            out << interval << " ";
        }
        return out;
    }
} // namespace samurai
