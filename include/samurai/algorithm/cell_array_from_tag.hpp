// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#include "../cell_array.hpp"
#include "../cell_flag.hpp"
#include "../timers.hpp"

namespace samurai
{
    /// The cells the tags add to and remove from a CellArray (update_cell_array_from_tag).
    template <class CA>
    struct TagChanges
    {
        CA add_m;    ///< the parents of the cells to coarsen, one level down
        CA remove_m; ///< the cells to coarsen
        CA add_p;    ///< the children of the cells to refine, one level up
        CA remove_p; ///< the cells to refine
    };

    namespace detail
    {
        /// What the tag of a cell does to it at a given level.
        enum class TagAction : std::uint8_t
        {
            keep,
            coarsen,
            refine
        };

        /// The word-wise read of one-byte tags: 8 tags per 64-bit word.
        class TagWords
        {
          public:

            TagWords(bool can_refine, bool can_coarsen)
                : m_refine_mask(can_refine ? ones : 0)
                , m_coarsen_mask(can_coarsen ? ones : 0)
            {
            }

            /// True when the n tags at `tags` all give keep. They start at position `first` of the tag array.
            bool all_keep(const std::uint8_t* tags, std::size_t first, std::size_t n) const
            {
                if (n < 8 and first + n >= 8 and std::endian::native == std::endian::little)
                {
                    // the word that ends with the last tag, without its low bytes: the tags before the first one
                    return (changed(load(tags + n - 8)) >> (8 * (8 - n))) == 0;
                }
                if (n < 8)
                {
                    std::uint64_t w = 0;
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        w |= std::uint64_t{tags[i]} << (8 * i);
                    }
                    return changed(w) == 0;
                }
                // the last word may overlap the previous one
                std::uint64_t acc = changed(load(tags + n - 8));
                for (std::size_t i = 0; i + 8 < n; i += 8)
                {
                    acc |= changed(load(tags + i));
                }
                return acc == 0;
            }

            /// True when the 8 tags at `tags` all give `action`.
            bool same_action_8(const std::uint8_t* tags, TagAction action) const
            {
                const std::uint64_t w = load(tags);
                switch (action)
                {
                    case TagAction::refine:
                        return refined(w) == ones;
                    case TagAction::coarsen:
                        return coarsened(w) == ones;
                    default:
                        return changed(w) == 0;
                }
            }

          private:

            static_assert(static_cast<int>(CellFlag::keep) == 1 and static_cast<int>(CellFlag::coarsen) == 2
                              and static_cast<int>(CellFlag::refine) == 4,
                          "the bit operations below assume these flags");
            static constexpr std::uint64_t ones = 0x0101010101010101ULL;

            static std::uint64_t load(const std::uint8_t* tags)
            {
                std::uint64_t w;
                std::memcpy(&w, tags, sizeof(w));
                return w;
            }

            // Bit 0 of each byte tells whether its cell is refined, coarsened, refined or coarsened.
            std::uint64_t refined(std::uint64_t w) const
            {
                return (w >> 2) & m_refine_mask;
            }

            std::uint64_t coarsened(std::uint64_t w) const
            {
                return (w >> 1) & ~w & ~refined(w) & m_coarsen_mask;
            }

            std::uint64_t changed(std::uint64_t w) const
            {
                return refined(w) | ((w >> 1) & ~w & m_coarsen_mask);
            }

            std::uint64_t m_refine_mask;
            std::uint64_t m_coarsen_mask;
        };

        /// The tags of the cells of one level, read interval by interval.
        template <class T>
        class LevelTags
        {
          public:

            /// Refine applies below the max level, coarsen above the min level.
            LevelTags(const T* tags, bool can_refine, bool can_coarsen)
                : m_tags(tags)
                , m_can_refine(can_refine)
                , m_can_coarsen(can_coarsen)
                , m_words(can_refine, can_coarsen)
            {
            }

            /// Refine wins over coarsen, and keep cancels coarsen.
            TagAction action(const T& tag) const
            {
                if (m_can_refine and (tag & static_cast<std::uint8_t>(CellFlag::refine)))
                {
                    return TagAction::refine;
                }
                if (m_can_coarsen and (tag & static_cast<std::uint8_t>(CellFlag::coarsen))
                    and not(tag & static_cast<std::uint8_t>(CellFlag::keep)))
                {
                    return TagAction::coarsen;
                }
                return TagAction::keep;
            }

            /// True when all the cells of i keep their level.
            template <class TInterval>
            bool all_keep(const TInterval& i) const
            {
                const T* tags = first_tag(i);
                if constexpr (one_byte)
                {
                    return m_words.all_keep(tags, static_cast<std::size_t>(tags - m_tags), i.size());
                }
                else
                {
                    return std::all_of(tags,
                                       tags + i.size(),
                                       [&](const T& tag)
                                       {
                                           return action(tag) == TagAction::keep;
                                       });
                }
            }

            /// Calls f(action, run) on each maximal run of cells of i with the same action.
            template <class TInterval, class Func>
            void for_each_run(const TInterval& i, Func&& f) const
            {
                const T* tags       = first_tag(i);
                const auto tag_of_x = [&](auto x)
                {
                    return tags + static_cast<std::size_t>(x - i.start);
                };
                TagAction run  = action(tags[0]);
                auto run_start = i.start;
                for (auto x = i.start + 1; x < i.end; ++x)
                {
                    if constexpr (one_byte)
                    {
                        // most cells continue the current run: skip them 8 at a time
                        while (i.end - x >= 8 and m_words.same_action_8(tag_of_x(x), run))
                        {
                            x += 8;
                        }
                        if (x == i.end)
                        {
                            break;
                        }
                    }
                    const TagAction a = action(*tag_of_x(x));
                    if (a != run)
                    {
                        f(run, TInterval{run_start, x});
                        run       = a;
                        run_start = x;
                    }
                }
                f(run, TInterval{run_start, i.end});
            }

          private:

            static constexpr bool one_byte = std::is_same_v<T, std::uint8_t>;

            template <class TInterval>
            const T* first_tag(const TInterval& i) const
            {
                return m_tags + static_cast<std::size_t>(i.index + i.start);
            }

            const T* m_tags;
            bool m_can_refine;
            bool m_can_coarsen;
            TagWords m_words;
        };

        template <class Tag>
        auto make_level_tags(const Tag& tag, std::size_t level)
        {
            using T = std::remove_cv_t<std::remove_reference_t<decltype(*tag.data())>>;
            return LevelTags<T>(tag.data(), level<tag.mesh().max_level(), level> tag.mesh().min_level());
        }

        // The rows below the intervals [first, last) of lca[d]. yz holds the coordinates above d.
        template <std::size_t d, class LCA, class Coord, class Func>
        void for_each_row(const LCA& lca, std::size_t first, std::size_t last, Coord& yz, Func&& f)
        {
            if constexpr (d == 0)
            {
                f(yz, std::span(lca[0].data() + first, last - first));
            }
            else
            {
                const auto& intervals = lca[d];
                const auto& offsets   = lca.offsets(d);
                for (std::size_t i = first; i < last; ++i)
                {
                    for (auto y = intervals[i].start; y < intervals[i].end; ++y)
                    {
                        yz[d - 1]             = y;
                        const auto row_offset = static_cast<std::size_t>(intervals[i].index + y);
                        for_each_row<d - 1>(lca, offsets[row_offset], offsets[row_offset + 1], yz, f);
                    }
                }
            }
        }

        /// Calls f(yz, intervals) on each row of lca, in order. Faster than the iterator of LevelCellArray.
        template <class LCA, class Func>
        void for_each_row(const LCA& lca, Func&& f)
        {
            typename LCA::coord_type yz;
            yz.fill(0);
            for_each_row<LCA::dim - 1>(lca, 0, lca[LCA::dim - 1].size(), yz, f);
        }

        /// Rows are ordered by their last coordinate first, as in a LevelCellArray.
        template <class Coord>
        bool row_before(const Coord& a, const Coord& b)
        {
            for (std::size_t d = a.size(); d-- > 0;)
            {
                if (a[d] != b[d])
                {
                    return a[d] < b[d];
                }
            }
            return false;
        }

        template <class Coord>
        bool same_row(const Coord& a, const Coord& b)
        {
            return std::equal(a.cbegin(), a.cend(), b.cbegin());
        }

        /// Intervals stored row by row, in the order of a LevelCellArray.
        template <std::size_t dim, class TInterval>
        class IntervalRows
        {
          public:

            using coord_type = typename LevelCellArray<dim, TInterval>::coord_type;

            std::size_t size() const
            {
                return m_rows.size();
            }

            const coord_type& yz(std::size_t row) const
            {
                return m_rows[row].yz;
            }

            std::span<const TInterval> intervals(std::size_t row) const
            {
                return std::span(m_intervals.data() + m_rows[row].first, m_rows[row].last - m_rows[row].first);
            }

            /// Appends an interval after all the others, in the same row or a later one.
            void add(const TInterval& i, const coord_type& yz)
            {
                if (m_rows.empty() or not same_row(m_rows.back().yz, yz))
                {
                    m_rows.push_back({yz, m_intervals.size(), m_intervals.size()});
                }
                m_intervals.push_back(i);
                m_rows.back().last = m_intervals.size();
            }

            /// The children of the cells of `parents`, one level up.
            static IntervalRows children(const IntervalRows& parents)
            {
                IntervalRows rows;
                if (!parents.m_rows.empty())
                {
                    coord_type child_yz;
                    child_yz.fill(0);
                    rows.add_children(parents, 0, parents.m_rows.size(), static_cast<int>(dim) - 2, child_yz);
                }
                return rows;
            }

          private:

            // The rows [first, last) of parents share their coordinates above d. Their children come in order
            // when the coordinate d is set before the lower ones.
            void add_children(const IntervalRows& parents, std::size_t first, std::size_t last, int d, coord_type& yz)
            {
                if (d < 0) // all coordinates are set: a single row
                {
                    for (const auto& i : parents.intervals(first))
                    {
                        add({2 * i.start, 2 * i.end}, yz);
                    }
                    return;
                }
                const auto dd = static_cast<std::size_t>(d);
                for (std::size_t group = first; group < last;)
                {
                    const auto y          = parents.m_rows[group].yz[dd];
                    std::size_t group_end = group + 1;
                    while (group_end < last and parents.m_rows[group_end].yz[dd] == y)
                    {
                        ++group_end;
                    }
                    for (yz[dd] = 2 * y; yz[dd] < 2 * y + 2; ++yz[dd])
                    {
                        add_children(parents, group, group_end, d - 1, yz);
                    }
                    group = group_end;
                }
            }

            struct Row
            {
                coord_type yz;
                std::size_t first; ///< [first, last) in m_intervals
                std::size_t last;
            };

            std::vector<Row> m_rows;
            std::vector<TInterval> m_intervals;
        };

        /// Adds to the row yz of lca the union of the sorted intervals of each source.
        template <std::size_t dim, class TInterval, std::size_t n>
        void add_row_union(LevelCellArray<dim, TInterval>& lca,
                           const typename LevelCellArray<dim, TInterval>::coord_type& yz,
                           std::array<std::span<const TInterval>, n> sources)
        {
            const auto non_empty = std::count_if(sources.cbegin(),
                                                 sources.cend(),
                                                 [](const auto& s)
                                                 {
                                                     return !s.empty();
                                                 });
            if (non_empty == 1) // nothing to merge
            {
                for (const auto& s : sources)
                {
                    for (const auto& i : s)
                    {
                        lca.add_interval_back(i, yz);
                    }
                }
                return;
            }

            // take the lowest start each time, joining the intervals that overlap
            bool has_current = false;
            TInterval current;
            while (true)
            {
                std::span<const TInterval>* lowest = nullptr;
                for (auto& s : sources)
                {
                    if (!s.empty() and (lowest == nullptr or s.front().start < lowest->front().start))
                    {
                        lowest = &s;
                    }
                }
                if (lowest == nullptr)
                {
                    break;
                }
                const TInterval i = lowest->front();
                *lowest           = lowest->subspan(1);
                if (has_current and i.start <= current.end)
                {
                    current.end = std::max(current.end, i.end);
                }
                else
                {
                    if (has_current)
                    {
                        lca.add_interval_back(current, yz);
                    }
                    current     = i;
                    has_current = true;
                }
            }
            if (has_current)
            {
                lca.add_interval_back(current, yz);
            }
        }

        /// Calls f(action, run, yz) on each run of cells of lca to refine or to coarsen, in the order of lca.
        /// Appends to `changed` the position in lca[0] of the intervals that hold such a run.
        template <class LCA, class T, class Func>
        void for_each_tag_change(const LCA& lca, const LevelTags<T>& tags, std::vector<std::size_t>& changed, Func&& f)
        {
            const auto* first_interval = lca[0].data();
            for_each_row(lca,
                         [&](const auto& yz, const auto& intervals)
                         {
                             for (const auto& i : intervals)
                             {
                                 if (tags.all_keep(i))
                                 {
                                     continue;
                                 }
                                 changed.push_back(static_cast<std::size_t>(&i - first_interval));
                                 tags.for_each_run(i,
                                                   [&](TagAction action, const auto& run)
                                                   {
                                                       if (action != TagAction::keep)
                                                       {
                                                           f(action, run, yz);
                                                       }
                                                   });
                             }
                         });
        }

        /// The parents of the even cells of `cells`, one level down, if the row yz is even in every direction:
        /// only the first child of a parent adds it.
        template <class TInterval, class Coord>
        bool first_child_parents(const TInterval& cells, const Coord& yz, TInterval& parents, Coord& parents_yz)
        {
            for (std::size_t d = 0; d < yz.size(); ++d)
            {
                if (yz[d] & 1)
                {
                    return false;
                }
                parents_yz[d] = yz[d] >> 1;
            }
            parents = {(cells.start + 1) >> 1, (cells.end + 1) >> 1};
            return parents.start < parents.end;
        }

        /// Adds to new_lca the union of the kept cells of old_lca, of `children` and of `parents`.
        /// `changed` lists the positions in old_lca[0] of the intervals whose cells do not all keep their level.
        template <std::size_t dim, class TInterval, class T>
        void add_new_level(LevelCellArray<dim, TInterval>& new_lca,
                           const LevelCellArray<dim, TInterval>& old_lca,
                           const LevelTags<T>& tags,
                           const std::vector<std::size_t>& changed,
                           const IntervalRows<dim, TInterval>& children,
                           const IntervalRows<dim, TInterval>& parents)
        {
            using span_t     = std::span<const TInterval>;
            using coord_type = typename IntervalRows<dim, TInterval>::coord_type;

            std::size_t next_child  = 0;
            std::size_t next_parent = 0;
            // adds the rows of children and parents before yz, or all of them when yz is null
            const auto add_added_rows_before = [&](const coord_type* yz)
            {
                while (true)
                {
                    const bool has_child  = next_child < children.size();
                    const bool has_parent = next_parent < parents.size();
                    if (!has_child and !has_parent)
                    {
                        return;
                    }
                    const auto& row = (has_child and (!has_parent or row_before(children.yz(next_child), parents.yz(next_parent))))
                                        ? children.yz(next_child)
                                        : parents.yz(next_parent);
                    if (yz != nullptr and !row_before(row, *yz))
                    {
                        return;
                    }
                    span_t c;
                    span_t p;
                    if (has_child and same_row(children.yz(next_child), row))
                    {
                        c = children.intervals(next_child++);
                    }
                    if (has_parent and same_row(parents.yz(next_parent), row))
                    {
                        p = parents.intervals(next_parent++);
                    }
                    add_row_union<dim, TInterval, 2>(new_lca, row, {c, p});
                }
            };

            // the new level has about the size of the old one: grow its arrays once
            for (std::size_t d = 0; d < dim; ++d)
            {
                new_lca[d].reserve(old_lca[d].size() + children.size() + parents.size());
                if (d > 0)
                {
                    new_lca.offsets(d).reserve(old_lca.offsets(d).size() + children.size() + parents.size());
                }
            }

            const auto* first_interval = old_lca[0].data();
            std::size_t next_changed   = 0;
            std::vector<TInterval> kept;
            for_each_row(
                old_lca,
                [&](const coord_type& yz, const span_t& intervals)
                {
                    add_added_rows_before(&yz);

                    // the old intervals, cut where cells change level
                    span_t row_kept    = intervals;
                    const auto row_end = static_cast<std::size_t>(intervals.data() + intervals.size() - first_interval);
                    if (next_changed < changed.size() and changed[next_changed] < row_end)
                    {
                        kept.clear();
                        for (const auto& i : intervals)
                        {
                            if (next_changed < changed.size() and changed[next_changed] == static_cast<std::size_t>(&i - first_interval))
                            {
                                ++next_changed;
                                tags.for_each_run(i,
                                                  [&](TagAction action, const TInterval& run)
                                                  {
                                                      if (action == TagAction::keep)
                                                      {
                                                          kept.push_back(run);
                                                      }
                                                  });
                            }
                            else
                            {
                                kept.push_back(i);
                            }
                        }
                        row_kept = span_t(kept);
                    }

                    span_t c;
                    span_t p;
                    if (next_child < children.size() and same_row(children.yz(next_child), yz))
                    {
                        c = children.intervals(next_child++);
                    }
                    if (next_parent < parents.size() and same_row(parents.yz(next_parent), yz))
                    {
                        p = parents.intervals(next_parent++);
                    }
                    add_row_union<dim, TInterval, 3>(new_lca, yz, {row_kept, c, p});
                });
            add_added_rows_before(nullptr);
        }
    } // namespace detail

    /// The cells to add and to remove to apply the tags of the cells of old_ca: a cell tagged refine below the
    /// max level is replaced by its children, a cell tagged coarsen and not keep above the min level by its parent.
    template <std::size_t dim, class TInterval, size_t max_size, class Tag>
    TagChanges<CellArray<dim, TInterval, max_size>> cells_changed_by_tag(const CellArray<dim, TInterval, max_size>& old_ca, const Tag& tag)
    {
        using detail::TagAction;
        using coord_type = typename LevelCellArray<dim, TInterval>::coord_type;
        using rows_t     = detail::IntervalRows<dim, TInterval>;

        TagChanges<CellArray<dim, TInterval, max_size>> changes;
        std::vector<std::size_t> changed;
        // On an empty (sub)mesh min_level() > max_level(): no level is visited.
        for (std::size_t level = old_ca.min_level(); level <= old_ca.max_level(); ++level)
        {
            changed.clear();
            rows_t refined;
            detail::for_each_tag_change(old_ca[level],
                                        detail::make_level_tags(tag, level),
                                        changed,
                                        [&](TagAction action, const TInterval& cells, const coord_type& yz)
                                        {
                                            TInterval parents;
                                            coord_type parents_yz;
                                            if (action == TagAction::refine)
                                            {
                                                changes.remove_p[level].add_interval_back(cells, yz);
                                                refined.add(cells, yz);
                                            }
                                            else
                                            {
                                                changes.remove_m[level].add_interval_back(cells, yz);
                                                if (detail::first_child_parents(cells, yz, parents, parents_yz))
                                                {
                                                    changes.add_m[level - 1].add_interval_back(parents, parents_yz);
                                                }
                                            }
                                        });
            const rows_t children = rows_t::children(refined);
            for (std::size_t r = 0; r < children.size(); ++r)
            {
                for (const auto& i : children.intervals(r))
                {
                    changes.add_p[level + 1].add_interval_back(i, children.yz(r));
                }
            }
        }
        return changes;
    }

    /// The cell array after applying the tags of the cells of old_ca (see cells_changed_by_tag).
    /// Level by level: the cells of old_ca that keep their level, the children of the refined cells of the level
    /// below and the parents of the coarsened cells of the level above, merged row by row.
    template <std::size_t dim, class TInterval, size_t max_size, class Tag>
    CellArray<dim, TInterval, max_size> update_cell_array_from_tag(const CellArray<dim, TInterval, max_size>& old_ca, const Tag& tag)
    {
        ScopedTimer timer("update_cell_array_from_tag");
        using detail::TagAction;
        using coord_type = typename LevelCellArray<dim, TInterval>::coord_type;
        using rows_t     = detail::IntervalRows<dim, TInterval>;

        // what the tags change, level by level
        std::array<std::vector<std::size_t>, max_size + 1> changed;
        std::array<rows_t, max_size + 1> refined;
        std::array<rows_t, max_size + 1> parents;
        // On an empty (sub)mesh min_level() > max_level(): no level is visited.
        for (std::size_t level = old_ca.min_level(); level <= old_ca.max_level(); ++level)
        {
            detail::for_each_tag_change(old_ca[level],
                                        detail::make_level_tags(tag, level),
                                        changed[level],
                                        [&](TagAction action, const TInterval& cells, const coord_type& yz)
                                        {
                                            TInterval p;
                                            coord_type p_yz;
                                            if (action == TagAction::refine)
                                            {
                                                refined[level].add(cells, yz);
                                            }
                                            else if (detail::first_child_parents(cells, yz, p, p_yz))
                                            {
                                                parents[level - 1].add(p, p_yz);
                                            }
                                        });
        }

        CellArray<dim, TInterval, max_size> new_ca;
        for (std::size_t level = tag.mesh().min_level(); level <= tag.mesh().max_level(); ++level)
        {
            const rows_t children = level > 0 ? rows_t::children(refined[level - 1]) : rows_t{};
            detail::add_new_level(new_ca[level], old_ca[level], detail::make_level_tags(tag, level), changed[level], children, parents[level]);
        }
        return new_ca;
    }
}
