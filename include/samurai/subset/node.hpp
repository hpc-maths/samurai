// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtensor/xfixed.hpp>

#include "../algorithm.hpp"
#include "apply.hpp"
#include "concepts.hpp"
#include "samurai/list_of_intervals.hpp"
#include "start_end_fct.hpp"
#include "visitor.hpp"

namespace samurai
{

    template <class Set, class Func>
    void apply(Set&& global_set, Func&& func);

    template <class Op, class StartEndOp, class... S>
    class Subset
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type                   = std::tuple<S...>;
        using interval_t                 = get_interval_t<S...>;

        Subset(Op&& op, StartEndOp&& start_end_op, S&&... s)
            : m_operator(std::forward<Op>(op))
            , m_start_end_op(std::forward<StartEndOp>(start_end_op))
            , m_s(std::forward<S>(s)...)
            , m_ref_level(compute_max(s.ref_level()...))
            , m_level(compute_max(s.level()...))
            , m_min_level(m_level)
        {
            std::apply(
                [this](auto&&... args)
                {
                    (args.ref_level(m_ref_level), ...);
                },
                m_s);
        }

        auto& on(auto level)
        {
            auto ilevel = static_cast<std::size_t>(level);
            if (ilevel > m_ref_level)
            {
                ref_level(ilevel);
            }
            m_min_level = std::min(m_min_level, ilevel);
            m_level     = ilevel;
            return *this;
        }

        template <class Func>
        void operator()(Func&& func)
        {
            apply(*this, std::forward<Func>(func));
        }

        template <class... ApplyOp>
        void apply_op(ApplyOp&&... op)
        {
            auto func = [&](auto& interval, auto& index)
            {
                (op(m_level, interval, index), ...);
            };
            apply(*this, func);
        }

        template <std::size_t d, class Func_goback_beg, class Func_goback_end>
        auto get_local_set(auto level, auto& index, Func_goback_beg&& goback_fct_beg, Func_goback_end&& goback_fct_end)
        {
            int shift = static_cast<int>(this->ref_level()) - static_cast<int>(this->level());
            m_start_end_op(m_level, m_min_level, m_ref_level);

            return std::apply(
                [this, &index, shift, level, &goback_fct_beg, &goback_fct_end](auto&&... args)
                {
                    return SetTraverser(shift,
                                        get_operator<d>(m_operator),
                                        args.template get_local_set<d>(
                                            level,
                                            index,
                                            m_start_end_op.template goback<d + 1>(std::forward<Func_goback_beg>(goback_fct_beg)),
                                            m_start_end_op.template goback<d + 1, true>(std::forward<Func_goback_end>(goback_fct_end)))...);
                },
                m_s);
        }

        template <std::size_t d>
        auto get_local_set(auto level, auto& index)
        {
            return get_local_set<d>(level, index, default_function_(), default_function_());
        }

        template <std::size_t d, class Func_start, class Func_end>
        auto get_start_and_stop_function(Func_start&& start_fct, Func_end&& end_fct)
        {
            m_start_end_op(m_level, m_min_level, m_ref_level);

            return std::apply(
                [this, &start_fct, &end_fct](auto&& arg, auto&&... args)
                {
                    if constexpr (std::is_same_v<Op, DifferenceOp>)
                    {
                        return std::make_tuple(std::move(arg.template get_start_and_stop_function<d>(
                                                   m_start_end_op.template start<d>(std::forward<Func_start>(start_fct)),
                                                   m_start_end_op.template end<d>(std::forward<Func_end>(end_fct)))),
                                               std::move(args.template get_start_and_stop_function<d>(
                                                   m_start_end_op.template start<d, true>(std::forward<Func_start>(start_fct)),
                                                   m_start_end_op.template end<d, true>(std::forward<Func_end>(end_fct))))...);
                    }
                    else
                    {
                        return std::make_tuple(std::move(arg.template get_start_and_stop_function<d>(
                                                   m_start_end_op.template start<d>(std::forward<Func_start>(start_fct)),
                                                   m_start_end_op.template end<d>(std::forward<Func_end>(end_fct)))),
                                               std::move(args.template get_start_and_stop_function<d>(
                                                   m_start_end_op.template start<d>(std::forward<Func_start>(start_fct)),
                                                   m_start_end_op.template end<d>(std::forward<Func_end>(end_fct))))...);
                    }
                },
                m_s);
        }

        template <std::size_t d>
        auto get_start_and_stop_function()
        {
            return get_start_and_stop_function<d>(default_function(), default_function());
        }

        auto level() const
        {
            return m_level;
        }

        auto ref_level() const
        {
            return m_ref_level;
        }

        void ref_level(auto level)
        {
            m_ref_level = level;
            std::apply(
                [this](auto&&... args)
                {
                    (args.ref_level(m_ref_level), ...);
                },
                m_s);
        }

        bool exist() const
        {
            return std::apply(
                [this](auto&&... args)
                {
                    return m_operator.exist(args...);
                },
                m_s);
        }

      protected:

        Op m_operator;
        StartEndOp m_start_end_op;
        set_type m_s;
        std::size_t m_ref_level;
        std::size_t m_level;
        std::size_t m_min_level;
    };

    template <class lca_t>
        requires IsLCA<lca_t>
    struct Self
    {
        static constexpr std::size_t dim = lca_t::dim;
        using interval_t                 = typename lca_t::interval_t;
        using value_t                    = typename interval_t::value_t;

        explicit Self(const lca_t& lca)
            : m_lca(lca)
            , m_level(lca.level())
            , m_ref_level(m_level)
            , m_min_level(m_level)
        {
        }

        template <class Func>
        void operator()(Func&& func)
        {
            apply(*this, std::forward<Func>(func));
        }

        template <class... ApplyOp>
        void apply_op(ApplyOp&&... op)
        {
            auto func = [&](auto& interval, auto& index)
            {
                (op(m_level, interval, index), ...);
            };
            apply(*this, func);
        }

        template <std::size_t d, class Func_goback_beg, class Func_goback_end>
        auto get_local_set(auto level, auto& index, Func_goback_beg&& goback_fct_beg, Func_goback_end&& goback_fct_end)
        {
            if (m_lca[d - 1].empty())
            {
                return IntervalListVisitor(IntervalListRange(m_lca[d - 1], 0, 0));
            }

            if constexpr (dim == d)
            {
                m_offsets[d - 1].clear();
                m_offsets[d - 1].push_back({0, m_lca[d - 1].size()});

                return IntervalListVisitor(m_lca.level(),
                                           m_level,
                                           m_ref_level,
                                           IntervalListRange(m_lca[d - 1], 0, static_cast<std::ptrdiff_t>(m_lca[d - 1].size())));
            }
            else
            {
                if (m_offsets[d].empty() || m_lca[d].empty())
                {
                    return IntervalListVisitor(IntervalListRange(m_lca[d - 1], 0, 0));
                }

                auto new_goback_fct_beg = m_func.template goback<d + 1>(std::forward<Func_goback_beg>(goback_fct_beg));

                if (level <= m_level && level >= m_lca.level())
                {
                    m_offsets[d - 1].clear();

                    auto current_index = start_shift(new_goback_fct_beg(level, index[d - 1]).second,
                                                     static_cast<int>(m_lca.level()) - static_cast<int>(m_level));
                    auto j             = find_on_dim(m_lca, d, m_offsets[d][0][0], m_offsets[d][0][1], current_index);

                    if (j == std::numeric_limits<std::size_t>::max())
                    {
                        return IntervalListVisitor(IntervalListRange(m_lca[d - 1], 0, 0));
                    }

                    auto io       = static_cast<std::size_t>(m_lca[d][j].index + current_index);
                    auto& offsets = m_lca.offsets(d);
                    m_offsets[d - 1].push_back({offsets[io], offsets[io + 1]});

                    return IntervalListVisitor(m_lca.level(),
                                               m_level,
                                               m_ref_level,
                                               IntervalListRange(m_lca[d - 1],
                                                                 static_cast<std::ptrdiff_t>(offsets[io]),
                                                                 static_cast<std::ptrdiff_t>(offsets[io + 1])));
                }
                else
                {
                    auto new_goback_fct_end = m_func.template goback<d + 1, true>(std::forward<Func_goback_end>(goback_fct_end));

                    auto min_index = start_shift(new_goback_fct_beg(level, index[d - 1]).second,
                                                 static_cast<int>(m_lca.level()) - static_cast<int>(m_level));

                    auto max_index = end_shift(new_goback_fct_end(level, index[d - 1] + 1).second,
                                               static_cast<int>(m_lca.level()) - static_cast<int>(m_level));

                    m_work[d - 1].clear();
                    m_offsets[d - 1].clear();

                    if constexpr (d == dim - 1)
                    {
                        auto j_min = lower_bound_interval(m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_offsets[d][0][0]),
                                                          m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_offsets[d][0][1]),
                                                          min_index);
                        auto j_max = upper_bound_interval(j_min, m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_offsets[d][0][1]), max_index)
                                   - 1;

                        if (j_min != m_lca[d].end() && j_min <= j_max)
                        {
                            auto start_offset = static_cast<std::size_t>(j_min->index + j_min->start);
                            if (j_min->contains(min_index))
                            {
                                start_offset = static_cast<std::size_t>(j_min->index + min_index);
                            }

                            auto end_offset = static_cast<std::size_t>(j_max->index + j_max->end);
                            if (j_max->contains(max_index))
                            {
                                end_offset = static_cast<std::size_t>(j_max->index + max_index);
                            }

                            if (start_offset == end_offset)
                            {
                                return IntervalListVisitor(IntervalListRange(m_lca[d - 1], 0, 0));
                            }

                            m_offsets[d - 1].push_back({start_offset, end_offset});

                            ListOfIntervals<value_t> list_of_intervals;
                            for (std::size_t o = m_lca.offsets(d)[start_offset]; o < m_lca.offsets(d)[end_offset]; ++o)
                            {
                                auto start = m_lca[d - 1][o].start;
                                auto end   = m_lca[d - 1][o].end;
                                list_of_intervals.add_interval({start, end});
                            }

                            for (auto& i : list_of_intervals)
                            {
                                m_work[d - 1].push_back(i);
                            }
                        }
                    }
                    else
                    {
                        ListOfIntervals<value_t> list_of_intervals;

                        for (auto& offset : m_offsets[d])
                        {
                            for (std::size_t io = offset[0]; io < offset[1]; ++io)
                            {
                                auto j_min = lower_bound_interval(
                                    m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_lca.offsets(d + 1)[io]),
                                    m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_lca.offsets(d + 1)[io + 1]),
                                    min_index);
                                auto j_max = upper_bound_interval(
                                                 j_min,
                                                 m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_lca.offsets(d + 1)[io + 1]),
                                                 max_index)
                                           - 1;

                                if (j_min != m_lca[d].begin() + static_cast<std::ptrdiff_t>(m_lca.offsets(d + 1)[io + 1]) && j_min <= j_max)
                                {
                                    auto start_offset = static_cast<std::size_t>(j_min->index + j_min->start);
                                    if (j_min->contains(min_index))
                                    {
                                        start_offset = static_cast<std::size_t>(j_min->index + min_index);
                                    }

                                    auto end_offset = static_cast<std::size_t>(j_max->index + j_max->end);
                                    if (j_max->contains(max_index))
                                    {
                                        end_offset = static_cast<std::size_t>(j_max->index + max_index);
                                    }

                                    if (start_offset == end_offset)
                                    {
                                        continue;
                                    }

                                    m_offsets[d - 1].push_back({start_offset, end_offset});

                                    for (std::size_t o = m_lca.offsets(d)[start_offset]; o < m_lca.offsets(d)[end_offset]; ++o)
                                    {
                                        auto start = m_lca[d - 1][o].start;
                                        auto end   = m_lca[d - 1][o].end;
                                        list_of_intervals.add_interval({start, end});
                                    }
                                }
                            }

                            for (auto& i : list_of_intervals)
                            {
                                m_work[d - 1].push_back(i);
                            }
                        }
                    }
                    if (m_work[d - 1].empty())
                    {
                        return IntervalListVisitor(IntervalListRange(m_lca[d - 1], 0, 0));
                    }
                    return IntervalListVisitor(m_lca.level(), m_level, m_ref_level, IntervalListRange(m_lca[d - 1], m_work[d - 1]));
                }
            }
        }

        template <std::size_t d>
        auto get_local_set(auto level, auto& index)

        {
            return get_local_set<d>(level, index, default_function_(), default_function_());
        }

        template <std::size_t d, class Func_start, class Func_end>
        auto get_start_and_stop_function(Func_start&& start_fct, Func_end&& end_fct)
        {
            m_func(m_level, m_min_level, m_ref_level);
            auto new_start_fct = m_func.template start<d>(std::forward<Func_start>(start_fct));
            auto new_end_fct   = m_func.template end<d>(std::forward<Func_end>(end_fct));
            return std::make_tuple(std::move(new_start_fct), std::move(new_end_fct));
        }

        template <std::size_t d>
        auto get_start_and_stop_function()

        {
            return get_start_and_stop_function<d>(default_function(), default_function());
        }

        auto ref_level() const
        {
            return m_ref_level;
        }

        void ref_level(auto level)
        {
            m_ref_level = level;
        }

        auto level() const
        {
            return m_level;
        }

        auto& on(auto level)
        {
            m_min_level = std::min(m_min_level, static_cast<std::size_t>(level));
            m_level     = static_cast<std::size_t>(level);
            return *this;
        }

        bool exist() const
        {
            return !m_lca.empty();
        }

        const lca_t& m_lca;
        std::size_t m_level;
        std::size_t m_ref_level;
        std::size_t m_min_level;
        start_end_function<dim> m_func;
        std::array<std::vector<interval_t>, dim - 1> m_work;
        std::array<std::vector<std::array<std::size_t, 2>>, dim> m_offsets;
    };

    namespace detail
    {
        template <std::size_t dim, class interval_t>
        auto transform(const LevelCellArray<dim, interval_t>& lca)
        {
            return Self<LevelCellArray<dim, interval_t>>(lca);
        }

        template <std::size_t dim, class interval_t>
        auto transform(LevelCellArray<dim, interval_t>& lca)
        {
            return Self<LevelCellArray<dim, interval_t>>(lca);
        }

        template <class E>
        auto transform(E&& e)
        {
            return std::forward<E>(e);
        }
    }

    template <class... sets_t>
    auto intersection(sets_t&&... sets)
    {
        static constexpr std::size_t dim = get_set_dim_v<sets_t...>;
        return std::apply(
            [](auto&&... args)
            {
                return Subset(IntersectionOp(), start_end_function<dim>(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto union_(sets_t&&... sets)
    {
        static constexpr std::size_t dim = get_set_dim_v<sets_t...>;
        return std::apply(
            [](auto&&... args)
            {
                return Subset(UnionOp(), start_end_function<dim>(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto difference(sets_t&&... sets)
    {
        static constexpr std::size_t dim = get_set_dim_v<sets_t...>;
        return std::apply(
            [](auto&&... args)
            {
                return Subset(DifferenceOp(), start_end_function<dim>(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class set_t, class stencil_t>
    auto translate(set_t&& set, const stencil_t& stencil)
    {
        constexpr std::size_t dim = std::decay_t<set_t>::dim;
        return Subset(SelfOp(),
                      start_end_translate_function<dim>(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)),
                      detail::transform(std::forward<set_t>(set)));
    }

    template <class set_t>
    auto contraction(set_t&& set, int c)
    {
        constexpr std::size_t dim = std::decay_t<set_t>::dim;
        return Subset(SelfOp(), start_end_contraction_function<dim>(c), detail::transform(std::forward<set_t>(set)));
    }

    template <class lca_t>
    auto self(lca_t&& lca)
    {
        return Self<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
