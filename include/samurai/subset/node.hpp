// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <ios>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtensor/xfixed.hpp>

#include "../algorithm.hpp"
#include "concepts.hpp"
#include "interval_interface.hpp"
#include "samurai/list_of_intervals.hpp"

// namespace samurai::experimental
namespace samurai
{

    template <class Set, class Func>
    void apply(Set&& global_set, Func&& func);

    template <class Operator, class... S>
    class SetOp
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type                   = std::tuple<S...>;
        using interval_t                 = get_interval_t<S...>;

        SetOp(int shift, const Operator& op, S&&... s)
            : m_shift(shift)
            , m_operator(op)
            , m_s(std::forward<S>(s)...)
        {
        }

        auto shift() const
        {
            return m_shift;
        }

        bool is_in(auto scan) const
        {
            return std::apply(
                [this, scan](auto&&... args)
                {
                    return m_operator.is_in(scan, args...);
                },
                m_s);
        }

        bool is_empty() const
        {
            return std::apply(
                [this](auto&&... args)
                {
                    return m_operator.is_empty(args...);
                },
                m_s);
        }

        auto min() const
        {
            return std::apply(
                [](auto&&... args)
                {
                    return compute_min(args.min()...);
                },
                m_s);
        }

        template <class StartEnd>
        void next(auto scan, StartEnd&& start_and_stop)
        {
            zip_apply(
                [scan](auto& arg, auto& start_end_fct)
                {
                    arg.next(scan, start_end_fct);
                },
                m_s,
                std::forward<StartEnd>(start_and_stop));
        }

      private:

        int m_shift;
        Operator m_operator;
        set_type m_s;
    };

    struct IntersectionOp
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) && ...);
        }

        bool is_empty(const auto&... args) const
        {
            return (args.is_empty() || ...);
        }

        bool exist(const auto&... args) const
        {
            return (args.exist() && ...);
        }
    };

    struct UnionOp
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) || ...);
        }

        bool is_empty(const auto&... args) const
        {
            return (args.is_empty() && ...);
        }

        bool exist(const auto&... args) const
        {
            return (args.exist() || ...);
        }
    };

    struct DifferenceOp
    {
        bool is_in(auto scan, const auto& arg, const auto&... args) const
        {
            return arg.is_in(scan) && !(args.is_in(scan) || ...);
        }

        bool is_empty(const auto& arg, const auto&...) const
        {
            return arg.is_empty();
        }

        bool exist(const auto& arg, const auto&...) const
        {
            return arg.exist();
        }
    };

    struct Difference2Op
    {
        bool is_in(auto scan, const auto& arg, const auto&...) const
        {
            return arg.is_in(scan);
        }

        bool is_empty(const auto& arg, const auto&...) const
        {
            return arg.is_empty();
        }

        bool exist(const auto& arg, const auto&...) const
        {
            return arg.exist();
        }
    };

    template <std::size_t d, class operator_t>
    auto get_operator(const operator_t& op)
    {
        return op;
    }

    template <std::size_t d>
    auto get_operator(const DifferenceOp& op)
    {
        if constexpr (d == 1)
        {
            return op;
        }
        else
        {
            return Difference2Op();
        }
    }

    struct TranslationOp
    {
        bool is_in(auto scan, const auto& arg) const
        {
            return arg.is_in(scan);
        }

        bool is_empty(const auto& arg) const
        {
            return arg.is_empty();
        }

        bool exist(const auto& arg) const
        {
            return arg.exist();
        }
    };

    template <class Op, class StartEndOp, class... S>
    class subset
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type                   = std::tuple<S...>;
        using interval_t                 = get_interval_t<S...>;

        subset(Op&& op, StartEndOp&& start_end_op, S&&... s)
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
            if (static_cast<int>(level) > m_ref_level)
            {
                ref_level(level);
            }
            m_min_level = std::min(m_min_level, static_cast<int>(level));
            m_level     = static_cast<int>(level);
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
                (op(static_cast<std::size_t>(m_level), interval, index), ...);
            };
            apply(*this, func);
        }

        template <std::size_t d, class Func_goback>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index, Func_goback&& goback_fct)
        {
            int shift = this->ref_level() - this->level();
            m_start_end_op(m_level, m_min_level, m_ref_level);

            return std::apply(
                [this, &index, shift, level, &goback_fct](auto&&... args)
                {
                    return SetOp(
                        shift,
                        get_operator<d>(m_operator),
                        args.template get_local_set<d>(level,
                                                       index,
                                                       m_start_end_op.template goback<d + 1>(std::forward<Func_goback>(goback_fct)))...);
                },
                m_s);
        }

        template <std::size_t d>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index)
        {
            return get_local_set<d>(level, index, default_function());
        }

        template <std::size_t d, class Func_start, class Func_end>
        auto get_start_and_stop_function(Func_start&& start_fct, Func_end&& end_fct)
        {
            m_start_end_op(m_level, m_min_level, m_ref_level);

            return std::apply(
                [this, &start_fct, &end_fct](auto&&... args)
                {
                    return std::make_tuple(std::move(
                        args.template get_start_and_stop_function<d>(m_start_end_op.template start<d>(std::forward<Func_start>(start_fct)),
                                                                     m_start_end_op.template end<d>(std::forward<Func_end>(end_fct))))...);
                },
                m_s);
        }

        template <std::size_t d>
        auto get_start_and_stop_function()
        {
            return get_start_and_stop_function<d>(default_function(), default_function());
        }

        int level() const
        {
            return m_level;
        }

        int ref_level() const
        {
            return m_ref_level;
        }

        void ref_level(auto level)
        {
            m_ref_level = static_cast<int>(level);
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
        int m_ref_level;
        int m_level;
        int m_min_level;
    };

    template <class lca_t>
        requires IsLCA<lca_t>
    struct Self
    {
        static constexpr std::size_t dim = lca_t::dim;
        using interval_t                 = typename lca_t::interval_t;
        using value_t                    = typename interval_t::value_t;

        Self(const lca_t& lca)
            : m_lca(lca)
            , m_level(static_cast<int>(lca.level()))
            , m_ref_level(m_level)
            , m_min_level(m_level)
        {
        }

        template <class Func>
        void operator()(Func&& func)
        {
            apply(*this, std::forward<Func>(func));
        }

        template <std::size_t d, class Func_goback>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index, Func_goback&& goback_fct)
        {
            if (m_lca[d - 1].empty())
            {
                return IntervalVector(IntervalIterator(m_lca[d - 1], 0, 0));
            }

            if constexpr (dim == d)
            {
                return IntervalVector(m_lca.level(),
                                      m_level,
                                      m_ref_level,
                                      IntervalIterator(m_lca[d - 1], 0, static_cast<std::ptrdiff_t>(m_lca[d - 1].size())));
            }
            else
            {
                if (m_lca[d].empty())
                {
                    return IntervalVector(IntervalIterator(m_lca[d - 1], 0, 0));
                }

                auto new_goback_fct = m_func.template goback<d + 1>(std::forward<Func_goback>(goback_fct));

                if (static_cast<std::size_t>(level) >= m_lca.level())
                {
                    auto current_index = start_shift(new_goback_fct(m_level, index[d - 1]), static_cast<int>(m_lca.level()) - m_level);
                    auto j             = find_on_dim(m_lca, d, 0, m_lca[d].size(), current_index);

                    if (j == std::numeric_limits<std::size_t>::max())
                    {
                        return IntervalVector(IntervalIterator(m_lca[d - 1], 0, 0));
                    }

                    auto io       = static_cast<std::size_t>(m_lca[d][j].index + current_index);
                    auto& offsets = m_lca.offsets(d);

                    return IntervalVector(m_lca.level(),
                                          m_level,
                                          m_ref_level,
                                          IntervalIterator(m_lca[d - 1],
                                                           static_cast<std::ptrdiff_t>(offsets[io]),
                                                           static_cast<std::ptrdiff_t>(offsets[io + 1])));
                }
                else
                {
                    auto min_index = start_shift(new_goback_fct(m_level, index[d - 1]), static_cast<int>(m_lca.level()) - m_level);
                    auto max_index = start_shift(new_goback_fct(m_level, index[d - 1] + 1), static_cast<int>(m_lca.level()) - m_level);

                    auto j_min = lower_bound_interval(m_lca[d].begin(), m_lca[d].end(), min_index);
                    auto j_max = upper_bound_interval(j_min, m_lca[d].end(), max_index) - 1;

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
                            return IntervalVector(IntervalIterator(m_lca[d - 1], 0, 0));
                        }
                        start_offset = m_lca.offsets(d)[start_offset];
                        end_offset   = m_lca.offsets(d)[end_offset];
                        ListOfIntervals<value_t> list_of_intervals;
                        // for (std::size_t o = start_offset; o < end_offset; ++o)
                        // {
                        //     auto start = start_shift(m_lca[d - 1][o].start, m_level - static_cast<int>(m_lca.level()));
                        //     auto end   = start_shift(m_lca[d - 1][o].end, m_level - static_cast<int>(m_lca.level()));
                        //     list_of_intervals.add_interval({start, end});
                        // }

                        for (std::size_t o = start_offset; o < end_offset; ++o)
                        {
                            auto start = m_lca[d - 1][o].start;
                            auto end   = m_lca[d - 1][o].end;
                            list_of_intervals.add_interval({start, end});
                        }

                        // std::cout << "offset " << start_offset << " " << end_offset << std::endl;
                        // std::vector<interval_t> intervals;
                        m_work.clear();
                        // intervals.reserve(list_of_intervals.size());
                        for (auto& i : list_of_intervals)
                        {
                            m_work.push_back(i);
                            // std::cout << i << " ";
                        }
                        // std::cout << std::endl;

                        // return IntervalVector(m_level, m_level, m_ref_level, IntervalIterator(m_lca[d - 1], m_work));
                        return IntervalVector(m_lca.level(), m_level, m_ref_level, IntervalIterator(m_lca[d - 1], m_work));
                    }
                    return IntervalVector(IntervalIterator(m_lca[d - 1], 0, 0));
                }
            }
        }

        template <std::size_t d>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index)

        {
            return get_local_set<d>(level, index, default_function());
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

        auto& on(int level)
        {
            m_min_level = std::min(m_min_level, level);
            m_level     = level;
            return *this;
        }

        bool exist() const
        {
            return !m_lca.empty();
        }

        const lca_t& m_lca;
        int m_level;
        int m_ref_level;
        int m_min_level;
        start_end_function<dim> m_func;
        std::vector<interval_t> m_work;
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
                return subset(IntersectionOp(), start_end_function<dim>(), std::forward<decltype(args)>(args)...);
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
                return subset(UnionOp(), start_end_function<dim>(), std::forward<decltype(args)>(args)...);
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
                return subset(DifferenceOp(), start_end_function<dim>(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class set_t, class stencil_t>
    auto translate(set_t&& set, const stencil_t& stencil)
    {
        constexpr std::size_t dim = std::decay_t<set_t>::dim;
        return subset(TranslationOp(),
                      start_end_translate_function<dim>(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)),
                      detail::transform(std::forward<set_t>(set)));
        // return subset(TranslationOp(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)), detail::transform(set));
    }

    template <class lca_t>
    auto self(lca_t&& lca)
    {
        return Self<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
