// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtensor/xfixed.hpp>

#include "../algorithm.hpp"
#include "concepts.hpp"
#include "interval_interface.hpp"

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

        void next(auto scan)
        {
            std::apply(
                [scan](auto&&... args)
                {
                    (args.next(scan), ...);
                },
                m_s);
        }

      private:

        int m_shift;
        Operator m_operator;
        set_type m_s;
    };

    struct IntersectionOp : public detail::IntervalInfo
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) && ...);
        }

        bool is_empty(const auto&... args) const
        {
            return (args.is_empty() || ...);
        }
    };

    struct UnionOp : public detail::IntervalInfo
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) || ...);
        }

        bool is_empty(const auto&... args) const
        {
            return (args.is_empty() && ...);
        }
    };

    struct DifferenceOp : public detail::IntervalInfo
    {
        bool is_in(auto scan, const auto& arg, const auto&... args) const
        {
            return arg.is_in(scan) && !(args.is_in(scan) || ...);
        }

        bool is_empty(const auto& arg, const auto&...) const
        {
            return arg.is_empty();
        }
    };

    template <std::size_t dim>
    class TranslationOp
    {
      public:

        TranslationOp(const xt::xtensor_fixed<int, xt::xshape<dim>>& t)
            : m_t(t)
        {
        }

        bool is_in(auto scan, const auto& arg) const
        {
            return arg.is_in(scan);
        }

        bool is_empty(const auto& arg) const
        {
            return arg.is_empty();
        }

        inline auto get_func(auto level, auto min_level, auto max_level, auto d) const
        {
            return start_end_translate_function(level, min_level, max_level, m_t[d]);
        }

      private:

        xt::xtensor_fixed<int, xt::xshape<dim>> m_t;
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

        template <std::size_t d, class Func_start, class Func_end>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index, Func_start&& start_fct, Func_end&& end_fct)
        {
            int shift      = this->ref_level() - this->level();
            m_start_end_op = m_operator.get_func(m_level, m_min_level, m_ref_level, d);

            return std::apply(
                [this, &index, shift, level, &start_fct, &end_fct](auto&&... args)
                {
                    return SetOp(shift,
                                 m_operator,
                                 args.template get_local_set<d>(level,
                                                                index,
                                                                m_start_end_op.start(std::forward<Func_start>(start_fct)),
                                                                m_start_end_op.end(std::forward<Func_end>(end_fct)))...);
                },
                m_s);
        }

        template <std::size_t d>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index)
        {
            return get_local_set<d>(level, index, default_function(), default_function());
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

        template <std::size_t d, class Func_start, class Func_end>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index, Func_start&& start_fct, Func_end&& end_fct)
        {
            using iterator_t  = decltype(m_lca[d - 1].cbegin());
            using offset_it_t = offset_iterator<iterator_t>;

            m_func             = start_end_function(m_level, m_min_level, m_ref_level);
            auto new_start_fct = m_func.start(std::forward<Func_start>(start_fct));
            auto new_end_fct   = m_func.end(std::forward<Func_end>(end_fct));

            if constexpr (dim == d)
            {
                if (m_lca[d - 1].empty())
                {
                    // return IntervalVector<offset_it_t>();
                    return IntervalVector<iterator_t>();
                }
                // auto begin = offset_it_t(m_lca[d - 1].begin(), m_lca[d - 1].end());
                // auto end   = offset_it_t(m_lca[d - 1].end(), m_lca[d - 1].end());
                // return IntervalVector<offset_it_t>(m_lca.level(), m_level, m_min_level, m_ref_level, begin, end);
                return IntervalVector<iterator_t>(m_lca.level(),
                                                  m_level,
                                                  m_min_level,
                                                  m_ref_level,
                                                  m_lca[d - 1].begin(),
                                                  m_lca[d - 1].end(),
                                                  new_start_fct,
                                                  new_end_fct);
            }
            else
            {
                if (static_cast<std::size_t>(level) >= m_lca.level())
                {
                    auto current_index = index[d - 1] >> (static_cast<std::size_t>(level) - m_lca.level());
                    auto j             = samurai::find_on_dim(m_lca, d, 0, m_lca[d].size(), current_index);

                    std::cout << "j " << j << std::endl;
                    if (j == std::numeric_limits<std::size_t>::max())
                    {
                        return IntervalVector<offset_it_t>();
                    }

                    // std::cout << "j " << j << " index: " << m_lca[d][j].index << " " << current_index << std::endl;
                    auto io       = static_cast<std::size_t>(m_lca[d][j].index + current_index);
                    auto& offsets = m_lca.offsets(d);
                    // std::cout << io << " " << offsets[io] << " " << offsets[io + 1] << std::endl;
                    auto begin = offset_it_t(m_lca[d - 1].begin() + static_cast<std::ptrdiff_t>(offsets[io]),
                                             m_lca[d - 1].begin() + static_cast<std::ptrdiff_t>(offsets[io + 1]));
                    auto end   = offset_it_t(m_lca[d - 1].begin() + static_cast<std::ptrdiff_t>(offsets[io + 1]),
                                           m_lca[d - 1].begin() + static_cast<std::ptrdiff_t>(offsets[io + 1]));
                    return IntervalVector<offset_it_t>(m_lca.level(), m_level, m_min_level, m_ref_level, begin, end);
                }
                else
                {
                    std::cout << index[d - 1] << std::endl;
                    auto min_index = index[d - 1] << (m_lca.level() - static_cast<std::size_t>(level));
                    auto max_index = (index[d - 1] + 1) << (m_lca.level() - static_cast<std::size_t>(level));

                    auto comp = [](const auto& interval, auto v)
                    {
                        return interval.end < v;
                    };

                    auto j_min = std::lower_bound(m_lca[d].begin(), m_lca[d].end(), min_index, comp);
                    auto j_max = std::lower_bound(j_min, m_lca[d].end(), max_index, comp);

                    std::cout << min_index << " " << max_index << std::endl;
                    if (j_min != m_lca[d].end())
                    {
                        // std::cout << *j_min << " " << *j_max << std::endl;
                        std::size_t start_offset = 0;
                        auto ii                  = min_index;
                        do
                        {
                            if (j_min->contains(ii))
                            {
                                start_offset = static_cast<std::size_t>(j_min->index + ii);
                                break;
                            }
                            ii++;
                        } while (ii < max_index);

                        std::size_t end_offset = 0;
                        ii                     = max_index;
                        do
                        {
                            if (j_max->contains(ii - 1))
                            {
                                end_offset = static_cast<std::size_t>(j_max->index + ii);
                                break;
                            }
                            ii--;
                        } while (ii > min_index);
                        // std::cout << "offset " << start_offset << " " << end_offset << std::endl;

                        std::vector<iterator_t> obegin(end_offset - start_offset);
                        std::vector<iterator_t> oend(end_offset - start_offset);
                        for (std::size_t o = start_offset; o < end_offset; ++o)
                        {
                            obegin.emplace_back(m_lca[d - 1].cbegin() + static_cast<std::ptrdiff_t>(m_lca.offsets(d)[o]));
                            oend.emplace_back(m_lca[d - 1].cbegin() + static_cast<std::ptrdiff_t>(m_lca.offsets(d)[o + 1]));
                        }
                        auto begin = offset_it_t(obegin, oend);
                        auto end   = offset_it_t(oend, oend);
                        return IntervalVector<offset_it_t>(m_lca.level(), m_level, m_min_level, m_ref_level, begin, end);
                    }
                    return IntervalVector<offset_it_t>();
                }
            }
        }

        template <std::size_t d>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index)
        {
            return get_local_set<d>(level, index, default_function(), default_function());
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

        // void on_parent(int level)
        // {
        //     m_min_level = std::min(m_min_level, level);
        // }

        const lca_t& m_lca;
        int m_level;
        int m_ref_level;
        int m_min_level;
        start_end_function m_func;
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
        return std::apply(
            [](auto&&... args)
            {
                return subset(IntersectionOp(), start_end_function(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto union_(sets_t&&... sets)
    {
        return std::apply(
            [](auto&&... args)
            {
                return subset(UnionOp(), start_end_function(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto difference(sets_t&&... sets)
    {
        return std::apply(
            [](auto&&... args)
            {
                return subset(DifferenceOp(), start_end_function(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class set_t, class stencil_t>
    auto translate(set_t&& set, const stencil_t& stencil)
    {
        constexpr std::size_t dim = std::decay_t<set_t>::dim;
        return subset(TranslationOp(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)),
                      start_end_translate_function(),
                      detail::transform(std::forward<set_t>(set)));
        // return subset(TranslationOp(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)), detail::transform(set));
    }

    template <class lca_t>
    auto self(lca_t&& lca)
    {
        return Self<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
