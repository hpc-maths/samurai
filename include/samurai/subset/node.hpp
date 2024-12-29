// Copyright 2018-2024 the samurai's authors
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

        SetOp(int shift, Operator op, const S&... s)
            : m_shift(shift)
            , m_operator(op)
            , m_s(s...)
        {
        }

        auto shift() const
        {
            return m_shift;
        }

        bool is_in(auto scan) const
        {
            return std::apply(
                [*this, scan](const auto&... args)
                {
                    return m_operator.is_in(scan, args...);
                },
                m_s);
        }

        bool is_empty() const
        {
            return std::apply(
                [*this](const auto&... args)
                {
                    return m_operator.is_empty(args...);
                },
                m_s);
        }

        auto min() const
        {
            return std::apply(
                [](auto&... args)
                {
                    return compute_min(args.min()...);
                },
                m_s);
        }

        void next(auto scan)
        {
            next(scan, m_operator);
        }

        template <class IntervalOp>
        void next(auto scan, IntervalOp op)
        {
            std::apply(
                [scan, &op](auto&... args)
                {
                    (args.next(scan, op), ...);
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

        inline auto start_op(int level, const auto it)
        {
            return it->start + start_shift(m_t[0], level - m_level);
        }

        inline auto end_op(int level, const auto it)
        {
            return it->end + end_shift(m_t[0], level - m_level);
        }

        inline void set_level(auto level)
        {
            m_level = level;
        }

      private:

        xt::xtensor_fixed<int, xt::xshape<dim>> m_t;
        int m_level{std::numeric_limits<int>::min()};
    };

    template <class Op, class... S>
    class subset
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type                   = std::tuple<S...>;
        using interval_t                 = get_interval_t<S...>;

        subset(Op&& op, S&&... s)
            : m_operator(std::forward<Op>(op))
            , m_s(std::forward<S>(s)...)
            , m_ref_level(compute_max(s.ref_level()...))
            , m_level(compute_max(s.level()...))
            , m_min_level(m_level)
        {
            std::apply(
                [*this](auto&&... args)
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
            on_parent(static_cast<int>(level));
            return *this;
        }

        void on_parent(auto level)
        {
            std::apply(
                [level](auto&&... args)
                {
                    (args.on_parent(level), ...);
                },
                m_s);
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

        template <std::size_t d>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index)
        {
            int shift = this->ref_level() - this->level();
            m_operator.set_level(this->level());
            return std::apply(
                [*this, &index, shift, level](auto&&... args)
                {
                    return SetOp(shift, m_operator, args.template get_local_set<d>(level, index)...);
                },
                m_s);
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
                [*this](auto&&... args)
                {
                    (args.ref_level(m_ref_level), ...);
                },
                m_s);
        }

      protected:

        Op m_operator;
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

        template <std::size_t d>
        auto get_local_set(int level, xt::xtensor_fixed<int, xt::xshape<dim - 1>>& index)
        {
            using iterator_t  = decltype(m_lca[d - 1].cbegin());
            using offset_it_t = offset_iterator<iterator_t>;
            if constexpr (dim == d)
            {
                if (m_lca[d - 1].empty())
                {
                    return IntervalVector<offset_it_t>();
                }
                auto begin = offset_it_t(m_lca[d - 1].begin(), m_lca[d - 1].end());
                auto end   = offset_it_t(m_lca[d - 1].end(), m_lca[d - 1].end());
                return IntervalVector<offset_it_t>(m_lca.level(), m_level, m_min_level, m_ref_level, begin, end);
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

        void on_parent(int level)
        {
            m_min_level = std::min(m_min_level, level);
        }

        const lca_t& m_lca;
        int m_level;
        int m_ref_level;
        int m_min_level;
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
                return subset(IntersectionOp(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto union_(sets_t&&... sets)
    {
        return std::apply(
            [](auto&&... args)
            {
                return subset(UnionOp(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto difference(sets_t&&... sets)
    {
        return std::apply(
            [](auto&&... args)
            {
                return subset(DifferenceOp(), std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class set_t, class stencil_t>
    auto translate(const set_t& set, const stencil_t& stencil)
    {
        constexpr std::size_t dim = set_t::dim;
        // return subset(TranslationOp(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)), detail::transform(std::forward<set_t>(set)));
        return subset(TranslationOp(xt::xtensor_fixed<int, xt::xshape<dim>>(stencil)), detail::transform(set));
    }

    template <class lca_t>
    auto self(lca_t&& lca)
    {
        return Self<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
