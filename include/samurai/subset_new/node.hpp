// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../algorithm.hpp"
#include "concepts.hpp"
#include "interval_interface.hpp"

namespace samurai::experimental
{
    template <class Operator, class... S>
    class SetOp
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type   = std::tuple<S...>;
        using interval_t = get_interval_t<S...>;

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

        TranslationOp(const std::array<int, dim>& t)
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

        std::array<int, dim> m_t;
        int m_level{std::numeric_limits<int>::min()};
    };

    template <class Op, class... S>
    class subset
    {
      public:

        static constexpr std::size_t dim = get_set_dim_v<S...>;
        using set_type   = std::tuple<S...>;
        using interval_t = get_interval_t<S...>;

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
            if (level > m_ref_level)
            {
                ref_level(level);
            }
            m_min_level = std::min(m_min_level, level);
            m_level     = level;
            on_parent(level);
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

        template <std::size_t dim>
        auto get_local_set()
        {
            int shift = this->ref_level() - this->level();
            m_operator.set_level(this->level());
            return std::apply(
                [*this, shift](auto&&... args)
                {
                    return SetOp(shift, m_operator, args.template get_local_set<dim>()...);
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
            m_ref_level = level;
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

        template<std::size_t dim>
        auto get_local_set()
        {
            return IntervalVector(m_lca.level(), m_level, m_min_level, m_ref_level, m_lca[dim-1].begin(), m_lca[dim-1].end());
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

    template <class set_t, std::size_t dim>
    auto translation(set_t&& set, const std::array<int, dim>& t)
    {
        return subset(TranslationOp(t), detail::transform(std::forward<set_t>(set)));
    }

    template <class lca_t>
    auto self(lca_t&& lca)
    {
        return Self<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
