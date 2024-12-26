// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include "samurai/algorithm.hpp"
#include "samurai/interval.hpp"
#include <concepts>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "crtp.hpp"
#include "interval_interface.hpp"

namespace samurai::experimental
{
    constexpr auto compute_min(auto const& value, auto const&... args)
    {
        if constexpr (sizeof...(args) == 0u) // Single argument case!
        {
            return value;
        }
        else // For the Ts...
        {
            const auto min = compute_min(args...);
            return value < min ? value : min;
        }
    }

    constexpr auto compute_max(auto const& value, auto const&... args)
    {
        if constexpr (sizeof...(args) == 0u) // Single argument case!
        {
            return value;
        }
        else // For the Ts...
        {
            const auto max = compute_max(args...);
            return value > max ? value : max;
        }
    }

    template <class Set, class Func>
    void apply(Set&& global_set, Func&& func)
    {
        auto set = global_set.get_local_set();
        apply(set, std::forward<Func>(func));
    }

    template <class Operator, class... S>
    class SetOp;

    template <class T>
    struct is_setop : std::false_type
    {
    };

    template <class... T>
    struct is_setop<SetOp<T...>> : std::true_type
    {
    };

    template <class T>
    constexpr bool is_setop_v{is_setop<std::decay_t<T>>::value};

    template <typename T>
    concept IsSetOp = is_setop_v<T>;

    template <typename T>
    concept IsIntervalVector = std::is_base_of_v<IntervalVector<typename std::decay_t<T>::iterator_t>, std::decay_t<T>>;

    template <class Set, class Func>
        requires IsSetOp<Set> || IsIntervalVector<Set>
    void apply(Set&& set, Func&& func)
    {
        Interval<int, long long> result;
        int r_ipos = 0;
        set.next(0);
        auto scan = set.min();
        // std::cout << "first scan " << scan << std::endl;

        while (scan < sentinel)
        {
            bool is_in = set.is_in(scan);
            // std::cout << std::boolalpha << "is_in: " << is_in << std::endl;

            if (is_in && r_ipos == 0)
            {
                result.start = scan;
                r_ipos       = 1;
            }
            else if (!is_in && r_ipos == 1)
            {
                result.end = scan;
                r_ipos     = 0;
                // std::cout << result << " " << set.shift() << std::endl;
                auto true_result = result >> static_cast<std::size_t>(set.shift());
                func(true_result);
            }

            set.next(scan);
            scan = set.min();
            // std::cout << "scan " << scan << std::endl;
        }
    }

    template <class Operator, class... S>
    class SetOp
    {
      public:

        using set_type = std::tuple<std::decay_t<S>...>;

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
    };

    struct UnionOp : public detail::IntervalInfo
    {
        bool is_in(auto scan, const auto&... args) const
        {
            return (args.is_in(scan) || ...);
        }
    };

    struct DifferenceOp : public detail::IntervalInfo
    {
        bool is_in(auto scan, const auto& arg, const auto&... args) const
        {
            return arg.is_in(scan) && !(args.is_in(scan) || ...);
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

        inline auto start_op(int level, const auto it)
        {
            return it->start + detail::start_shift(m_t[0], level - m_level);
        }

        inline auto end_op(int level, const auto it)
        {
            return it->end + detail::end_shift(m_t[0], level - m_level);
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

        using set_type = std::tuple<S...>;

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

        auto get_local_set()
        {
            int shift = this->ref_level() - this->level();
            m_operator.set_level(this->level());
            return std::apply(
                [*this, shift](auto&&... args)
                {
                    return SetOp(shift, m_operator, args.get_local_set()...);
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

    template <class T>
    struct is_lca : std::false_type
    {
    };

    template <std::size_t dim, class interval_t>
    struct is_lca<LevelCellArray<dim, interval_t>> : std::true_type
    {
    };

    template <class T>
    using is_lca_v = typename is_lca<std::decay_t<T>>::value;

    template <typename T>
    concept IsLCA = std::same_as<LevelCellArray<T::dim, typename T::interval_t>, T>;

    template <class lca_t>
        requires IsLCA<lca_t>
    struct Identity
    {
        static constexpr std::size_t dim = lca_t::dim;
        using interval_t                 = typename lca_t::interval_t;
        using value_t                    = typename interval_t::value_t;

        Identity(const lca_t& lca)
            : m_lca(lca)
            , m_level(static_cast<int>(lca.level()))
            , m_ref_level(m_level)
            , m_min_level(m_level)
        {
        }

        auto get_local_set()
        {
            return IntervalVector(m_lca.level(), m_level, m_min_level, m_ref_level, m_lca[0].begin(), m_lca[0].end());
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
            return Identity<LevelCellArray<dim, interval_t>>(lca);
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
    auto identity(lca_t&& lca)
    {
        return Identity<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
