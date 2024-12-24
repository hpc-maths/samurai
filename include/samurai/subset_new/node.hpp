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
        Interval<int, long long> result;
        int r_ipos = 0;
        auto scan  = set.min();
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
                auto true_result = result >> set.shift();
                func(true_result);
            }

            set.next(scan);
            scan = set.min();
            // std::cout << "scan " << scan << std::endl;
        }
    }

    template <class... S>
    class subset
    {
        using set_type = std::tuple<std::decay_t<S>...>;

      public:

        subset(S&&... s)
            : m_s(std::forward<S>(s)...)
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

        void on(auto level)
        {
            if (level > m_ref_level)
            {
                ref_level(level);
            }
            m_min_level = std::min(m_min_level, level);
            m_level     = level;
            on_parent(level);
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
            return std::apply(
                [](auto&&... args)
                {
                    (args.get_local_set(), ...);
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

        Identity(lca_t& lca)
            : m_lca(lca)
            , m_level(static_cast<int>(lca.level()))
            , m_ref_level(m_level)
            , m_min_level(m_level)
        {
        }

        auto get_local_set() const
        {
            return IntervalVector<std::decay_t<decltype(m_lca[0])>>(m_lca.level(),
                                                                    m_level,
                                                                    m_min_level,
                                                                    m_ref_level,
                                                                    m_lca[0].begin(),
                                                                    m_lca[0].end());
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

        lca_t& m_lca;
        int m_level;
        int m_ref_level;
        int m_min_level;
    };

    template <class D>
    struct SetOp : public crtp_base<D>
    {
        bool is_in(auto scan) const
        {
            return this->derived_cast().is_in_impl(scan);
        }

        auto min() const
        {
            return std::apply(
                [](auto&... args)
                {
                    return compute_min(args.min()...);
                },
                this->derived_cast().get_elements());
        }

        void next(auto scan)
        {
            std::apply(
                [scan](auto&... args)
                {
                    (args.next(scan), ...);
                },
                this->derived_cast().get_elements());
        }
    };

    template <class... S>
    class IntersectionOp : public SetOp<IntersectionOp<S...>>
    {
      public:

        using set_type = std::tuple<std::decay_t<S>...>;

        IntersectionOp(int shift, const S&... s)
            : m_shift(shift)
            , m_s(s...)
        {
        }

        auto shift() const
        {
            return m_shift;
        }

        bool is_in_impl(auto scan) const
        {
            return std::apply(
                [scan](const auto&... args)
                {
                    return (args.is_in(scan) && ...);
                },
                m_s);
        }

        const auto& get_elements() const
        {
            return m_s;
        }

        auto& get_elements()
        {
            return m_s;
        }

      private:

        int m_shift;
        set_type m_s;
    };

    template <class... S>
    struct Intersection : public subset<S...>

    {
        using base = subset<S...>;
        using base::base;

        auto& on(auto level)
        {
            base::on(level);
            return *this;
        }

        auto get_local_set() const
        {
            int shift = this->ref_level() - this->level();
            return std::apply(
                [shift](auto&... args)
                {
                    return IntersectionOp(shift, args.get_local_set()...);
                },
                this->m_s);
        }
    };

    template <class... S>
    class UnionOp : public SetOp<UnionOp<S...>>
    {
      public:

        using set_type = std::tuple<std::decay_t<S>...>;

        UnionOp(int shift, const S&... s)
            : m_shift(shift)
            , m_s(s...)
        {
        }

        auto shift() const
        {
            return m_shift;
        }

        bool is_in_impl(auto scan) const
        {
            return std::apply(
                [scan](const auto&... args)
                {
                    return (args.is_in(scan) || ...);
                },
                m_s);
        }

        const auto& get_elements() const
        {
            return m_s;
        }

        auto& get_elements()
        {
            return m_s;
        }

      private:

        int m_shift;
        set_type m_s;
    };

    template <class... S>
    struct Union : public subset<S...>

    {
        using base = subset<S...>;
        using base::base;

        auto& on(auto level)
        {
            base::on(level);
            return *this;
        }

        auto get_local_set() const
        {
            int shift = this->ref_level() - this->level();
            return std::apply(
                [shift](auto&... args)
                {
                    return UnionOp(shift, args.get_local_set()...);
                },
                this->m_s);
        }
    };

    template <class... S>
    struct Difference : public subset<S...>
    {
        bool is_in(std::size_t d, auto scan) const
        {
            if (d == 0)
            {
                return std::apply(
                    [d, scan, this](const auto& arg, const auto&... args)
                    {
                        return arg.is_in(d, scan) && !(args.is_in(d, scan) || ...);
                    },
                    this->m_s);
            }
            return std::apply(
                [d, scan, this](const auto&... args)
                {
                    return (args.is_in(d, scan) || ...);
                },
                this->m_s);
        }
    };

    template <class... lca_t>
        requires(IsLCA<lca_t> && ...)
    auto make_intersection(const lca_t&... lca)
    {
        return Intersection<Identity<lca_t>...>(Identity(lca)...);
    }

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
                return Intersection<std::decay_t<decltype(args)>...>(std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class... sets_t>
    auto union_(sets_t&&... sets)
    {
        return std::apply(
            [](auto&&... args)
            {
                return Union<std::decay_t<decltype(args)>...>(std::forward<decltype(args)>(args)...);
            },
            std::make_tuple(detail::transform(std::forward<sets_t>(sets))...));
    }

    template <class lca_t>
    auto identity(lca_t&& lca)
    {
        return Identity<std::decay_t<lca_t>>(std::forward<lca_t>(lca));
    }
}
