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

namespace samurai::experimental
{
    static constexpr int sentinel = std::numeric_limits<int>::max();

    namespace detail
    {
        template <class T>
        inline T end_shift(T value, T shift)
        {
            return shift >= 0 ? value << shift : ((value - 1) >> -shift) + 1;
        }

        template <class T>
        inline T start_shift(T value, T shift)
        {
            return shift >= 0 ? value << shift : value >> -shift;
        }

    } // namespace detail

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
    void apply(Set&& set, Func&& func)
    {
        set.reset();
        Interval<int, long long> result;
        int r_ipos = 0;
        auto scan  = set.min();
        // std::cout << "first scan " << scan << std::endl;

        while (scan < sentinel)
        {
            bool is_in = set.is_in(0, scan);
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
                // std::cout << result << " " << set.ref_level() << " " << set.level() << std::endl;
                auto true_result = result >> (set.ref_level() - set.level());
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

        auto min()
        {
            return std::apply(
                [](auto&&... args)
                {
                    return compute_min(args.min()...);
                },
                m_s);
        }

        void reset()
        {
            std::apply(
                [](auto&&... args)
                {
                    (args.reset(), ...);
                },
                m_s);
        }

        void next(auto scan)
        {
            return std::apply(
                [scan](auto&&... args)
                {
                    (args.next(scan), ...);
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

        Identity(const lca_t& lca)
            : m_lca(lca)
            , m_level(static_cast<int>(lca.level()))
            , m_ref_level(m_level)
            , m_min_level(m_level)
        {
        }

        void reset()
        {
            m_d = dim - 1;
            if (m_lca[m_d].empty())
            {
                m_current[m_d] = sentinel;
                return;
            }
            m_is_start.fill(0);
            m_index[m_d]   = 0;
            m_offset[m_d]  = m_lca[m_d].size();
            int min_shift  = m_min_level - static_cast<int>(m_lca.level());
            int ref_shift  = m_ref_level - m_min_level;
            m_current[m_d] = detail::start_shift(detail::start_shift(m_lca[m_d][m_index[m_d]].start, min_shift), ref_shift);
        }

        bool is_in(std::size_t, auto scan) const
        {
            return m_current[m_d] != sentinel && !((scan < m_current[m_d]) ^ m_is_start[m_d]);
        }

        auto min()
        {
            return m_current[m_d];
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

        void next(auto scan)
        {
            if (m_current[m_d] == scan)
            {
                int min_shift = m_min_level - static_cast<int>(m_lca.level());
                int ref_shift = m_ref_level - m_min_level;
                // std::cout << min_shift << " " << ref_shift << std::endl;
                if (m_is_start[m_d] == 0)
                {
                    auto end       = m_lca[m_d][m_index[m_d]].end;
                    auto start     = m_lca[m_d][m_index[m_d] + 1].start;
                    m_current[m_d] = detail::end_shift(detail::end_shift(end, min_shift), ref_shift);
                    while (m_index[m_d] < m_offset[m_d] - 1
                           && m_current[m_d] >= detail::start_shift(detail::start_shift(start, min_shift), ref_shift))
                    {
                        m_index[m_d]++;
                        end            = m_lca[m_d][m_index[m_d]].end;
                        m_current[m_d] = detail::end_shift(detail::end_shift(end, min_shift), ref_shift);
                        start          = m_lca[m_d][m_index[m_d] + 1].start;
                        // std::cout << m_current[m_d] << std::endl;
                    }
                    m_is_start[m_d] = 1;
                    // std::cout << "finish" << std::endl;
                }
                else
                {
                    m_index[m_d]++;
                    if (m_index[m_d] == m_offset[m_d])
                    {
                        m_current[m_d] = std::numeric_limits<value_t>::max();
                        return;
                    }
                    m_current[m_d]  = detail::start_shift(detail::start_shift(m_lca[m_d][m_index[m_d]].start, min_shift), ref_shift);
                    m_is_start[m_d] = 0;
                }
                // std::cout << "m_current: " << m_current[m_d] << " " << m_is_start[m_d] << " " << m_offset[m_d] << std::endl;
            }
        }

        const lca_t& m_lca;
        std::array<value_t, dim> m_current;
        std::array<int, dim> m_is_start;
        std::array<std::size_t, dim> m_index;
        std::array<std::size_t, dim> m_offset;
        std::size_t m_d;
        int m_level;
        int m_ref_level;
        int m_min_level;
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

        bool is_in(std::size_t d, auto scan) const
        {
            return std::apply(
                [d, scan](const auto&... args)
                {
                    return (args.is_in(d, scan) && ...);
                },
                this->m_s);
        }
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

        bool is_in(std::size_t d, auto scan) const
        {
            return std::apply(
                [d, scan, this](const auto&... args)
                {
                    return (args.is_in(d, scan) || ...);
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
