// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "assert_log_trace.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace samurai
{
    template <class value_t, class index_t>
    struct Interval;
}

template <typename T, typename Compare>
inline void sort_indexes(const std::vector<T>& v, Compare cmp, std::vector<size_t>& idx)
{
    // initialize original index locations
    idx.resize(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(idx.begin(),
                     idx.end(),
                     [&v, &cmp](const size_t i1, const size_t i2) -> bool
                     {
                         return cmp(v[i1], v[i2]);
                     });
}

template <typename T, typename Compare>
inline void sort_indexes(const std::vector<T>& v, std::vector<size_t>& idx)
{
    sort_indexes(v, std::less{}, idx);
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class value_t, class index_t>
struct std::hash<samurai::Interval<value_t, index_t>>
{
    std::size_t operator()(const samurai::Interval<value_t, index_t>& s) const noexcept
    {
        std::size_t seed = 0;
        ::hash_combine(seed, s.start);
        ::hash_combine(seed, s.end);
        ::hash_combine(seed, s.step);
        ::hash_combine(seed, s.index);
        return seed;
    }
};

template <class... T>
struct std::hash<std::tuple<T...>>
{
    template <std::size_t... I>
    std::size_t hash_tuple(const std::tuple<T...>& s, std::integer_sequence<std::size_t, I...>) const noexcept
    {
        std::size_t seed = 0;
        (::hash_combine(seed, std::get<I>(s)), ...);
        return seed;
    }

    std::size_t operator()(const std::tuple<T...>& s) const noexcept
    {
        return hash_tuple(s, std::index_sequence_for<T...>{});
    }
};

namespace samurai
{
    template <class T>
    class subset_node;

    template <std::size_t d>
    using Dim = std::integral_constant<std::size_t, d>;

    template <class... T>
    using void_t = void;

    template <class F, class... CT>
    class field_function;

    template <template <std::size_t dim, class T> class OP, class... CT>
    class field_operator_function;

    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA = false>
    class VectorField;

    template <class mesh_t, class value_t>
    class ScalarField;

    namespace detail
    {

        template <int S1, int S2>
        struct check_dim : std::false_type
        {
        };

        template <int S>
        struct check_dim<S, S> : std::true_type
        {
        };

        template <int S>
        struct check_dim<-1, S> : std::true_type
        {
        };

        template <typename T, typename = void>
        struct has_dim : std::false_type
        {
        };

        template <typename T>
        struct has_dim<T, void_t<decltype(T::dim)>> : std::true_type
        {
        };

        template <int S, class T, bool = has_dim<T>::value>
        struct get_dim
        {
        };

        template <int S, class T>
        struct get_dim<S, T, false>
        {
            static constexpr int dim = S;
        };

        template <int S, class T>
        struct get_dim<S, T, true>
        {
            static_assert(check_dim<S, T::dim>::value, "dim must be the same for all nodes");
            static constexpr int dim = T::dim;
        };

        template <int S, class... CT>
        struct compute_dim_impl
        {
        };

        template <int S>
        struct compute_dim_impl<S>
        {
            static constexpr int dim = S;
        };

        template <int S, class C0, class... CT>
        struct compute_dim_impl<S, C0, CT...>
        {
            static constexpr int dim = compute_dim_impl<get_dim<S, C0>::dim, CT...>::dim;
        };

        template <class... CT>
        constexpr std::size_t compute_dim()
        {
            return compute_dim_impl<-1, std::decay_t<CT>...>::dim;
        }

        template <class T, class = void>
        struct has_mesh_t : std::false_type
        {
        };

        template <class T>
        struct has_mesh_t<T, std::void_t<typename T::mesh_t>> : std::true_type
        {
        };

        template <class S, class T, bool>
        struct get_mesh_t;

        template <class S, class T>
        struct get_mesh_t<S, T, false>
        {
            using type = S;
        };

        template <class S, class T>
        struct get_mesh_t<S, T, true>
        {
            using type = typename T::mesh_t;
        };

        template <class S, class... CT>
        struct compute_mesh_impl
        {
            using type = S;
        };

        template <class M>
        struct compute_mesh_impl<M>
        {
            using type = M;
        };

        template <class M, class C0, class... CT>
        struct compute_mesh_impl<M, C0, CT...>
        {
            using type = typename compute_mesh_impl<typename get_mesh_t<M, C0, has_mesh_t<C0>::value>::type, CT...>::type;
        };

        template <class... CT>
        struct compute_mesh_t
        {
            using type = typename compute_mesh_impl<void, std::decay_t<CT>...>::type;
        };

        template <class E>
        struct is_field_function : std::false_type
        {
        };

        template <class F, class... CT>
        struct is_field_function<field_function<F, CT...>> : std::true_type
        {
        };

        template <template <std::size_t dim, class T> class OP, class... CT>
        struct is_field_function<field_operator_function<OP, CT...>> : std::true_type
        {
        };

        template <class... T>
        auto& extract_mesh(const std::tuple<T...>& t)
        {
            return extract_mesh(t, std::index_sequence_for<T...>());
        }

        template <class... T, std::size_t... Is>
        auto& extract_mesh(const std::tuple<T...>& t, std::index_sequence<Is...>)
        {
            return extract_mesh(std::get<Is>(t)...);
        }

        template <class... T>
        constexpr std::size_t compute_n_comp()
        {
            return (0 + ... + T::n_comp);
        }

        template <class... T>
        struct common_type
        {
            using type = std::common_type_t<typename T::value_type...>;
        };

        template <class... T>
        using common_type_t = typename common_type<T...>::type;

        template <class Head, class... Tail>
        auto& extract_mesh(Head&& h, Tail&&... t)
        {
            if constexpr (is_field_function<std::decay_t<Head>>::value)
            {
                return extract_mesh(h.arguments());
            }
            else if constexpr (has_mesh_t<std::decay_t<Head>>::value)
            {
                return h.mesh();
            }
            else
            {
                return extract_mesh(std::forward<Tail>(t)...);
            }
        }

        template <class T>
        constexpr T do_max(const T& v)
        {
            return v;
        }

        template <class C0, class... CT>
        struct interval_type
        {
            using check = std::disjunction<
                std::is_same<typename std::remove_reference<C0>::type::interval_t, typename std::remove_reference<CT>::type::interval_t>...>;
            static_assert(check::value, "interval type must be the same");
            using type = typename std::remove_reference<C0>::type::interval_t;
        };

        template <class T1, class T2, class... Rest>
        constexpr typename std::common_type<T1, T2, Rest...>::type do_max(T1 const& v0, T2 const& v1, const Rest&... rest)
        {
            return do_max(v0 < v1 ? v1 : v0, rest...);
        }

        template <class T>
        constexpr T do_min(const T& v)
        {
            return v;
        }

        template <class T1, class T2, class... Rest>
        constexpr typename std::common_type<T1, T2, Rest...>::type do_min(T1 const& v0, T2 const& v1, const Rest&... rest)
        {
            return do_min(v0 < v1 ? v0 : v1, rest...);
        }

        /**
         * @brief test if template parameter is SOA of AOS Field (false by default like VectorField)
         *
         * @tparam T type to test
         */
        template <class T>
        struct is_soa : std::false_type
        {
        };

        // specialization for VectorField
        template <class Mesh, class value_t, std::size_t n_comp, bool SOA>
        struct is_soa<VectorField<Mesh, value_t, n_comp, SOA>> : std::bool_constant<SOA>
        {
        };

        /**
         * @brief helper to get value of is_soa
         *
         * @tparam T type to test
         */
        template <class T>
        inline constexpr bool is_soa_v = is_soa<std::decay_t<T>>::value;

        /**
         * @brief test if template parameter is a samurai field (ScalarField or VectorField)
         *
         * @tparam T type to test
         */
        template <class T>
        struct is_field_type : std::false_type
        {
        };

        // specialization for VectorField
        template <class Mesh, class value_t, std::size_t n_comp, bool SOA>
        struct is_field_type<VectorField<Mesh, value_t, n_comp, SOA>> : std::true_type
        {
        };

        // specialization for ScalarField
        template <class Mesh, class value_t>
        struct is_field_type<ScalarField<Mesh, value_t>> : std::true_type
        {
        };

        /**
         * @brief helper to get value of is_field_type
         *
         * @tparam T type to test
         */
        template <class T>
        inline constexpr bool is_field_type_v = is_field_type<std::decay_t<T>>::value;

    } // namespace detail

    template <class R, class T1, class T2>
    R safe_subs(T1 a, T2 b)
    {
        return static_cast<R>(static_cast<std::ptrdiff_t>(a) - static_cast<std::ptrdiff_t>(b));
    }

    template <class Field, class index_t>
    inline auto& field_value(Field& f, const typename Field::cell_t& cell, [[maybe_unused]] index_t field_i)
    {
        return field_value(f, cell.index, field_i);
    }

    template <class Field, class index_t>
    inline auto& field_value(Field& f, const typename Field::index_t& cell_index, [[maybe_unused]] index_t field_i)
    {
        using size_type = typename Field::size_type;
        if constexpr (Field::is_scalar)
        {
            return f[static_cast<size_type>(cell_index)];
        }
        else
        {
            return f[static_cast<size_type>(cell_index)][field_i];
        }
    }

    template <class LCA>
    auto get_periodic_shift(const LCA& domain, std::size_t level, std::size_t d)
    {
        static constexpr std::size_t dim = LCA::dim;
        using interval_value_t           = typename LCA::interval_t::value_t;

        const auto& min_indices   = domain.min_indices();
        const auto& max_indices   = domain.max_indices();
        const std::size_t delta_l = domain.level() - level;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;
        shift.fill(0);
        shift[d] = (max_indices[d] - min_indices[d]) >> delta_l;
        return shift;
    }

    // template <class Field>
    // inline auto&
    // field_value(typename Field::value_type* data, const typename Field::index_t& cell_index, [[maybe_unused]] std::size_t field_i)
    // {
    //     if constexpr (Field::is_scalar)
    //     {
    //         return *data[cell_index];
    //     }
    //     else if constexpr (detail::is_soa_v<Field>)
    //     {
    //         static_assert(Field::is_scalar || !detail::is_soa_v<Field>, "field_value() is not implemented for SOA fields");
    //         return *data[field_i /*  *n_cells */ + cell_index];
    //     }
    //     else
    //     {
    //         return *data[cell_index * Field::n_comp + field_i];
    //     }
    // }

    //------------------------------//
    // Greater Common Divisor (GCD) //
    //------------------------------//

    // Function to compute the greatest common divisor (GCD) of two integers
    template <typename T>
        requires std::integral<T>
    T gcd_int(T a, T b)
    {
        while (b != 0)
        {
            T temp = b;
            b      = a % b;
            a      = temp;
        }
        return a;
    }

    // Function to compute the GCD of two floating-point values
    template <typename T>
        requires std::floating_point<T>
    T gcd_float(T a, T b)
    {
        if (a == 0.0 && b == 0.0)
        {
            return 0.0; // GCD of 0 and 0 is 0
        }

        if (a == 0.0)
        {
            return std::abs(b); // GCD of 0 and b is |b|
        }

        if (b == 0.0)
        {
            return std::abs(a); // GCD of a and 0 is |a|
        }

        // Scale the floating-point values to integers by finding a common denominator
        int scale_a = 0;
        int scale_b = 0;

        T temp_a = std::abs(a);
        T temp_b = std::abs(b);

        while (std::floor(temp_a) != temp_a)
        {
            temp_a *= 10;
            scale_a++;
        }

        while (std::floor(temp_b) != temp_b)
        {
            temp_b *= 10;
            scale_b++;
        }

        int scale = std::max(scale_a, scale_b);

        // Use the largest integer type to avoid overflow
        using IntegerType = long long;

        IntegerType int_a = static_cast<IntegerType>(std::abs(a) * std::pow(10, scale));
        IntegerType int_b = static_cast<IntegerType>(std::abs(b) * std::pow(10, scale));

        // Compute the GCD of the scaled integers
        IntegerType int_gcd = gcd_int(int_a, int_b);

        // Scale the GCD back to the original scale
        return static_cast<T>(int_gcd) / std::pow(10, scale);
    }

} // namespace samurai
