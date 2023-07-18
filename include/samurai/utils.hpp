// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <algorithm>
#include <functional>
#include <tuple>
#include <type_traits>

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

    template <template <class T> class OP, class... CT>
    class field_operator_function;

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

        template <template <class T> class OP, class... CT>
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
        constexpr std::size_t compute_size()
        {
            return (0 + ... + T::size);
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
    } // namespace detail

    template <class R, class T1, class T2>
    R safe_subs(T1 a, T2 b)
    {
        return static_cast<R>(static_cast<std::ptrdiff_t>(a) - static_cast<std::ptrdiff_t>(b));
    }

    template <class Field>
    auto& field_value(Field& f, typename Field::cell_t& cell, [[maybe_unused]] std::size_t field_i)
    {
        if constexpr (Field::size == 1)
        {
            return f[cell];
        }
        else
        {
            return f[cell][field_i];
        }
    }

} // namespace samurai
