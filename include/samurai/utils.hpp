#pragma once

#include <algorithm>
#include <type_traits>

namespace samurai
{
    template<class T>
    class subset_node;

    template<std::size_t d>
    using Dim = std::integral_constant<std::size_t, d>;

    template<class... T>
    using void_t = void;

    namespace detail
    {

        template<int S1, int S2>
        struct check_dim : std::false_type
        {
        };

        template<int S>
        struct check_dim<S, S> : std::true_type
        {
        };

        template<int S>
        struct check_dim<-1, S> : std::true_type
        {
        };

        template<typename T, typename = void>
        struct has_dim : std::false_type
        {
        };

        template<typename T>
        struct has_dim<T, void_t<decltype(T::dim)>> : std::true_type
        {
        };

        template<int S, class T, bool = has_dim<T>::value>
        struct get_dim
        {
        };

        template<int S, class T>
        struct get_dim<S, T, false>
        {
            static constexpr int dim = S;
        };

        template<int S, class T>
        struct get_dim<S, T, true>
        {
            static_assert(check_dim<S, T::dim>::value,
                          "dim must be the same for all nodes");
            static constexpr int dim = T::dim;
        };

        template<int S, class... CT>
        struct compute_dim_impl
        {
        };

        template<int S>
        struct compute_dim_impl<S>
        {
            static constexpr int dim = S;
        };

        template<int S, class C0, class... CT>
        struct compute_dim_impl<S, C0, CT...>
        {
            static constexpr int dim =
                compute_dim_impl<get_dim<S, C0>::dim, CT...>::dim;
        };

        template<class... CT>
        constexpr std::size_t compute_dim()
        {
            return compute_dim_impl<-1, std::decay_t<CT>...>::dim;
        }

        template<class T>
        constexpr T const &do_max(T const &v)
        {
            return v;
        }

        template<class C0, class... CT>
        struct interval_type
        {
            using check = std::is_same<
                typename std::remove_reference<C0>::type::interval_t,
                typename std::remove_reference<CT>::type::interval_t...>;
            static_assert(check::value, "interval type must be the same");
            using type = typename std::remove_reference<C0>::type::interval_t;
        };

        template<class T1, class T2, class... Rest>
        constexpr typename std::common_type<T1, T2, Rest...>::type const &
        do_max(T1 const &v0, T2 const &v1, Rest const &... rest)
        {
            return do_max(v0 < v1 ? v1 : v0, rest...);
        }

        template<class T>
        constexpr T const &do_min(T const &v)
        {
            return v;
        }

        template<class T1, class T2, class... Rest>
        constexpr typename std::common_type<T1, T2, Rest...>::type const &
        do_min(T1 const &v0, T2 const &v1, Rest const &... rest)
        {
            return do_min(v0 < v1 ? v0 : v1, rest...);
        }
    }
}