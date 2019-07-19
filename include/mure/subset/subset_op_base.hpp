#pragma once

#include <algorithm>
#include <tuple>
#include <type_traits>

#include "../level_cell_array.hpp"
#include "subset_node.hpp"

namespace mure
{
    namespace detail
    {
        template<std::size_t S1, size_t S2>
        struct check_dim : std::false_type
        {
        };

        template<std::size_t S>
        struct check_dim<S, S> : std::true_type
        {
        };

        template<std::size_t, class... CT>
        struct compute_dim_impl;

        template<std::size_t S>
        struct compute_dim_impl<S>
        {
            static constexpr std::size_t dim = S;
        };

        template<size_t S, class C1, class... CT>
        struct compute_dim_impl<S, C1, CT...>
        {
            static_assert(
                check_dim<S, std::remove_reference<C1>::type::dim>::value,
                "dim must be the same for all nodes");
            static constexpr std::size_t dim = compute_dim_impl<S, CT...>::dim;
        };

        template<class C0, class... CT>
        constexpr std::size_t compute_dim()
        {
            return compute_dim_impl<std::remove_reference<C0>::type::dim,
                                    CT...>::dim;
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

    template<class F, class... CT>
    class subset_operator;

    template<class CT>
    struct make_projection
    {
    };

    template<class Func, class... CT>
    struct make_projection<subset_operator<Func, CT...>>
    {
        using type = subset_operator<
            Func, typename make_projection<std::decay_t<CT>>::type...>;

        template<std::size_t... I>
        static type apply_impl(std::size_t ref_level,
                               const subset_operator<Func, CT...> &t,
                               std::index_sequence<I...>)
        {
            return type(
                Func(),
                make_projection<std::decay_t<
                    typename std::tuple_element<I, std::tuple<CT...>>::type>>::
                    apply(ref_level, std::get<I>(t.get_tuple()))...);
        }

        static type apply(std::size_t ref_level,
                          const subset_operator<Func, CT...> &t)
        {
            return apply_impl(ref_level, t,
                              std::make_index_sequence<sizeof...(CT)>());
        }
    };

    template<class T>
    struct make_projection<subset_node<T>>
    {
        using type = subset_node<projection_op<T>>;

        static type apply(std::size_t ref_level, const subset_node<T> &t)
        {
            return projection_op<T>(ref_level, t.get_node());
        }
    };

    template<class CT>
    using make_projection_t = typename make_projection<CT>::type;

    /******************************
     * subset_operator definition *
     ******************************/

    template<class F, class... CT>
    class subset_operator {
      public:
        using functor_type = F;
        using tuple_type = std::tuple<CT...>;

        static constexpr std::size_t dim = detail::compute_dim<CT...>();
        using interval_t = typename detail::interval_type<CT...>::type;
        using coord_index_t = typename interval_t::value_t;

        template<class Func, class... CTA>
        subset_operator(Func &&f, CTA &&... e)
            : m_e(std::forward<CTA>(e)...), m_functor(std::forward<Func>(f))
        {}

        ~subset_operator() = default;

        subset_operator(const subset_operator &) = default;
        subset_operator &operator=(const subset_operator &) = default;

        subset_operator(subset_operator &&) = default;
        subset_operator &operator=(subset_operator &&) = default;

        template<class Func>
        void operator()(Func &&func);

        template<std::size_t... I>
        auto on_impl(std::size_t ref_level, std::index_sequence<I...>) const;

        auto on(std::size_t ref_level) const;

        bool eval(int scan, std::size_t dim) const;
        void update(int scan, int sentinel);
        void reset();
        void decrement_dim(int i);
        void increment_dim();
        int min() const;
        int max() const;
        const tuple_type &get_tuple() const;

        template<std::size_t... I>
        void get_interval_index_impl(std::vector<std::size_t> &index, std::index_sequence<I...>)
        {
            (void)std::initializer_list<int>{
            (std::get<I>(m_e).get_interval_index(index), 0)...};
        }

        void get_interval_index(std::vector<std::size_t> &index){
            return get_interval_index_impl(index,
                           std::make_index_sequence<sizeof...(CT)>());
        }
      private:
        template<std::size_t... I, class... Args>
        bool eval_impl(int scan, std::size_t dim,
                       std::index_sequence<I...>) const;

        template<std::size_t... I, class... Args>
        void update_impl(int scan, int sentinel, std::index_sequence<I...>);

        template<std::size_t... I, class... Args>
        void reset_impl(std::index_sequence<I...>);

        template<std::size_t... I, class... Args>
        void decrement_dim_impl(int i, std::index_sequence<I...>);

        template<std::size_t... I, class... Args>
        void increment_dim_impl(std::index_sequence<I...>);

        template<std::size_t... I, class... Args>
        int min_impl(std::index_sequence<I...>) const;

        template<std::size_t... I, class... Args>
        int max_impl(std::index_sequence<I...>) const;

        template<class Func, std::size_t d>
        void sub_apply(Func &&func, std::integral_constant<std::size_t, d>);

        template<class Func>
        void sub_apply(Func &&func, std::integral_constant<std::size_t, 0>);

        template<class Func, std::size_t d>
        void apply(Func &&func, std::integral_constant<std::size_t, d>);

        tuple_type m_e;
        functor_type m_functor;
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> m_index_yz;
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> m_result;
    };

    /**********************************
     * subset_operator implementation *
     **********************************/

    template<class F, class... CT>
    template<class Func>
    inline void subset_operator<F, CT...>::operator()(Func &&func)
    {
        reset();
        apply(std::forward<Func>(func),
              std::integral_constant<std::size_t, dim - 1>{});
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline auto
    subset_operator<F, CT...>::on_impl(std::size_t ref_level,
                                       std::index_sequence<I...>) const
    {
        using new_op_type =
            subset_operator<F, make_projection_t<std::decay_t<CT>>...>;

        return new_op_type(
            F(),
            make_projection<std::decay_t<typename std::tuple_element<
                I, tuple_type>::type>>::apply(ref_level, std::get<I>(m_e))...);
    }

    template<class F, class... CT>
    inline auto subset_operator<F, CT...>::on(std::size_t ref_level) const
    {
        return on_impl(ref_level, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline bool subset_operator<F, CT...>::eval(int scan, std::size_t dim) const
    {
        return eval_impl(scan, dim, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::update(int scan, int sentinel)
    {
        return update_impl(scan, sentinel,
                           std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::reset()
    {
        return reset_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::decrement_dim(int i)
    {
        return decrement_dim_impl(i, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::increment_dim()
    {
        return increment_dim_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline int subset_operator<F, CT...>::min() const
    {
        return min_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline int subset_operator<F, CT...>::max() const
    {
        return max_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline auto subset_operator<F, CT...>::get_tuple() const
        -> const tuple_type &
    {
        return m_e;
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline bool
    subset_operator<F, CT...>::eval_impl(int scan, std::size_t dim,
                                         std::index_sequence<I...>) const
    {
        return m_functor(dim, std::get<I>(m_e).eval(scan, dim)...);
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline void
    subset_operator<F, CT...>::update_impl(int scan, int sentinel,
                                           std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).update(scan, sentinel), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline void subset_operator<F, CT...>::reset_impl(std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).reset(), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline void
    subset_operator<F, CT...>::decrement_dim_impl(int i,
                                                  std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).decrement_dim(i), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline void
    subset_operator<F, CT...>::increment_dim_impl(std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).increment_dim(), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline int
    subset_operator<F, CT...>::min_impl(std::index_sequence<I...>) const
    {
        return detail::do_min(std::get<I>(m_e).min()...);
    }

    template<class F, class... CT>
    template<std::size_t... I, class... Args>
    inline int
    subset_operator<F, CT...>::max_impl(std::index_sequence<I...>) const
    {
        return detail::do_max(std::get<I>(m_e).max()...);
    }

    template<class F, class... CT>
    template<class Func, std::size_t d>
    inline void
    subset_operator<F, CT...>::sub_apply(Func &&func,
                                         std::integral_constant<std::size_t, d>)
    {
        for (int i = m_result[d].start; i < m_result[d].end; ++i)
        {
            m_index_yz[d - 1] = i;

            decrement_dim(i);
            apply(std::forward<Func>(func),
                  std::integral_constant<std::size_t, d - 1>{});
            increment_dim();
        }
    }

    template<class F, class... CT>
    template<class Func>
    inline void
    subset_operator<F, CT...>::sub_apply(Func &&func,
                                         std::integral_constant<std::size_t, 0>)
    {
        std::vector<std::size_t> index;
        get_interval_index(index);
        func(m_index_yz, m_result, index);
    }

    template<class F, class... CT>
    template<class Func, std::size_t d>
    inline void
    subset_operator<F, CT...>::apply(Func &&func,
                                     std::integral_constant<std::size_t, d>)
    {
        std::size_t r_ipos = 0;
        auto scan = min();
        auto sentinel = max() + 1;
        interval_t result;

        while (scan < sentinel)
        {
            auto in_res = eval(scan, d);

            if (in_res ^ (r_ipos & 1))
            {
                if (r_ipos == 0)
                {
                    result.start = scan;
                    r_ipos = 1;
                }
                else
                {
                    result.end = scan;
                    r_ipos = 0;

                    if (result.is_valid())
                    {
                        m_result[d] = result;
                        sub_apply(std::forward<Func>(func),
                                  std::integral_constant<std::size_t, d>{});
                    }
                }
            }
            update(scan, sentinel);
            scan = min();
        }
    }

    namespace detail
    {
        template<class T, class enable = void>
        struct get_arg_impl
        {
            template<class R>
            decltype(auto) operator()(R &&r)
            {
                return std::forward<R>(r);
            }
        };

        template<class T>
        struct get_arg_impl<T, std::enable_if_t<is_node_op<T>::value>>
        {
            template<class R>
            decltype(auto) operator()(R &&r)
            {
                auto arg = get_arg_node(std::forward<T>(r));
                using arg_type = decltype(arg);
                return subset_node<arg_type>(std::forward<arg_type>(arg));
            }
        };

        template<>
        template<std::size_t Dim, class TInterval>
        struct get_arg_impl<LevelCellArray<Dim, TInterval>>
        {
            using mesh_t = LevelCellArray<Dim, TInterval>;

            template<class R>
            decltype(auto) operator()(R &&r)
            {
                return subset_node<mesh_node<mesh_t>>(
                    std::forward<mesh_node<mesh_t>>(r));
            }

        };
    }

    template<class T>
    decltype(auto) get_arg(T &&t)
    {
        detail::get_arg_impl<std::decay_t<T>> inv;
        return inv(std::forward<T>(t));
    }

    template<class F, class... E>
    inline auto make_subset_operator(E &&... e)
    {
        using function_type = subset_operator<F, E...>;
        using functor_type = typename function_type::functor_type;
        return function_type(functor_type(), std::forward<E>(e)...);
    }
}