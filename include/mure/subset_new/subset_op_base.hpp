#pragma once

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>

// #include <spdlog/spdlog.h>
// #include <spdlog/fwd.h>

#include "../level_cell_array.hpp"
#include "../static_algorithm.hpp"
#include "../utils.hpp"
#include "subset_node.hpp"

namespace mure
{
    template<class F, class... CT>
    class subset_operator;

    /******************************
     * subset_operator definition *
     ******************************/

    /**
     * @class subset_operator
     * @brief Define a subset of different intervals.
     *
     * @tparam F the function type (intersection, union, difference,...)
     * @tparam CT the closure types for arguments of the function
     */
    template<class F, class... CT>
    class subset_operator {
      public:
        using functor_type = F;
        using tuple_type = std::tuple<std::decay_t<CT>...>;

        static constexpr std::size_t dim = detail::compute_dim<CT...>();
        using interval_t = typename detail::interval_type<CT...>::type;
        using coord_index_t = typename interval_t::coord_index_t;

        // template<class Func, class... CTA>
        subset_operator(F &&f, CT &&... e): m_e(std::forward<CT>(e)...), m_functor(std::forward<F>(f))
        {
            init(common_level());
        }

        ~subset_operator() = default;

        subset_operator(const subset_operator &) = default;
        subset_operator &operator=(const subset_operator &) = default;

        subset_operator(subset_operator &&) = default;
        subset_operator &operator=(subset_operator &&) = default;

        template<class Func>
        void operator()(Func&& func);

        template<class... Op>
        void apply_op(std::size_t level, Op&&... op);

        auto on(std::size_t ref_level) const;

        bool eval(coord_index_t scan, std::size_t dim) const;
        void update(coord_index_t scan, coord_index_t sentinel);
        void reset();
        void init(std::size_t ref_level);
        void decrement_dim(coord_index_t i);
        void increment_dim();
        void set_shift(std::size_t ref_level, std::size_t common_level);
        coord_index_t min() const;
        coord_index_t max() const;
        std::size_t common_level() const;
        bool is_empty() const;
        const tuple_type &get_tuple() const;

        template<std::size_t... I>
        void get_interval_index_impl(std::vector<std::size_t> &index, std::index_sequence<I...>)
        {
            (void)std::initializer_list<int>{
                (std::get<I>(m_e).get_interval_index(index), 0)...};
        }

        void get_interval_index(std::vector<std::size_t> &index)
        {
            return get_interval_index_impl(
                index, std::make_index_sequence<sizeof...(CT)>());
        }

      private:
        template<std::size_t... I>
        void init_impl(std::size_t ref_level, std::index_sequence<I...>);

        template<std::size_t... I>
        bool eval_impl(coord_index_t scan, std::size_t dim, std::index_sequence<I...>) const;

        template<std::size_t... I>
        void update_impl(coord_index_t scan, coord_index_t sentinel, std::index_sequence<I...>);

        template<std::size_t... I>
        void reset_impl(std::index_sequence<I...>);

        template<std::size_t... I>
        void decrement_dim_impl(coord_index_t i, std::index_sequence<I...>);

        template<std::size_t... I>
        void increment_dim_impl(std::index_sequence<I...>);

        template<std::size_t... I>
        coord_index_t min_impl(std::index_sequence<I...>) const;

        template<std::size_t... I>
        coord_index_t max_impl(std::index_sequence<I...>) const;

        template<std::size_t... I>
        std::size_t common_level_impl(std::index_sequence<I...>) const;

        template<std::size_t... I>
        bool is_empty_impl(std::index_sequence<I...>) const;

        template<std::size_t... I>
        void set_shift_impl(std::index_sequence<I...>, std::size_t ref_level, std::size_t common_level);

        template<class Func, std::size_t d>
        void sub_apply(Func &&func, std::integral_constant<std::size_t, d>);

        template<class Func>
        void sub_apply(Func &&func, std::integral_constant<std::size_t, 0>);

        template<class Func, std::size_t d>
        void apply(Func &&func, std::integral_constant<std::size_t, d>);

        tuple_type m_e;
        functor_type m_functor;
        std::size_t m_ref_level;
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
        apply(std::forward<Func>(func), std::integral_constant<std::size_t, dim - 1>{});
    }

    template<class F, class... CT>
    template<class... Op>
    inline void subset_operator<F, CT...>::apply_op(std::size_t level, Op&&... op)
    {
        reset();
        auto func = [&](auto &index, auto &interval, auto &) {
            (void)std::initializer_list<int>{
                (op(level, interval[0], index), 0)...};
        };
        apply(func, std::integral_constant<std::size_t, dim - 1>{});
    }

    template<class F, class... CT>
    inline auto subset_operator<F, CT...>::on(std::size_t ref_level) const
    {
        subset_operator<F, CT...> that{*this};
        that.init(ref_level);
        return that;
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::init(std::size_t ref_level)
    {
        m_ref_level = ref_level;
        init_impl(ref_level, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline void
    subset_operator<F, CT...>::init_impl(std::size_t ref_level,
                                         std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).set_shift(ref_level, common_level()), 0)...};
    }

    template<class F, class... CT>
    inline bool subset_operator<F, CT...>::eval(coord_index_t scan, std::size_t dim) const
    {
        return eval_impl(scan, dim, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::update(coord_index_t scan, coord_index_t sentinel)
    {
        return update_impl(scan, sentinel, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::reset()
    {
        return reset_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline bool subset_operator<F, CT...>::is_empty() const
    {
        return is_empty_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::decrement_dim(coord_index_t i)
    {
        return decrement_dim_impl(i, std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::increment_dim()
    {
        return increment_dim_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline auto subset_operator<F, CT...>::min() const -> coord_index_t
    {
        return min_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline auto subset_operator<F, CT...>::max() const -> coord_index_t
    {
        return max_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline std::size_t subset_operator<F, CT...>::common_level() const
    {
        return common_level_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template<class F, class... CT>
    inline void subset_operator<F, CT...>::set_shift(std::size_t ref_level, std::size_t common_level)
    {
        set_shift_impl(std::make_index_sequence<sizeof...(CT)>(), ref_level, common_level);
    }

    template<class F, class... CT>
    inline auto subset_operator<F, CT...>::get_tuple() const
        -> const tuple_type &
    {
        return m_e;
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline bool
    subset_operator<F, CT...>::eval_impl(coord_index_t scan, std::size_t dim,
                                         std::index_sequence<I...>) const
    {
        return m_functor(dim, std::get<I>(m_e).eval(scan, dim)...);
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline bool
    subset_operator<F, CT...>::is_empty_impl(std::index_sequence<I...>) const
    {
        return m_functor.is_empty(std::get<I>(m_e).is_empty()...);
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline void
    subset_operator<F, CT...>::update_impl(coord_index_t scan, coord_index_t sentinel,
                                           std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).update(scan, sentinel), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline void subset_operator<F, CT...>::reset_impl(std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).reset(), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline void subset_operator<F, CT...>::decrement_dim_impl(coord_index_t i,
                                                  std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).decrement_dim(i), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline void subset_operator<F, CT...>::increment_dim_impl(std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).increment_dim(), 0)...};
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline auto subset_operator<F, CT...>::min_impl(std::index_sequence<I...>) const -> coord_index_t
    {
        return detail::do_min(std::get<I>(m_e).min()...);
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline auto subset_operator<F, CT...>::max_impl(std::index_sequence<I...>) const -> coord_index_t
    {
        return detail::do_max(std::get<I>(m_e).max()...);
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline std::size_t subset_operator<F, CT...>::common_level_impl(std::index_sequence<I...>) const
    {
        return detail::do_max(std::get<I>(m_e).common_level()...);
    }

    template<class F, class... CT>
    template<std::size_t... I>
    inline void subset_operator<F, CT...>::set_shift_impl(std::index_sequence<I...>, std::size_t ref_level, std::size_t common_level)
    {
        (void)std::initializer_list<int>{
            (std::get<I>(m_e).set_shift(ref_level, common_level), 0)...};
    }

    template<class F, class... CT>
    template<class Func, std::size_t d>
    inline void subset_operator<F, CT...>::sub_apply(Func &&func, std::integral_constant<std::size_t, d>)
    {
        for (int i = m_result[d].start; i < m_result[d].end; ++i)
        {
            m_index_yz[d - 1] = i;

            decrement_dim(i);
            apply(std::forward<Func>(func), std::integral_constant<std::size_t, d - 1>{});
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
        if (m_ref_level <= common_level())
        {
            func(m_index_yz, m_result, index);
        }
        else
        {
            std::size_t shift = m_ref_level - common_level();
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> shift_index_yz = m_index_yz << shift;
            xt::xtensor_fixed<interval_t, xt::xshape<dim>> shift_result;
            for(std::size_t d=0; d<dim; ++d)
            {
                shift_result[d] = m_result[d] << shift;
            }
            static_nested_loop<dim - 1>(0, 1 << shift, 1, [&](auto stencil) {
                auto index = xt::eval(shift_index_yz + stencil);
                func(index, shift_result, index);
            });
        }
    }

    template<class F, class... CT>
    template<class Func, std::size_t d>
    inline void
    subset_operator<F, CT...>::apply(Func &&func, std::integral_constant<std::size_t, d>)
    {
        if (is_empty())
        {
            return;
        }

        std::size_t r_ipos = 0;
        coord_index_t scan = min();
        coord_index_t sentinel = max() + 1;
        interval_t result;

        while (scan < sentinel)
        {
            bool in_res = eval(scan, d);
            //spdlog::debug("For dimension {}, scan = {} and sentinel = {}, in_res = {}", d, scan, sentinel, in_res);

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
                        //spdlog::debug("For dimension {}, matched interval = {}", d, result);
                        m_result[d] = result;
                        sub_apply(std::forward<Func>(func), std::integral_constant<std::size_t, d>{});
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
            auto operator()(R &&r)
            {
                auto arg = get_arg_node(std::forward<T>(r));
                using arg_type = decltype(arg);
                return subset_node<arg_type>(std::forward<arg_type>(arg));
            }
        };

        template<std::size_t Dim, class TInterval>
        struct get_arg_impl<LevelCellArray<Dim, TInterval>>
        {
            using mesh_t = LevelCellArray<Dim, TInterval>;

            template<class R>
            auto operator()(const R& r)
            {
                return subset_node<mesh_node<mesh_t>>(r);
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