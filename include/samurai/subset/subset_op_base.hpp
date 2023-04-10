// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>

#include "../level_cell_array.hpp"
#include "../static_algorithm.hpp"
#include "../utils.hpp"
#include "subset_node.hpp"

namespace samurai
{
    ////////////////////////////////
    // subset_operator definition //
    ////////////////////////////////

    /**
     * @class subset_operator
     * @brief Define a subset of a set of intervals.
     *
     * @tparam F the function type (intersection, union, difference,...)
     * @tparam CT the closure types for arguments of the function
     */
    template <class F, class... CT>
    class subset_operator
    {
      public:

        using functor_type = F;
        using tuple_type   = std::tuple<std::decay_t<CT>...>;

        static constexpr std::size_t dim = detail::compute_dim<CT...>();
        using interval_t                 = typename detail::interval_type<CT...>::type;
        using coord_index_t              = typename interval_t::coord_index_t;

        subset_operator(F&& f, CT&&... e);
        auto on(std::size_t ref_level) const;

        template <class Func>
        void operator()(Func&& func);

        template <class Func>
        void apply_interval_index(Func&& func);

        template <class... Op>
        void apply_op(Op&&... op);

        void reset();
        void init(std::size_t ref_level);

        bool eval(coord_index_t scan, std::size_t d) const;
        void update(coord_index_t scan, coord_index_t sentinel);

        void increment_dim();
        void decrement_dim(coord_index_t i);

        void set_shift(std::size_t ref_level, std::size_t common_level);

        coord_index_t min() const;
        coord_index_t max() const;
        std::size_t common_level() const;
        std::size_t level() const;

        bool is_empty() const;

        void get_interval_index(std::vector<std::size_t>& index) const;

      private:

        template <std::size_t... I>
        void init_impl(std::size_t ref_level, std::index_sequence<I...>);

        template <std::size_t... I>
        bool eval_impl(coord_index_t scan, std::size_t d, std::index_sequence<I...>) const;

        template <std::size_t... I>
        void update_impl(coord_index_t scan, coord_index_t sentinel, std::index_sequence<I...>);

        template <std::size_t... I>
        void reset_impl(std::index_sequence<I...>);

        template <std::size_t... I>
        void decrement_dim_impl(coord_index_t i, std::index_sequence<I...>);

        template <std::size_t... I>
        void increment_dim_impl(std::index_sequence<I...>);

        template <std::size_t... I>
        coord_index_t min_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        coord_index_t max_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        std::size_t common_level_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        bool is_empty_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        void set_shift_impl(std::index_sequence<I...>, std::size_t ref_level, std::size_t common_level);

        template <class Func, std::size_t d>
        void sub_apply(Func&& func, std::integral_constant<std::size_t, d>);

        template <class Func>
        void sub_apply(Func&& func, std::integral_constant<std::size_t, 0>);

        template <class Func, std::size_t d>
        void apply(Func&& func, std::integral_constant<std::size_t, d>);

        template <std::size_t... I>
        void get_interval_index_impl(std::vector<std::size_t>& index, std::index_sequence<I...>) const;

        //! The sets of the function defining the subset.
        tuple_type m_e;
        //! The function defining the subset (intersection, difference, union,
        //! ...)
        functor_type m_functor;
        //! The level where we want a result if it exists.
        std::size_t m_ref_level = 0;
        //! Storage of the current value in each dimension greater than 0
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> m_index_yz;
        //! Intervals found for each dimension
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> m_result;
    };

    ////////////////////////////////////
    // subset_operator implementation //
    ////////////////////////////////////

    /**
     * Construct a subset from an algebra of sets.
     * @param f function to apply on the sets (intersection, union, difference,
     * ...)
     * @param e list of sets apply by f
     */
    template <class F, class... CT>
    inline subset_operator<F, CT...>::subset_operator(F&& f, CT&&... e)
        : m_e(std::forward<CT>(e)...)
        , m_functor(std::forward<F>(f))
    {
        init(common_level());
    }

    /**
     * Apply a function on the subset
     * @param func function to apply on each element of the subset
     */
    template <class F, class... CT>
    template <class Func>
    inline void subset_operator<F, CT...>::operator()(Func&& func)
    {
        reset();
        auto func_hack = [&](auto& interval, auto& index, auto&)
        {
            std::forward<Func>(func)(interval, index);
        };

        apply(func_hack, std::integral_constant<std::size_t, dim - 1>{});
    }

    /**
     * Apply a function on the subset
     * @param func function to apply on each element of the subset
     */
    template <class F, class... CT>
    template <class Func>
    inline void subset_operator<F, CT...>::apply_interval_index(Func&& func)
    {
        reset();
        auto func_hack = [&](auto&, auto&, auto& interval_index)
        {
            std::forward<Func>(func)(interval_index);
        };

        apply(func_hack, std::integral_constant<std::size_t, dim - 1>{});
    }

    /**
     * Apply one or more operators on the subset
     * @param op operator to apply on each element of the subset
     * @sa operator
     */
    template <class F, class... CT>
    template <class... Op>
    inline void subset_operator<F, CT...>::apply_op(Op&&... op)
    {
        reset();
        auto func = [&](auto& interval, auto& index, auto&)
        {
            (void)std::initializer_list<int>{(op(m_ref_level, interval, index), 0)...};
        };
        apply(func, std::integral_constant<std::size_t, dim - 1>{});
    }

    /**
     * Specify the reference level where each set must be compared.
     * @param ref_level the reference level
     */
    template <class F, class... CT>
    inline auto subset_operator<F, CT...>::on(std::size_t ref_level) const
    {
        subset_operator<F, CT...> that{*this};
        that.init(ref_level);
        return that;
    }

    /**
     * Initialize the subset by defining the reference level and the shift
     * to be performed by each node of the subset to obtain the reference level.
     * @param ref_level the reference level
     */
    template <class F, class... CT>
    inline void subset_operator<F, CT...>::init(std::size_t ref_level)
    {
        m_ref_level = ref_level;
        init_impl(ref_level, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::init_impl(std::size_t ref_level, std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).set_shift(ref_level, common_level()), 0)...};
    }

    /**
     * Evaluate if the value scan is in the subset.
     * @param scan the value to evaluate
     * @param dim the current dimension
     */
    template <class F, class... CT>
    inline bool subset_operator<F, CT...>::eval(coord_index_t scan, std::size_t d) const
    {
        return eval_impl(scan, d, std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Update the value of each node of the subset.
     * @param scan the current state
     * @param sentinel the end value of the algorithm
     */
    template <class F, class... CT>
    inline void subset_operator<F, CT...>::update(coord_index_t scan, coord_index_t sentinel)
    {
        return update_impl(scan, sentinel, std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Reset the algorithm to find the subset on each node.
     */
    template <class F, class... CT>
    inline void subset_operator<F, CT...>::reset()
    {
        return reset_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Check if the subset is empty.
     *
     * For example:
     *
     *   - the difference between an empty set and other sets is empty
     *   - the intersection of an empty set  and other sets is empty
     *   - ...
     */
    template <class F, class... CT>
    inline bool subset_operator<F, CT...>::is_empty() const
    {
        return is_empty_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Initialize the next dimension with the interval found for the
     * current dimension.
     * @param i the current value found for the current dimension
     */
    template <class F, class... CT>
    inline void subset_operator<F, CT...>::decrement_dim(coord_index_t i)
    {
        return decrement_dim_impl(i, std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Increment by 1 the current dimension
     */
    template <class F, class... CT>
    inline void subset_operator<F, CT...>::increment_dim()
    {
        return increment_dim_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Return the minimum value of the current values of each node of the
     * subset.
     */
    template <class F, class... CT>
    inline auto subset_operator<F, CT...>::min() const -> coord_index_t
    {
        return min_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Return the maximum value of the current values of each node of the
     * subset.
     */
    template <class F, class... CT>
    inline auto subset_operator<F, CT...>::max() const -> coord_index_t
    {
        return max_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Return the common level of the subset which is the maximum level of all
     * nodes.
     */
    template <class F, class... CT>
    inline std::size_t subset_operator<F, CT...>::common_level() const
    {
        return common_level_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Return the level of the resulting subset.
     */
    template <class F, class... CT>
    inline std::size_t subset_operator<F, CT...>::level() const
    {
        return m_ref_level;
    }

    /**
     * Initialize the shift for each node which is computed by the difference
     * between the reference level and the level of each node.
     */
    template <class F, class... CT>
    inline void subset_operator<F, CT...>::set_shift(std::size_t ref_level, std::size_t common_level)
    {
        set_shift_impl(std::make_index_sequence<sizeof...(CT)>(), ref_level, common_level);
    }

    /**
     * Apply the algorithm to find the subset on the dimension d - 1
     * given a matching interval for the dimension d
     * @param func function to apply on the subset at the end
     */
    template <class F, class... CT>
    template <class Func, std::size_t d>
    inline void subset_operator<F, CT...>::sub_apply(Func&& func, std::integral_constant<std::size_t, d>)
    {
        for (int i = m_result[d].start; i < m_result[d].end; ++i)
        {
            m_index_yz[d - 1] = i;

            decrement_dim(i);
            apply(std::forward<Func>(func), std::integral_constant<std::size_t, d - 1>{});
            increment_dim();
        }
    }

    /**
     * Apply the function on the dimension 0 for the matching interval
     * @param func function to apply on the subset at the end
     */
    template <class F, class... CT>
    template <class Func>
    inline void subset_operator<F, CT...>::sub_apply(Func&& func, std::integral_constant<std::size_t, 0>)
    {
        std::vector<std::size_t> index;
        // Store into index the intervals of each node that are
        // in the subset.
        get_interval_index(index);

        // If the ref_level <= to common_level then the result
        // is a projection to a lower level which means that the result
        // is already at the right level and we can call func on it.
        // Otherwise, we have found the right interval subset on the common
        // level which is lower than the ref_level and we need to project the
        // result on the ref_level before calling func on it.
        if (m_ref_level <= common_level())
        {
            func(m_result[0], m_index_yz, index);
        }
        else
        {
            std::size_t shift                                                    = m_ref_level - common_level();
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> shift_index_yz = m_index_yz << shift;
            xt::xtensor_fixed<interval_t, xt::xshape<dim>> shift_result;
            for (std::size_t d = 0; d < dim; ++d)
            {
                shift_result[d] = m_result[d] << shift;
            }
            static_nested_loop<dim - 1>(0,
                                        1 << shift,
                                        1,
                                        [&](auto stencil)
                                        {
                                            auto index_yz = xt::eval(shift_index_yz + stencil);
                                            func(shift_result[0], index_yz, index);
                                        });
        }
    }

    /**
     * Core algorithm to find the subset
     * @param func function to apply on the subset at the end
     *             if it exists
     */
    template <class F, class... CT>
    template <class Func, std::size_t d>
    inline void subset_operator<F, CT...>::apply(Func&& func, std::integral_constant<std::size_t, d>)
    {
        // If we already know that this subset is empty, do nothing.
        //
        // Examples:
        // - the intersection of an empty set with other sets is empty
        // - the difference of an empty set with other sets is empty
        if (is_empty())
        {
            return;
        }

        interval_t result;
        std::size_t r_ipos = 0;

        coord_index_t scan     = min();
        coord_index_t sentinel = max() + 1;

        // spdlog::debug("scan -> {} sentinel -> {}", scan, sentinel);
        while (scan < sentinel)
        {
            // Check if scan is in the subset
            bool in_res = eval(scan, d);
            // spdlog::debug("in_res for {} -> {}", d, in_res);

            // Two cases:
            //
            // - in_res is true and thus scan is in the subset. If
            //   the interval result is not set yet (r_ipos = 0) then
            //   start the result interval. Otherwise, do nothing.
            //
            // - in_res is false and thus scan is not in the subset. If
            //   the start of the interval result is already set (r_ipos = 1),
            //   it means that scan is the first value outside of the result.
            //   Thus, close the interval and use it for the next dimensions.

            if (in_res ^ (r_ipos & 1)) // NOLINT(hicpp-signed-bitwise)
            {
                if (r_ipos == 0)
                {
                    result.start = scan;
                    r_ipos       = 1;
                }
                else
                {
                    result.end = scan;
                    r_ipos     = 0;

                    // spdlog::debug("result found {}", result);
                    if (result.is_valid())
                    {
                        m_result[d] = result;
                        sub_apply(std::forward<Func>(func), std::integral_constant<std::size_t, d>{});
                    }
                }
            }

            // update the position on the intervals on each node using scan and
            // sentinel. if the current position of the node is scan then move
            // it to the next position. if we already scan all the intervals of
            // a node then set the current value equal to sentinel.
            update(scan, sentinel);
            scan = min();
        }
    }

    template <class F, class... CT>
    inline void subset_operator<F, CT...>::get_interval_index(std::vector<std::size_t>& index) const
    {
        return get_interval_index_impl(index, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline bool subset_operator<F, CT...>::eval_impl(coord_index_t scan, std::size_t d, std::index_sequence<I...>) const
    {
        return m_functor(d, std::get<I>(m_e).eval(scan, d)...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline bool subset_operator<F, CT...>::is_empty_impl(std::index_sequence<I...>) const
    {
        return m_functor.is_empty(std::get<I>(m_e).is_empty()...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::update_impl(coord_index_t scan, coord_index_t sentinel, std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).update(scan, sentinel), 0)...};
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::reset_impl(std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).reset(), 0)...};
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::decrement_dim_impl(coord_index_t i, std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).decrement_dim(i), 0)...};
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::increment_dim_impl(std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).increment_dim(), 0)...};
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline auto subset_operator<F, CT...>::min_impl(std::index_sequence<I...>) const -> coord_index_t
    {
        return detail::do_min(std::get<I>(m_e).min()...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline auto subset_operator<F, CT...>::max_impl(std::index_sequence<I...>) const -> coord_index_t
    {
        return detail::do_max(std::get<I>(m_e).max()...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline std::size_t subset_operator<F, CT...>::common_level_impl(std::index_sequence<I...>) const
    {
        return detail::do_max(std::get<I>(m_e).common_level()...);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::set_shift_impl(std::index_sequence<I...>, std::size_t ref_level, std::size_t common_level)
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).set_shift(ref_level, common_level), 0)...};
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline void subset_operator<F, CT...>::get_interval_index_impl(std::vector<std::size_t>& index, std::index_sequence<I...>) const
    {
        (void)std::initializer_list<int>{(std::get<I>(m_e).get_interval_index(index), 0)...};
    }

    template <class D>
    class node_op;

    template <class Mesh>
    struct mesh_node;

    template <class E>
    using is_node_op = xt::is_crtp_base_of<node_op, E>;

    namespace detail
    {
        template <class T, class enable = void>
        struct get_arg_impl
        {
            template <class R>
            decltype(auto) operator()(R&& r)
            {
                return std::forward<R>(r);
            }
        };

        template <class T>
        struct get_arg_impl<T, std::enable_if_t<is_node_op<T>::value>>
        {
            template <class R>
            auto operator()(R&& r)
            {
                auto arg       = get_arg_node(std::forward<T>(r));
                using arg_type = decltype(arg);
                return subset_node<arg_type>(std::forward<arg_type>(arg));
            }
        };

        template <std::size_t Dim, class TInterval>
        struct get_arg_impl<LevelCellArray<Dim, TInterval>>
        {
            using mesh_t = LevelCellArray<Dim, TInterval>;

            template <class R>
            auto operator()(const R& r)
            {
                return subset_node<mesh_node<mesh_t>>(r);
            }
        };
    } // namespace detail

    template <class T>
    decltype(auto) get_arg(T&& t)
    {
        detail::get_arg_impl<std::decay_t<T>> inv;
        return inv(std::forward<T>(t));
    }

    template <class F, class... E>
    inline auto make_subset_operator(E&&... e)
    {
        using function_type = subset_operator<F, E...>;
        using functor_type  = typename function_type::functor_type;
        return function_type(functor_type(), std::forward<E>(e)...);
    }
} // namespace samurai
