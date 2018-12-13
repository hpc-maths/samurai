#pragma once

#include <algorithm>
#include <functional>

#include <xtensor/xfixed.hpp>

#include "func_node.hpp"
#include "interval.hpp"
#include "level_cell_array.hpp"
#include "tuple.hpp"

namespace mure
{
    template <class T, class U>
    void generic_assign(T& t, U&& u)
    {
        t = std::forward<U>(u);
    }

    template<class coord_index_t, class levelcellarray>
    void new_endpoint(coord_index_t scan, coord_index_t sentinel,
                      const levelcellarray& array, std::size_t end,
                      std::size_t& it, std::size_t& index, coord_index_t& endpoints)
    {
        if (scan == endpoints)
        {
            if (index == 1)
            {
                it += 1;
                endpoints = (it == end)?sentinel:array[it].start;
                index = 0;
            }
            else
            {
                endpoints = array[it].end;
                index = 1;
            }
        }
    };

    /*********************
     * SubSet definition *
     *********************/

    template<class MRConfig, class Operator, class... T>
    class SubSet
    {
        using expand = bool[];
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:

        using tuple_type = std::tuple<T...>;
        using operator_type = Operator;
        constexpr static std::size_t size = sizeof...(T);

        SubSet(Operator&& op, const tuple_type& level_cell_arrays)
               : m_op(std::forward<Operator>(op)), m_data(level_cell_arrays)
        {}

        template<class Func>
        void apply(Func&& func) const;

    private:

        template <std::size_t... I>
        void init_start_end(std::index_sequence<I...>,
                            std::array<std::size_t, size>& start,
                            std::array<std::size_t, size>& end
                            ) const;

        template<size_t... I, std::size_t d, class Func>
        void sub_apply(std::index_sequence<I...> iseq,
                        const interval_t& result,
                        const std::array<std::size_t, size>& index,
                        xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>>& index_yz,
                        Func&& func,
                        std::integral_constant<std::size_t, d>) const;

        template<size_t... I, class Func>
        void sub_apply(std::index_sequence<I...>,
                       const interval_t& result,
                       const std::array<std::size_t, size>& index,
                       xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>>& index_yz,
                       Func&& func,
                       std::integral_constant<std::size_t, 0>) const;

        template <std::size_t... I, std::size_t d, class Func>
        void apply_impl(std::index_sequence<I...> iseq,
                        const std::array<std::size_t, size>& start,
                        const std::array<std::size_t, size>& end,
                        xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>>& index_yz,
                        Func&& func,
                        std::integral_constant<std::size_t, d>) const;

        tuple_type m_data;
        operator_type m_op;
    };

    /*************************
     * SubSet implementation *
     *************************/

    template<class MRConfig, class Operator, class... T>
    template <std::size_t... I>
    void SubSet<MRConfig, Operator, T...>::init_start_end(std::index_sequence<I...>,
                                                          std::array<std::size_t, size>& start,
                                                          std::array<std::size_t, size>& end) const
    {
        expand{(generic_assign(start[I],
                            std::get<I>(m_data).beg_ind_last_dim()), false)...};
        expand{(generic_assign(end[I],
                            std::get<I>(m_data).size()), false)...};
    }

    template<class MRConfig, class Operator, class... T>
    template<size_t... I, std::size_t d, class Func>
    void SubSet<MRConfig, Operator, T...>::sub_apply(std::index_sequence<I...> iseq,
                                                     const interval_t& result,
                                                     const std::array<std::size_t, size>& index,
                                                     xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>>& index_yz,
                                                     Func&& func,
                                                     std::integral_constant<std::size_t, d>) const
    {
        for(int i=result.start; i<result.end; ++i)
        {
            index_yz[d-1] = i;

            std::array<std::size_t, size> new_start;
            std::array<std::size_t, size> new_end;
            expand{(generic_assign(new_start[I],
                                    std::get<I>(m_data).offset(std::get<I>(m_data)[index[I]].index + i)), false)...};
            expand{(generic_assign(new_end[I],
                                    std::get<I>(m_data).offset(std::get<I>(m_data)[index[I]].index + i + 1)), false)...};

            apply_impl(iseq, new_start, new_end,
                        index_yz, std::forward<Func>(func),
                        std::integral_constant<std::size_t, d>{});
        }
    }

    template<class MRConfig, class Operator, class... T>
    template<size_t... I, class Func>
    void SubSet<MRConfig, Operator, T...>::sub_apply(std::index_sequence<I...>,
                                                     const interval_t& result,
                                                     const std::array<std::size_t, size>& index,
                                                     xt::xtensor_fixed<coord_index_t,
                                                                       xt::xshape<dim-1>>& index_yz,
                                                     Func&& func,
                                                     std::integral_constant<std::size_t, 0>) const
    {
        func(index_yz, result);
    }

    template<class MRConfig, class Operator, class... T>
    template <std::size_t... I, std::size_t d, class Func>
    void SubSet<MRConfig, Operator, T...>::apply_impl(std::index_sequence<I...> iseq,
                                                      const std::array<std::size_t, size>& start,
                                                      const std::array<std::size_t, size>& end,
                                                      xt::xtensor_fixed<coord_index_t,
                                                                        xt::xshape<dim-1>>& index_yz,
                                                      Func&& func,
                                                      std::integral_constant<std::size_t, d>) const
    {
        std::array<coord_index_t, size> endpoints;
        std::array<coord_index_t, size> ends;
        expand{(generic_assign(endpoints[I],
                            std::get<I>(m_data)[start[I]].start), false)...};
        expand{(generic_assign(ends[I],
                            std::get<I>(m_data)[end[I] - 1].end), false)...};

        auto scan = *std::min_element(endpoints.begin(), endpoints.end());
        auto sentinel = *std::max_element(ends.begin(), ends.end()) + 1;

        std::array<std::size_t, size> index;
        expand{(generic_assign(index[I], start[I]), false)...};

        std::array<std::size_t, size> endpoints_index;
        endpoints_index.fill(0);

        std::size_t r_index = 0;
        typename MRConfig::interval_t result;

        auto in_ = repeat_as_tuple_t<size, bool>();
        while (scan < sentinel)
        {
            expand{(generic_assign(std::get<I>(in_), !((scan < endpoints[I]) ^ endpoints_index[I])), false)...};
            auto in_res = m_op(std::get<I>(in_)...);

            if (in_res ^ (r_index & 1))
            {
                if (r_index == 0)
                {
                    result.start = scan;
                    r_index = 1;
                }
                else
                {
                    result.end = scan;
                    r_index = 0;
                    std::array<std::size_t, size> new_index;
                    expand{(generic_assign(new_index[I], index[I] + (endpoints_index[I] - 1)), false)...};

                    sub_apply(iseq, result, new_index, index_yz,
                              std::forward<Func>(func),
                              std::integral_constant<std::size_t, d-1>{});
                }
            }

            expand{(new_endpoint(scan, sentinel, std::get<I>(m_data), end[I], index[I], endpoints_index[I], endpoints[I]), false)...};
            scan = *std::min_element(endpoints.begin(), endpoints.end());
        }
    }

    template<class MRConfig, class Operator, class... T>
    template<class Func>
    void SubSet<MRConfig, Operator, T...>::apply(Func&& func) const
    {
        std::array<std::size_t, size> start;
        std::array<std::size_t, size> end;

        init_start_end(std::make_index_sequence<sizeof...(T)>(), start, end);

        xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz;

        apply_impl(std::make_index_sequence<sizeof...(T)>(),
                    start, end, index_yz, std::forward<Func>(func),
                    std::integral_constant<std::size_t, dim>{});
    }

    template <class MRConfig, class Operator, class... Args>
    auto make_subset(Operator&& op, Args&&... args)
    {
        using subset_type = SubSet<MRConfig, Operator, Args...>;
        auto tuple_value = std::tie(std::forward<Args>(args)...);
        return subset_type(std::forward<Operator>(op), tuple_value);
    }
}