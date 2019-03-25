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
        // t = std::forward<U>(u);
        t = static_cast<T>(u);
    }

    template<class coord_index_t, class levelcellarray>
    void new_endpoint(coord_index_t scan, coord_index_t sentinel,
                      const levelcellarray& array, std::size_t end, int shift,
                      std::size_t& it, std::size_t& index, coord_index_t& endpoints)
    {
        if (scan == endpoints)
        {
            if (index == 1)
            {
                it += 1;
                if (shift < 0)
                {
                    endpoints = (it == end)?sentinel:(array[it].start<<-shift);
                }
                else
                {
                    endpoints = (it == end)?sentinel:(array[it].start>>shift);
                }
                index = 0;
            }
            else
            {
                if (shift < 0)
                {
                    endpoints = (array[it].end<<-shift);
                }
                else
                {
                    endpoints = (array[it].end>>shift) + ((array[it].end&1 && shift)?1:0);
                }
                index = 1;
            }
        }
    }

    /*********************
     * SubSet definition *
     *********************/

    /**
     * @class SubSet
     * @brief Create a subset from a tuple of list of intervals.
     *
     * The SubSet class implements an operator to apply to a tuple of
     * list of intervals.
     *
     * @tparam MRConfig The MuRe config class.
     * @tparam Operator The operator type applied to the tuple of list of intervals.
     * @tparam T The type of each element of the tuple.
     */
    template<class MRConfig, class Operator, class... T>
    class SubSet
    {
        using expand = bool[];
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:

        using tuple_type = std::tuple<T...>;
        using operator_type = Operator;
        constexpr static std::size_t size = sizeof...(T);

        SubSet(Operator&& op, const tuple_type& level_cell_arrays);

        SubSet(Operator&& op, const std::size_t common_level,
               const std::array<std::size_t, size> data_level,
               tuple_type& level_cell_arrays);

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
                       xt::xtensor_fixed<interval_t, xt::xshape<dim>>& result,
                       const std::array<index_t, size>& index,
                       xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& index_yz,
                       xt::xtensor_fixed<index_t,
                                         xt::xshape<dim, size>>& interval_index,
                       Func&& func,
                       std::integral_constant<std::size_t, d>) const;

        template<size_t... I, class Func>
        void sub_apply(std::index_sequence<I...>,
                       xt::xtensor_fixed<interval_t, xt::xshape<dim>>& result,
                       const std::array<index_t, size>& index,
                       xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& index_yz,
                       xt::xtensor_fixed<index_t,
                                         xt::xshape<dim, size>>& interval_index,
                       Func&& func,
                       std::integral_constant<std::size_t, 0>) const;

        template <std::size_t... I, std::size_t d, class Func>
        void apply_impl(std::index_sequence<I...> iseq,
                        xt::xtensor_fixed<interval_t, xt::xshape<dim>>& result,
                        const std::array<std::size_t, size>& start,
                        const std::array<std::size_t, size>& end,
                        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& index_yz,
                        xt::xtensor_fixed<index_t,
                                          xt::xshape<dim, size>>& interval_index,
                        Func&& func,
                        std::integral_constant<std::size_t, d>) const;

        operator_type m_op;
        tuple_type m_data;
        const std::size_t m_common_level;
        std::array<std::size_t, size> m_data_level;
    };

    /*************************
     * SubSet implementation *
     *************************/

    template<class MRConfig, class Operator, class... T>
    SubSet<MRConfig, Operator, T...>::SubSet(Operator&& op, const tuple_type& level_cell_arrays)
        : m_op(std::forward<Operator>(op)), m_data(level_cell_arrays),
          m_common_level(0)
    {
        m_data_level.fill(0);
    }

    template<class MRConfig, class Operator, class... T>
    SubSet<MRConfig, Operator, T...>::SubSet(Operator&& op, const std::size_t common_level,
                                             const std::array<std::size_t, size> data_level,
                                             tuple_type& level_cell_arrays)
        : m_op(std::forward<Operator>(op)), m_data(level_cell_arrays),
          m_common_level(common_level), m_data_level(data_level)
    {}

    template<class MRConfig, class Operator, class... T>
    template <std::size_t... I>
    void SubSet<MRConfig, Operator, T...>::init_start_end(std::index_sequence<I...>,
                                                          std::array<std::size_t, size>& start,
                                                          std::array<std::size_t, size>& end) const
    {
        (void)expand{(generic_assign(start[I], 0), false)...};
        (void)expand{(generic_assign(end[I],
                            std::get<I>(m_data)[MRConfig::dim-1].size()), false)...};
    }

    template<class MRConfig, class Operator, class... T>
    template<size_t... I, std::size_t d, class Func>
    void SubSet<MRConfig, Operator, T...>::sub_apply(std::index_sequence<I...> iseq,
                                                     xt::xtensor_fixed<interval_t,
                                                                       xt::xshape<dim>>& result,
                                                    const std::array<index_t, size>& index,
                                                     xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& index_yz,
                                                     xt::xtensor_fixed<index_t,
                                                                       xt::xshape<dim, size>>& interval_index,
                                                     Func&& func,
                                                     std::integral_constant<std::size_t, d>) const
    {
        std::array<int, size> ii;
        for(int i=result[d].start; i<result[d].end; ++i)
        {
            index_yz[d] = i;
            (void)expand{(generic_assign(ii[I],
                                   (static_cast<int>(m_data_level[I]-m_common_level)<0)?
                                        i>>static_cast<int>(m_common_level-m_data_level[I])
                                       :i<<(m_data_level[I]-m_common_level)), false)...};

            std::array<std::size_t, size> new_start;
            std::array<std::size_t, size> new_end;
            std::array<std::size_t, size> off_ind;
            (void)expand{(generic_assign(off_ind[I],
                                   (index[I] != -1)?
                                       static_cast<std::size_t>(std::get<I>(m_data)[d][static_cast<std::size_t>(index[I])].index + ii[I]): std::numeric_limits<std::size_t>::max())
                                   , false)...};

            (void)expand{(generic_assign(new_start[I],
                                   (index[I] != -1 and off_ind[I] < std::get<I>(m_data).offsets(d).size())?
                                       std::get<I>(m_data).offsets(d)[off_ind[I]]
                                      :0), false)...};
            (void)expand{(generic_assign(new_end[I],
                                   (index[I] != -1 and (off_ind[I] + 1) < std::get<I>(m_data).offsets(d).size())?
                                        std::get<I>(m_data).offsets(d)[off_ind[I] + 1]
                                      :new_start[I]), false)...};

            apply_impl(iseq, result, new_start, new_end,
                       index_yz, interval_index, std::forward<Func>(func),
                       std::integral_constant<std::size_t, d>{});
        }
    }

    template<class MRConfig, class Operator, class... T>
    template<size_t... I, class Func>
    void SubSet<MRConfig, Operator, T...>::sub_apply(std::index_sequence<I...>,
                                                     xt::xtensor_fixed<interval_t,
                                                                       xt::xshape<dim>>& result,
                                                     const std::array<index_t, size>&, //index
                                                     xt::xtensor_fixed<coord_index_t,
                                                                       xt::xshape<dim>>& index_yz,
                                                      xt::xtensor_fixed<index_t,
                                                                        xt::xshape<dim, size>>& interval_index,
                                                     Func&& func,
                                                     std::integral_constant<std::size_t, 0>) const
    {
        // xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> index_yz_ = xt::view(index_yz, xt::range(1, dim));
        func(index_yz, result, interval_index);
        // func(index_yz, result);
    }

    template<class MRConfig, class Operator, class... T>
    template <std::size_t... I, std::size_t d, class Func>
    void SubSet<MRConfig, Operator, T...>::apply_impl(std::index_sequence<I...> iseq,
                                                      xt::xtensor_fixed<interval_t,
                                                                        xt::xshape<dim>>& result_array,
                                                      const std::array<std::size_t, size>& start,
                                                      const std::array<std::size_t, size>& end,
                                                      xt::xtensor_fixed<coord_index_t,
                                                                        xt::xshape<dim>>& index_yz,
                                                      xt::xtensor_fixed<index_t,
                                                                        xt::xshape<dim, size>>& interval_index,
                                                      Func&& func,
                                                      std::integral_constant<std::size_t, d>) const
    {
        std::array<coord_index_t, size> endpoints;
        std::array<coord_index_t, size> ends;
        std::array<bool, size> is_valid;

        (void)expand{(generic_assign(endpoints[I],
                               (end[I] != start[I])?
                                   ((static_cast<int>(m_data_level[I]-m_common_level)<0)?
                                         std::get<I>(m_data)[d-1][start[I]].start<<static_cast<int>(m_common_level-m_data_level[I])
                                        :std::get<I>(m_data)[d-1][start[I]].start>>(m_data_level[I]-m_common_level))
                                 : std::numeric_limits<coord_index_t>::max()), false)...};
        (void)expand{(generic_assign(ends[I],
                               (end[I] != start[I])?
                                   ((static_cast<int>(m_data_level[I]-m_common_level)<0)?
                                         (std::get<I>(m_data)[d-1][end[I] - 1].end<<static_cast<int>(m_common_level-m_data_level[I]))
                                        :((std::get<I>(m_data)[d-1][end[I] - 1].end>>(m_data_level[I]-m_common_level))
                                         + ((std::get<I>(m_data)[d-1][end[I] - 1].end&1 && (m_data_level[I]-m_common_level))?1:0)))
                                 : std::numeric_limits<coord_index_t>::min()), false)...};

        (void)expand{(generic_assign(is_valid[I], (end[I] == start[I])?false:true), false)...};

        auto scan = *std::min_element(endpoints.begin(), endpoints.end());
        auto sentinel = *std::max_element(ends.begin(), ends.end()) + 1;

        std::array<std::size_t, size> index;
        (void)expand{(generic_assign(index[I], start[I]), false)...};

        std::array<std::size_t, size> endpoints_index;
        endpoints_index.fill(0);

        std::size_t r_index = 0;
        typename MRConfig::interval_t result;

        auto in_ = repeat_as_tuple_t<size, bool>();
        while (scan < sentinel)
        {
            (void)expand{(generic_assign(std::get<I>(in_), !((scan < endpoints[I]) ^ endpoints_index[I])&is_valid[I]), false)...} ;
            auto in_res = m_op(d-1, std::get<I>(in_)...);

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
                    std::array<index_t, size> new_index;
                    (void)expand{(generic_assign(new_index[I], index[I] + (endpoints_index[I] - 1)), false)...};
                    (void)expand{(generic_assign(interval_index(d-1, I), new_index[I]), false)...};

                    if (result.is_valid())
                    {
                        result_array[d-1] = result;
                        sub_apply(iseq, result_array, new_index, index_yz, interval_index,
                                std::forward<Func>(func),
                                std::integral_constant<std::size_t, d-1>{});
                    }
                }
            }

            (void)expand{(new_endpoint(scan, sentinel, std::get<I>(m_data)[d-1], end[I],
                                 static_cast<int>(m_data_level[I] - m_common_level),
                                 index[I], endpoints_index[I], endpoints[I]), false)...};
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

        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index_yz;
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> result;

        xt::xtensor_fixed<index_t, xt::xshape<dim, size>> interval_index;

        apply_impl(std::make_index_sequence<sizeof...(T)>(),
                   result,
                    start, end, index_yz, interval_index, std::forward<Func>(func),
                    std::integral_constant<std::size_t, dim>{});
    }

    template <class MRConfig, class Operator, class... Args>
    auto make_subset(Operator&& op, Args&&... args)
    {
        using subset_type = SubSet<MRConfig, Operator, Args...>;
        auto tuple_value = std::tie(std::forward<Args>(args)...);
        return subset_type(std::forward<Operator>(op), tuple_value);
    }

    template <class MRConfig, class Operator, class... Args>
    auto make_subset(Operator&& op, std::size_t common_level,
                     std::array<std::size_t, sizeof...(Args)> data_level, Args&&... args)
    {
        using subset_type = SubSet<MRConfig, Operator, Args...>;
        auto tuple_value = std::tie(std::forward<Args>(args)...);
        return subset_type(std::forward<Operator>(op), common_level,
                           data_level, tuple_value);
    }
}