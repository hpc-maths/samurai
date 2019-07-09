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
    template<class T, class U>
    void generic_assign(T &t, U &&u)
    {
        t = std::forward<U>(u);
    }

    template<class coord_index_t, class levelcellarray>
    void new_endpoint(coord_index_t scan, coord_index_t sentinel,
                      const levelcellarray &array, std::size_t end,
                      std::size_t &it, std::size_t &index,
                      coord_index_t &endpoints)
    {
        if (scan == endpoints)
        {
            if (index == 1)
            {
                it += 1;
                endpoints = (it == end) ? sentinel : array[it].start;
                index = 0;
            }
            else
            {
                endpoints = array[it].end;
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
     * @tparam Operator The operator type applied to the tuple of list of
     * intervals.
     */
    template<class MRConfig, class Operator, std::size_t Size>
    class SubSet {
        using expand = bool[];
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

      public:
        using operator_type = Operator;
        using set_type = typename std::array<LevelCellArray<MRConfig>, Size>;

        SubSet(Operator &&op, const set_type &level_cell_arrays);

        SubSet(Operator &&op, const std::size_t common_level,
               const std::array<std::size_t, Size> data_level,
               const set_type &level_cell_arrays);

        template<class Func>
        void apply(Func &&func) const;

      private:
        auto projection(const set_type &array);

        void init_start_end(std::array<std::size_t, Size> &start,
                            std::array<std::size_t, Size> &end) const;

        template<size_t... I, std::size_t d, class Func>
        void sub_apply(
            std::index_sequence<I...> iseq,
            xt::xtensor_fixed<interval_t, xt::xshape<dim>> &result,
            const std::array<index_t, Size> &index,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
            xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> &interval_index,
            Func &&func, std::integral_constant<std::size_t, d>) const;

        template<size_t... I, class Func>
        void sub_apply(
            std::index_sequence<I...>,
            xt::xtensor_fixed<interval_t, xt::xshape<dim>> &result,
            const std::array<index_t, Size> &index,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
            xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> &interval_index,
            Func &&func, std::integral_constant<std::size_t, 0>) const;

        template<std::size_t... I, std::size_t d, class Func>
        void apply_impl(
            std::index_sequence<I...> iseq,
            xt::xtensor_fixed<interval_t, xt::xshape<dim>> &result,
            const std::array<std::size_t, Size> &start,
            const std::array<std::size_t, Size> &end,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
            xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> &interval_index,
            Func &&func, std::integral_constant<std::size_t, d>) const;

        operator_type m_op;
        set_type m_data;
        const std::size_t m_common_level;
        std::size_t m_max_level;
        std::array<std::size_t, Size> m_data_level;
    };

    /*************************
     * SubSet implementation *
     *************************/

    template<class MRConfig, class Operator, std::size_t Size>
    SubSet<MRConfig, Operator, Size>::SubSet(Operator &&op,
                                             const set_type &level_cell_arrays)
        : m_op(std::forward<Operator>(op)), m_data(level_cell_arrays),
          m_common_level(0), m_max_level(0)
    {
        m_data_level.fill(0);
    }

    template<class MRConfig, class Operator, std::size_t Size>
    auto SubSet<MRConfig, Operator, Size>::projection(
        const set_type &level_cell_arrays)
    {
        set_type data;
        for (std::size_t i = 0; i < Size; ++i)
        {
            auto lca = level_cell_arrays[i];
            if (m_data_level[i] > m_common_level)
            {
                LevelCellList<MRConfig> lcl;
                std::size_t shift = m_data_level[i] - m_common_level;
                lca.for_each_interval_in_x([&](auto const &index_yz,
                                               auto const &interval) {
                    auto new_start = interval.start >> shift;
                    auto new_end = interval.end >> shift;
                    if (new_start == new_end)
                    {
                        new_end++;
                    }
                    lcl[index_yz >> shift].add_interval({new_start, new_end});
                });
                data[i] = {lcl};
            }
            else if (m_data_level[i] < m_common_level)
            {
                LevelCellList<MRConfig> lcl;
                std::size_t shift = m_common_level - m_data_level[i];
                lca.for_each_interval_in_x([&](auto const &index_yz,
                                               auto const &interval) {
                    // TODO: fix for 3D
                    for (int j = 0; j < 2 * shift; ++j)
                    {
                        lcl[xt::eval((index_yz << shift) + j)].add_interval(
                            {interval.start << shift, interval.end << shift});
                    }
                });
                data[i] = {lcl};
            }
            else
            {
                data[i] = {lca};
            }
        }
        return data;
    }

    template<class MRConfig, class Operator, std::size_t Size>
    SubSet<MRConfig, Operator, Size>::SubSet(
        Operator &&op, const std::size_t common_level,
        const std::array<std::size_t, Size> data_level,
        const set_type &level_cell_arrays)
        : m_op(std::forward<Operator>(op)), m_common_level(common_level),
          m_data_level(data_level)
    {
        m_data = projection(level_cell_arrays);
    }

    template<class MRConfig, class Operator, std::size_t Size>
    void SubSet<MRConfig, Operator, Size>::init_start_end(
        std::array<std::size_t, Size> &start,
        std::array<std::size_t, Size> &end) const
    {
        start.fill(0);
        for (std::size_t i = 0; i < Size; ++i)
        {
            end[i] = m_data[i][MRConfig::dim - 1].size();
        }
    }

    template<class MRConfig, class Operator, std::size_t Size>
    template<std::size_t... I, std::size_t d, class Func>
    void SubSet<MRConfig, Operator, Size>::sub_apply(
        std::index_sequence<I...> iseq,
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> &result,
        const std::array<index_t, Size> &index,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
        xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> &interval_index,
        Func &&func, std::integral_constant<std::size_t, d>) const
    {
        for (int i = result[d].start; i < result[d].end; ++i)
        {
            index_yz[d] = i;

            std::array<std::size_t, Size> new_start;
            std::array<std::size_t, Size> new_end;
            std::array<std::size_t, Size> off_ind;

            for (std::size_t j = 0; j < Size; ++j)
            {
                off_ind[j] =
                    (index[j] != -1)
                        ? static_cast<std::size_t>(
                              m_data[j][d][static_cast<std::size_t>(index[j])]
                                  .index +
                              i)
                        : std::numeric_limits<std::size_t>::max();
                new_start[j] = (index[j] != -1 and
                                off_ind[j] < m_data[j].offsets(d).size())
                                   ? m_data[j].offsets(d)[off_ind[j]]
                                   : 0;
                new_end[j] = (index[j] != -1 and
                              (off_ind[j] + 1) < m_data[j].offsets(d).size())
                                 ? m_data[j].offsets(d)[off_ind[j] + 1]
                                 : new_start[j];
            }

            apply_impl(iseq, result, new_start, new_end, index_yz,
                       interval_index, std::forward<Func>(func),
                       std::integral_constant<std::size_t, d>{});
        }
    }

    template<class MRConfig, class Operator, std::size_t Size>
    template<std::size_t... I, class Func>
    void SubSet<MRConfig, Operator, Size>::sub_apply(
        std::index_sequence<I...>,
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> &result,
        const std::array<index_t, Size> &, // index
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
        xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> &interval_index,
        Func &&func, std::integral_constant<std::size_t, 0>) const
    {
        func(index_yz, result, interval_index);
    }

    template<class MRConfig, class Operator, std::size_t Size>
    template<std::size_t... I, std::size_t d, class Func>
    void SubSet<MRConfig, Operator, Size>::apply_impl(
        std::index_sequence<I...> iseq,
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> &result_array,
        const std::array<std::size_t, Size> &start,
        const std::array<std::size_t, Size> &end,
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
        xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> &interval_index,
        Func &&func, std::integral_constant<std::size_t, d>) const
    {
        std::array<coord_index_t, Size> endpoints;
        std::array<coord_index_t, Size> ends;
        std::array<bool, Size> is_valid;
        std::array<std::size_t, Size> index;
        std::array<std::size_t, Size> endpoints_index;

        for (std::size_t i = 0; i < Size; ++i)
        {
            endpoints[i] = (end[i] != start[i])
                               ? m_data[i][d - 1][start[i]].start
                               : std::numeric_limits<coord_index_t>::max();
            ends[i] = (end[i] != start[i])
                          ? m_data[i][d - 1][end[i] - 1].end
                          : std::numeric_limits<coord_index_t>::min();
            is_valid[i] = (end[i] == start[i]) ? false : true;
            index[i] = start[i];
        }
        endpoints_index.fill(0);

        auto scan = *std::min_element(endpoints.begin(), endpoints.end());
        auto sentinel = *std::max_element(ends.begin(), ends.end()) + 1;

        std::size_t r_index = 0;
        typename MRConfig::interval_t result;

        auto in_ = repeat_as_tuple_t<Size, bool>();
        while (scan < sentinel)
        {
            (void)expand{
                (generic_assign(std::get<I>(in_),
                                !((scan < endpoints[I]) ^ endpoints_index[I]) &
                                    is_valid[I]),
                 false)...};
            auto in_res = m_op(d - 1, std::get<I>(in_)...);

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
                    std::array<index_t, Size> new_index;
                    for (std::size_t i = 0; i < Size; ++i)
                    {
                        new_index[i] = index[i] + (endpoints_index[i] - 1);
                        interval_index(d - 1, i) = new_index[i];
                    }

                    if (result.is_valid())
                    {
                        result_array[d - 1] = result;
                        sub_apply(iseq, result_array, new_index, index_yz,
                                  interval_index, std::forward<Func>(func),
                                  std::integral_constant<std::size_t, d - 1>{});
                    }
                }
            }

            for (std::size_t i = 0; i < Size; ++i)
            {
                new_endpoint(scan, sentinel, m_data[i][d - 1], end[i], index[i],
                             endpoints_index[i], endpoints[i]);
            }
            scan = *std::min_element(endpoints.begin(), endpoints.end());
        }
    }

    template<class MRConfig, class Operator, std::size_t Size>
    template<class Func>
    void SubSet<MRConfig, Operator, Size>::apply(Func &&func) const
    {
        std::array<std::size_t, Size> start;
        std::array<std::size_t, Size> end;

        init_start_end(start, end);

        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index_yz;
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> result;

        xt::xtensor_fixed<index_t, xt::xshape<dim, Size>> interval_index;

        apply_impl(std::make_index_sequence<Size>(), result, start, end,
                   index_yz, interval_index, std::forward<Func>(func),
                   std::integral_constant<std::size_t, dim>{});
    }

    template<class MRConfig, class Operator, std::size_t Size>
    auto make_subset(Operator &&op,
                     std::array<LevelCellArray<MRConfig>, Size> &args)
    {
        using subset_type = SubSet<MRConfig, Operator, Size>;
        return subset_type(std::forward<Operator>(op), args);
    }

    template<class MRConfig, class Operator, std::size_t Size>
    auto make_subset(Operator &&op, std::size_t common_level,
                     std::array<std::size_t, Size> data_level,
                     std::array<LevelCellArray<MRConfig>, Size> &args)
    {
        using subset_type = SubSet<MRConfig, Operator, Size>;
        return subset_type(std::forward<Operator>(op), common_level, data_level,
                           args);
    }
}