// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <type_traits>
#include <xtensor/xfixed.hpp>

#include "cell.hpp"

namespace samurai
{
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    class CellArray;

    template<std::size_t dim_, class TInterval>
    class LevelCellArray;

    template<class D, class Config>
    class Mesh_base;

    //////////////////////////////////////
    // for_each_interval implementation //
    //////////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_interval(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        for(auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            f(lca.level(), *it, it.index());
        }
    }

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_interval(LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        for(auto it = lca.begin(); it != lca.end(); ++it)
        {
            f(lca.level(), *it, it.index());
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_interval(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for(std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_interval(ca[level], std::forward<Func>(f));
            }
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_interval(CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for(std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_interval(ca[level], std::forward<Func>(f));
            }
        }
    }

    //////////////////////////////////
    // for_each_cell implementation //
    //////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func &&f)
    {
        using coord_index_t = typename TInterval::coord_index_t;
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index;

        for(auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            for(std::size_t d = 0; d < dim - 1; ++d)
            {
                index[d + 1] = it.index()[d];
            }

            for(coord_index_t i = it->start; i < it->end; ++i)
            {
                index[0] = i;
                Cell<coord_index_t, dim> cell{lca.level(), index, static_cast<std::size_t>(it->index + i)};
                f(cell);
            }
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_cell(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for(std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_cell(ca[level], std::forward<Func>(f));
            }
        }
    }

    template <class D, class Config, class Func>
    inline void for_each_cell(const Mesh_base<D, Config>& mesh, Func&& f)
    {
        using mesh_id_t = typename Config::mesh_id_t;
        for_each_cell(mesh[mesh_id_t::cells], std::forward<Func>(f));
    }

    /////////////////////////
    // find implementation //
    /////////////////////////

    namespace detail
    {
        template<class ForwardIt, class T>
        auto my_binary_search(ForwardIt first, ForwardIt last, const T& value)
        {
            auto comp = [](const auto& interval, auto value)
            {
                return interval.end < value;
            };
            auto result = std::lower_bound(first, last, value, comp);

            if (!(result == last) && !(comp(*result, value)))
            {
                if (result->contains(value))
                {
                    return static_cast<int>(std::distance(first, result));
                }
                else
                {
                    return -1;
                }
            }
            else{
                return -1;
            }
        }

        template <std::size_t dim, class TInterval,
                  class index_t = typename TInterval::index_t,
                  class coord_index_t = typename TInterval::coord_index_t>
        inline auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                              std::size_t start_index, std::size_t end_index,
                              std::array<coord_index_t, dim> coord,
                              std::integral_constant<std::size_t, 0>) -> index_t
        {
            auto find_index = my_binary_search(lca[0].cbegin() + start_index, lca[0].cbegin() + end_index, coord[0]);

            return (find_index != -1)? find_index + start_index: find_index;
        }

        template <std::size_t dim, class TInterval,
                  class index_t = typename TInterval::index_t,
                  class coord_index_t = typename TInterval::coord_index_t,
                  std::size_t N>
        inline auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                              std::size_t start_index, std::size_t end_index,
                              std::array<coord_index_t, dim> coord,
                              std::integral_constant<std::size_t, N>) -> index_t
        {
            index_t find_index = my_binary_search(lca[N].cbegin() + start_index, lca[N].cbegin() + end_index, coord[N]);

            if (find_index != -1)
            {
                auto off_ind = static_cast<std::size_t>(lca[N][find_index + start_index].index + coord[N]);
                find_index = find_impl(lca, lca.offsets(N)[off_ind], lca.offsets(N)[off_ind + 1],
                                        coord, std::integral_constant<std::size_t, N - 1>{});
            }
            return find_index;
        }
    } // namespace detail

    template <std::size_t dim, class TInterval,
              class index_t = typename TInterval::index_t,
              class coord_index_t = typename TInterval::coord_index_t>
    inline auto find(const LevelCellArray<dim, TInterval>& lca, std::array<coord_index_t, dim> coord) -> index_t
    {
        return detail::find_impl(lca, 0, lca[dim - 1].size(), coord, std::integral_constant<std::size_t, dim - 1>{});
    }

    template <std::size_t dim, class TInterval,
              class coord_index_t = typename TInterval::coord_index_t,
              class index_t = typename TInterval::index_t>
    inline auto find_on_dim(const LevelCellArray<dim, TInterval>& lca, std::size_t d,
                            std::size_t start_index, std::size_t end_index,
                            coord_index_t coord)
    {
        index_t find_index = detail::my_binary_search(lca[d].cbegin() + start_index, lca[d].cbegin() + end_index, coord);

        return (find_index != -1)? find_index + start_index: std::numeric_limits<std::size_t>::max();
    }

} // namespace samurai