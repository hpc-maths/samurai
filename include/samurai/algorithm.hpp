#pragma once

#include <xtensor/xfixed.hpp>

#include "cell.hpp"

namespace samurai
{
    template<std::size_t dim_, class TInterval, std::size_t max_size_>
    class CellArray;

    template<std::size_t dim_, class TInterval>
    class LevelCellArray;

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

    /////////////////////////
    // find implementation //
    /////////////////////////

    namespace detail
    {
        template <std::size_t dim, class TInterval,
                  class index_t = typename TInterval::index_t,
                  class coord_index_t = typename TInterval::coord_index_t>
        inline auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                              std::size_t start_index, std::size_t end_index,
                              std::array<coord_index_t, dim> coord,
                              std::integral_constant<std::size_t, 0>) -> index_t
        {
            for (std::size_t i = start_index; i < end_index; ++i)
            {
                const auto& interval = lca[0][i];
                if (interval.contains(coord[0]))
                {
                    return static_cast<index_t>(i);
                }
            }
            return -1;
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
            index_t find_index = -1;
            for (std::size_t i = start_index; i < end_index; ++i)
            {
                const auto& interval = lca[N][i];
                if (interval.contains(coord[N]))
                {
                    auto off_ind = static_cast<std::size_t>(interval.index + coord[N]);
                    find_index = find_impl(lca, lca.offsets(N)[off_ind], lca.offsets(N)[off_ind + 1],
                                           coord, std::integral_constant<std::size_t, N - 1>{});
                }
            }
            return find_index;
        }
    }

    template <std::size_t dim, class TInterval,
              class index_t = typename TInterval::index_t,
              class coord_index_t = typename TInterval::coord_index_t>
    inline auto find(const LevelCellArray<dim, TInterval>& lca, std::array<coord_index_t, dim> coord) -> index_t
    {
        return detail::find_impl(lca, 0, lca[dim - 1].size(), coord, std::integral_constant<std::size_t, dim - 1>{});
    }

    template <std::size_t dim, class TInterval,
              class coord_index_t = typename TInterval::coord_index_t>
    inline std::size_t find_on_dim(const LevelCellArray<dim, TInterval>& lca, std::size_t d,
                                   std::size_t start_index, std::size_t end_index,
                                   coord_index_t coord)
    {
        for (std::size_t i = start_index; i < end_index; ++i)
        {
            if (lca[d][i].contains(coord))
            {
                return i;
            }
        }
        return std::numeric_limits<std::size_t>::max();
    }

}