// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifdef SAMURAI_WITH_OPENMP
#include <omp.h>
#endif
#include <type_traits>

#include <xtensor/containers/xfixed.hpp>
#include <xtensor/views/xview.hpp>

#include "cell.hpp"
#include "mesh_holder.hpp"

#include "concepts.hpp"
#include "subset/node.hpp"

using namespace xt::placeholders;

namespace samurai
{
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    class CellArray;

    template <std::size_t dim_, class TInterval>
    class LevelCellArray;

    template <class D, class Config>
    class Mesh_base;

    template <class F, class... CT>
    class subset_operator;

    enum class Run
    {
        Sequential,
        Parallel
    };

    enum class Get
    {
        Cells,
        Intervals
    };

    ///////////////////////////////////
    // for_each_level implementation //
    ///////////////////////////////////

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    SAMURAI_INLINE void for_each_level(const CellArray<dim, TInterval, max_size>& ca, Func&& f, bool include_empty_levels = false)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (include_empty_levels || !ca[level].empty())
            {
                f(level);
            }
        }
    }

    template <class Mesh, class Func>
    SAMURAI_INLINE void for_each_level(const Mesh& mesh, Func&& f, bool include_empty_levels = false)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_level(mesh[mesh_id_t::cells], std::forward<Func>(f), include_empty_levels);
    }

    //////////////////////////////////////
    // for_each_interval implementation //
    //////////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    SAMURAI_INLINE void for_each_interval(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        if (!lca.empty())
        {
            for (auto it = lca.cbegin(); it != lca.cend(); ++it)
            {
                f(lca.level(), *it, it.index());
            }
        }
    }

    template <std::size_t dim, class TInterval, class Func>
    SAMURAI_INLINE void for_each_interval(LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        if (!lca.empty())
        {
            for (auto it = lca.begin(); it != lca.end(); ++it)
            {
                f(lca.level(), *it, it.index());
            }
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    SAMURAI_INLINE void for_each_interval(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            for_each_interval(ca[level], std::forward<Func>(f));
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    SAMURAI_INLINE void for_each_interval(CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            for_each_interval(ca[level], std::forward<Func>(f));
        }
    }

    template <mesh_like Mesh, class Func>
    SAMURAI_INLINE void for_each_interval(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::config::mesh_id_t;
        for_each_interval(mesh[mesh_id_t::cells], std::forward<Func>(f));
    }

    template <class Func, class Set>
    SAMURAI_INLINE void for_each_interval(const SetBase<Set>& set, Func&& f)
    {
        set(
            [&](const auto& i, const auto& index)
            {
                f(set.level(), i, index);
            });
    }

    //////////////////////////////////////////
    // for_each_meshinterval implementation //
    //////////////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    SAMURAI_INLINE void for_each_meshinterval(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        using MeshInterval = typename LevelCellArray<dim, TInterval>::mesh_interval_t;

        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            f(MeshInterval(lca.level(), *it, it.index()));
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    SAMURAI_INLINE void for_each_meshinterval(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_meshinterval(ca[level], std::forward<Func>(f));
            }
        }
    }

    template <class MeshIntervalType, class SetType, class Func>
    SAMURAI_INLINE void for_each_meshinterval(SetType& set, Func&& f)
    {
        MeshIntervalType mesh_interval(set.level());
        set(
            [&](const auto& i, const auto& index)
            {
                mesh_interval.i     = i;
                mesh_interval.index = index;
                f(mesh_interval);
            });
    }

    template <class MeshIntervalType, class SetType, class Func>
    SAMURAI_INLINE void parallel_for_each_meshinterval(SetType& set, Func&& f)
    {
#pragma omp parallel
#pragma omp single nowait
        set(
            [&](const auto& i, const auto& index)
            {
#pragma omp task
                {
                    MeshIntervalType mesh_interval(set.level());
                    mesh_interval.i     = i;
                    mesh_interval.index = index;
                    f(mesh_interval);
                }
            });
    }

    template <class MeshIntervalType, Run run_type, class SetType, class Func>
    SAMURAI_INLINE void for_each_meshinterval(SetType& set, Func&& f)
    {
        if constexpr (run_type == Run::Parallel)
        {
            parallel_for_each_meshinterval<MeshIntervalType>(set, std::forward<Func>(f));
        }
        else
        {
            for_each_meshinterval<MeshIntervalType>(set, std::forward<Func>(f));
        }
    }

    //////////////////////////////////
    // for_each_cell implementation //
    //////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    SAMURAI_INLINE void for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        using cell_t        = Cell<dim, TInterval>;
        using index_value_t = typename cell_t::value_t;
        typename cell_t::indices_t index;

        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                index[d + 1] = it.index()[d];
            }

            for (index_value_t i = it->start; i < it->end; ++i)
            {
                index[0] = i;
                cell_t cell{lca.origin_point(), lca.scaling_factor(), lca.level(), index, it->index + i};
                f(cell);
            }
        }
    }

    template <std::size_t dim, class TInterval, class Func>
    SAMURAI_INLINE void parallel_for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        using cell_t        = Cell<dim, TInterval>;
        using index_value_t = typename cell_t::value_t;

#pragma omp parallel
#pragma omp single nowait
        {
            for (auto it = lca.cbegin(); it != lca.cend(); ++it)
            {
#pragma omp task
                for (index_value_t i = it->start; i < it->end; ++i)
                {
                    typename cell_t::indices_t index;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        index[d + 1] = it.index()[d];
                    }
                    index[0] = i;
                    cell_t cell{lca.origin_point(), lca.scaling_factor(), lca.level(), index, it->index + i};
                    f(cell);
                }
            }
        }
    }

    template <Run run_type, std::size_t dim, class TInterval, class Func>
    SAMURAI_INLINE void for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        if constexpr (run_type == Run::Parallel)
        {
            parallel_for_each_cell(lca, std::forward<Func>(f));
        }
        else
        {
            for_each_cell(lca, std::forward<Func>(f));
        }
    }

    template <std::size_t dim, class TInterval, class Func, class F, class... CT>
    SAMURAI_INLINE void for_each_cell(const LevelCellArray<dim, TInterval>& lca, subset_operator<F, CT...> set, Func&& f)
    {
        using cell_t        = Cell<dim, TInterval>;
        using index_value_t = typename cell_t::value_t;
        typename cell_t::indices_t index;

        set(
            [&](const auto& interval, const auto& index_yz)
            {
                index[0]                         = interval.start;
                auto cell_index                  = lca.get_index(index);
                xt::view(index, xt::range(1, _)) = index_yz;
                for (index_value_t i = interval.start; i < interval.end; ++i)
                {
                    index[0] = i;
                    cell_t cell{lca.origin_point(), lca.scaling_factor(), set.level(), index, cell_index++};
                    f(cell);
                }
            });
    }

    template <Run run_type, std::size_t dim, class TInterval, std::size_t max_size, class Func>
    SAMURAI_INLINE void for_each_cell(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_cell<run_type>(ca[level], std::forward<Func>(f));
            }
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    SAMURAI_INLINE void for_each_cell(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for_each_cell<Run::Sequential>(ca, std::forward<Func>(f));
    }

    template <Run run_type, class Mesh, class Func>
    SAMURAI_INLINE void for_each_cell(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_cell<run_type>(mesh[mesh_id_t::cells], std::forward<Func>(f));
    }

    template <class Mesh, class Func>
    SAMURAI_INLINE void for_each_cell(const Mesh& mesh, Func&& f)
    {
        for_each_cell<Run::Sequential>(mesh, std::forward<Func>(f));
    }

    template <class Mesh, class Func>
    SAMURAI_INLINE void for_each_cell(const hold<Mesh>& mesh, Func&& f)
    {
        for_each_cell(mesh.get(), std::forward<Func>(f));
    }

    template <class Mesh, class coord_type, class Func>
    SAMURAI_INLINE void
    for_each_cell(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t& i, const coord_type& index, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using cell_t                     = Cell<dim, typename Mesh::interval_t>;
        using index_value_t              = typename cell_t::value_t;
        typename cell_t::indices_t coord;

        coord[0] = i.start;
        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            coord[d + 1] = index[d];
        }
        auto cell_index = mesh.get_index(level, coord);
        cell_t cell{mesh.origin_point(), mesh.scaling_factor(), level, coord, cell_index};
        for (index_value_t ii = 0; ii < static_cast<index_value_t>(i.size()); ++ii)
        {
            f(cell);
            cell.indices[0]++; // increment x coordinate
            cell.index++;      // increment cell index
        }
    }

    template <class Mesh, class SetType, class Func>
    SAMURAI_INLINE void for_each_cell(const Mesh& mesh, SetType& set, Func&& f)
    {
        set(
            [&](const auto& i, const auto& index)
            {
                for_each_cell(mesh, set.level(), i, index, std::forward<Func>(f));
            });
    }

    /////////////////////////
    // find implementation //
    /////////////////////////

    namespace detail
    {
        // template <class ForwardIt, class T>
        // auto interval_search(ForwardIt first, ForwardIt last, const T& value)
        // {
        //     auto comp = [](const auto& interval, auto v)
        //     {
        //         return interval.end < v;
        //     };

        //     auto result = std::lower_bound(first, last, value, comp);

        //     if (!(result == last) && !(comp(*result, value)))
        //     {
        //         if (result->contains(value))
        //         {
        //             return static_cast<int>(std::distance(first, result));
        //         }
        //     }
        //     return -1;
        // }

        template <class ForwardIt, class T>
        SAMURAI_INLINE auto interval_search(ForwardIt first, ForwardIt last, const T& value)
        {
            for (int dist = 0; first != last; ++first, ++dist)
            {
                if (first->contains(value))
                {
                    return dist;
                }
            }
            return -1;
        }

        // template <class ForwardIt, class T>
        // SAMURAI_INLINE auto interval_search(ForwardIt first, ForwardIt last, const T& value)
        // {
        //     auto it = std::find_if(first,
        //                            last,
        //                            [value](const auto& e)
        //                            {
        //                                return e.contains(value);
        //                            });
        //     return (it == last) ? -1 : static_cast<int>(std::distance(first, it));
        // }

        template <std::size_t dim, class TInterval, class index_t = typename TInterval::index_t, class coord_index_t = typename TInterval::coord_index_t>
        SAMURAI_INLINE auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                                      std::size_t start_index,
                                      std::size_t end_index,
                                      const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& coord,
                                      std::integral_constant<std::size_t, 0>) -> index_t
        {
            using lca_t     = const LevelCellArray<dim, TInterval>;
            using diff_t    = typename lca_t::const_iterator::difference_type;
            auto find_index = interval_search(lca[0].cbegin() + static_cast<diff_t>(start_index),
                                              lca[0].cbegin() + static_cast<diff_t>(end_index),
                                              coord[0]);

            return (find_index != -1) ? find_index + static_cast<diff_t>(start_index) : find_index;
        }

        template <std::size_t dim,
                  class TInterval,
                  class index_t       = typename TInterval::index_t,
                  class coord_index_t = typename TInterval::coord_index_t,
                  std::size_t N>
        SAMURAI_INLINE auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                                      std::size_t start_index,
                                      std::size_t end_index,
                                      const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& coord,
                                      std::integral_constant<std::size_t, N>) -> index_t
        {
            using lca_t        = const LevelCellArray<dim, TInterval>;
            using diff_t       = typename lca_t::const_iterator::difference_type;
            index_t find_index = interval_search(lca[N].cbegin() + static_cast<diff_t>(start_index),
                                                 lca[N].cbegin() + static_cast<diff_t>(end_index),
                                                 coord[N]);

            if (find_index != -1)
            {
                auto off_ind = static_cast<std::size_t>(lca[N][static_cast<std::size_t>(find_index) + start_index].index + coord[N]);
                find_index   = find_impl(lca,
                                       lca.offsets(N)[off_ind],
                                       lca.offsets(N)[off_ind + 1],
                                       coord,
                                       std::integral_constant<std::size_t, N - 1>{});
            }
            return find_index;
        }
    } // namespace detail

    template <std::size_t dim, class TInterval, class index_t = typename TInterval::index_t, class coord_index_t = typename TInterval::coord_index_t>
    SAMURAI_INLINE auto
    find(const LevelCellArray<dim, TInterval>& lca, const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& coord) -> index_t
    {
        return detail::find_impl(lca, 0, lca[dim - 1].size(), coord, std::integral_constant<std::size_t, dim - 1>{});
    }

    template <std::size_t dim, class TInterval, class coord_index_t = typename TInterval::coord_index_t, class index_t = typename TInterval::index_t>
    SAMURAI_INLINE auto
    find_on_dim(const LevelCellArray<dim, TInterval>& lca, std::size_t d, std::size_t start_index, std::size_t end_index, coord_index_t coord)
    {
        using lca_t        = const LevelCellArray<dim, TInterval>;
        using diff_t       = typename lca_t::const_iterator::difference_type;
        index_t find_index = detail::interval_search(lca[d].cbegin() + static_cast<diff_t>(start_index),
                                                     lca[d].cbegin() + static_cast<diff_t>(end_index),
                                                     coord);

        return (find_index != -1) ? static_cast<std::size_t>(find_index) + start_index : std::numeric_limits<std::size_t>::max();
    }

    //----------------------------------------//
    // Find a cell from Cartesian coordinates //
    //----------------------------------------//

    template <std::size_t dim, class TInterval>
    SAMURAI_INLINE auto
    find_cell(const LevelCellArray<dim, TInterval>& lca, const typename LevelCellArray<dim, TInterval>::cell_t::coords_t& cartesian_coords)
    {
        using indices_t = typename LevelCellArray<dim, TInterval>::cell_t::indices_t;
        using value_t   = indices_t::value_type;

        const indices_t indices = xt::cast<value_t>(xt::floor((cartesian_coords - lca.origin_point()) / lca.cell_length()));

        return find_cell(lca, indices);
    }

    template <std::size_t dim, class TInterval, std::size_t max_size>
    SAMURAI_INLINE auto find_cell(const CellArray<dim, TInterval, max_size>& ca,
                                  const typename CellArray<dim, TInterval, max_size>::cell_t::coords_t& cartesian_coords)
    {
        using cell_t = typename CellArray<dim, TInterval, max_size>::cell_t;

        cell_t cell;
        cell.length = 0; // cell not found
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            cell = find_cell(ca[level], cartesian_coords);
            if (cell.length != 0)
            {
                break;
            }
        }
        return cell;
    }

    template <class Mesh>
    SAMURAI_INLINE auto find_cell(const Mesh& mesh, const typename Mesh::cell_t::coords_t& cartesian_coords)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        return find_cell(mesh[mesh_id_t::cells], cartesian_coords);
    }

    //----------------------------------------//
    // Find a cell from indices               //
    //----------------------------------------//

    template <std::size_t dim, class TInterval>
    SAMURAI_INLINE auto
    find_cell(const LevelCellArray<dim, TInterval>& lca, const typename LevelCellArray<dim, TInterval>::cell_t::indices_t& indices)
    {
        using cell_t = typename LevelCellArray<dim, TInterval>::cell_t;

        cell_t cell;
        cell.length = 0; // cell not found

        cell.indices = indices;
        auto offset  = find(lca, cell.indices);
        if (offset >= 0)
        {
            auto interval     = lca[0][static_cast<std::size_t>(offset)];
            cell.index        = interval.index + cell.indices[0];
            cell.level        = lca.level();
            cell.length       = lca.cell_length();
            cell.origin_point = lca.origin_point();
        }
        return cell;
    }

    template <std::size_t dim, class TInterval, std::size_t max_size>
    SAMURAI_INLINE auto find_cell(const CellArray<dim, TInterval, max_size>& ca,
                                  const typename CellArray<dim, TInterval, max_size>::cell_t::indices_t& indices,
                                  const std::size_t level_ref)
    {
        using indices_t = typename CellArray<dim, TInterval, max_size>::cell_t::indices_t;
        using cell_t    = typename CellArray<dim, TInterval, max_size>::cell_t;

        cell_t cell;
        cell.length = 0; // cell not found

        for (std::size_t level = ca.min_level(); level <= level_ref; ++level)
        {
            // level < level_ref -> project indices to a lower level
            const indices_t shifted_indices = indices >> (level_ref - level);

            cell = find_cell(ca[level], shifted_indices);
            if (cell.length != 0)
            {
                return cell;
            }
        }

        return cell;
    }

    template <std::size_t dim, class TInterval, std::size_t max_size>
    SAMURAI_INLINE auto
    find_cell(const CellArray<dim, TInterval, max_size>& ca, const typename CellArray<dim, TInterval, max_size>::cell_t::indices_t& indices)
    {
        return find_cell(ca, indices, ca.max_level());
    }

    template <class Mesh>
    SAMURAI_INLINE auto find_cell(const Mesh& mesh, const typename Mesh::cell_t::indices_t& indices, const std::size_t level_ref)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        return find_cell(mesh[mesh_id_t::cells], indices, level_ref);
    }

    template <class Mesh>
    SAMURAI_INLINE auto find_cell(const Mesh& mesh, const typename Mesh::cell_t::indices_t& indices)
    {
        return find_cell(mesh, indices, mesh.max_level());
    }

} // namespace samurai
