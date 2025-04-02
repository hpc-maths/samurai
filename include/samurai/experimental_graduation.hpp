#pragma once

#include <samurai/cell_flag.hpp>
#include <samurai/mr/operators.hpp>

#include <samurai/array_of_interval_and_point.hpp>
#include <samurai/mesh.hpp>
#include <samurai/static_algorithm.hpp>

namespace samurai
{
    namespace experimental
    {
        namespace detail
        {
            template <size_t index_size, size_t dim, size_t dim_min>
            struct NestedExpand
            {
                static_assert(dim >= dim_min);
                using index_type = xt::xtensor_fixed<int, xt::xshape<index_size>>;

                template <typename TInterval>
                static auto run(index_type& idx, const LevelCellArray<index_size, TInterval>& lca, const int width)
                {
                    if constexpr (dim != dim_min)
                    {
                        idx[dim]       = -width;
                        auto subset_m1 = NestedExpand<index_size, dim - 1, dim_min>::run(idx, lca, width);
                        idx[dim]       = 0;
                        auto subset_0  = NestedExpand<index_size, dim - 1, dim_min>::run(idx, lca, width);
                        idx[dim]       = width;
                        auto subset_1  = NestedExpand<index_size, dim - 1, dim_min>::run(idx, lca, width);

                        return union_(subset_m1, subset_0, subset_1);
                    }
                    else
                    {
                        idx[dim]       = -width;
                        auto subset_m1 = translate(lca, idx);
                        idx[dim]       = 0;
                        auto subset_0  = translate(lca, idx);
                        idx[dim]       = width;
                        auto subset_1  = translate(lca, idx);

                        return union_(subset_m1, subset_0, subset_1);
                    }
                }
            };
        }

        template <size_t index_size, typename TInterval, size_t dim_min = 0, size_t dim_max = index_size>
        auto nestedExpand(const LevelCellArray<index_size, TInterval>& lca, const int width)
        {
            using index_type = typename detail::NestedExpand<index_size, dim_max - 1, dim_min>::index_type;
            index_type idx;
            for (size_t i = 0; i != dim_min; ++i)
            {
                idx[i] = 0;
            }
            for (size_t i = dim_max; i != index_size; ++i)
            {
                idx[i] = 0;
            }
            return detail::NestedExpand<index_size, dim_max - 1, dim_min>::run(idx, lca, width);
        }

        template <size_t dim, typename TInterval, typename MeshType, size_t max_size, typename TCoord>
        void
        list_intervals_to_remove(const size_t grad_width,
                                 const CellArray<dim, TInterval, max_size>& ca,
                                 [[maybe_unused]] const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
                                 const std::array<bool, dim>& is_periodic,
                                 const std::array<int, dim>& nb_cells_finest_level,
                                 std::array<ArrayOfIntervalAndPoint<TInterval, TCoord>, CellArray<dim, TInterval, max_size>::max_size>& out)
        {
            const size_t max_level      = ca.max_level();
            const size_t min_level      = ca.min_level();
            const size_t min_fine_level = (min_level + 2) - 1; // fine_level =  max_level, max_level-1, ..., min_level+2. Thus fine_level !=
                                                               // min_level+2-1
            const int max_width = int(grad_width) + 1;

            for (size_t i = 0; i != max_size; ++i)
            {
                out[i].clear();
            }
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            std::vector<mpi::request> req;
            for (const auto& mpi_neighbor : mpi_neighbourhood)
            {
                req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, ca));
            }
#endif // SAMURAI_WITH_MPI
            const auto list_overlapping_intervals =
                [min_level, min_fine_level, max_level, max_width, &nb_cells_finest_level, &is_periodic, &out](
                    const CellArray<dim, TInterval, max_size>& lhs_ca,
                    const CellArray<dim, TInterval, max_size>& rhs_ca) -> void
            {
                for (size_t fine_level = max_level; fine_level > min_fine_level; --fine_level)
                {
                    for (size_t coarse_level = fine_level - 2; coarse_level > min_level - 1; --coarse_level)
                    {
                        bool isIntersectionEmpty = true;
                        for (int width = 1; isIntersectionEmpty and width != max_width; ++width)
                        {
                            auto refine_subset = intersection(nestedExpand(lhs_ca[fine_level], 2 * width), rhs_ca[coarse_level]).on(coarse_level);
                            refine_subset(
                                [&](const auto& x_interval, const auto& yz)
                                {
                                    out[coarse_level].push_back(x_interval, yz);
                                    isIntersectionEmpty = false;
                                });
                        }
                    }
                }
                xt::xtensor_fixed<int, xt::xshape<dim>> translation = xt::xscalar(0);
                for (size_t d = 0; d != dim; ++d)
                {
                    if (is_periodic[d])
                    {
                        for (size_t fine_level = max_level; fine_level > min_fine_level; --fine_level)
                        {
                            const int delta_l = int(max_level - fine_level);
                            for (size_t coarse_level = fine_level - 2; coarse_level > min_level - 1; --coarse_level)
                            {
                                for (int width = 0; width != max_width; ++width)
                                {
                                    translation[d]     = (nb_cells_finest_level[d] >> delta_l) + 2 * width - 1;
                                    auto refine_subset = intersection(union_(translate(lhs_ca[fine_level], -translation),
                                                                             translate(lhs_ca[fine_level], translation)),
                                                                      rhs_ca[coarse_level])
                                                             .on(coarse_level);
                                    refine_subset(
                                        [&](const auto& x_interval, const auto& yz)
                                        {
                                            out[coarse_level].push_back(x_interval, yz);
                                        });
                                }
                            }
                        }
                        translation[d] = 0;
                    }
                }
            };

            list_overlapping_intervals(ca, ca);
#ifdef SAMURAI_WITH_MPI
            CellArray<dim, TInterval, max_size> neighbor_ca;
            for (const auto& mpi_neighbor : mpi_neighbourhood)
            {
                world.recv(mpi_neighbor.rank, world.rank(), neighbor_ca);
                list_overlapping_intervals(neighbor_ca, ca);
            }
            mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
        }

        // if add the intervals in add_m_interval
        // if dim = 2 then add_m_interval stores the y coord
        // if dim > 2 then add_intercal contains the 'inner_stencil' i.e. the coordinates y+s_x, z+s_z, etc.
        // add_m_idx contains an array of indexes starting from 0 to add_m_interval.size()-1
        // lca_add_m is the destination
        template <size_t dim, typename TInterval, typename TCoord>
        void add_list_of_interval_back(const std::vector<TInterval>& intervals,  // in
                                       const TCoord& yz,                         // in, only used when dim == 2
                                       const std::vector<TCoord>& inner_stencil, // in
                                       std::vector<size_t>& idx,                 // inout
                                       LevelCellArray<dim, TInterval>& lca)      // out
        {
            assert(dim > 1); // cannot be static as the function will be defined (but not called) when dim=1
            if constexpr (dim > 2)
            {
                std::stable_sort(idx.begin(),
                                 idx.end(),
                                 [&inner_stencil](const size_t lhs_idx, const size_t rhs_idx) -> bool
                                 {
                                     const auto& lhs = inner_stencil[lhs_idx];
                                     const auto& rhs = inner_stencil[rhs_idx];
                                     for (size_t i = dim - 2; i != 0; --i)
                                     {
                                         if (lhs[i] < rhs[i])
                                         {
                                             return true;
                                         }
                                         else if (lhs[i] > rhs[i])
                                         {
                                             return false;
                                         }
                                     }
                                     return lhs[0] < rhs[0];
                                 });
            }
            TCoord outer_stencil = xt::xscalar(0);
            for (outer_stencil[dim - 2] = 0; outer_stencil[dim - 2] != 2; ++outer_stencil[dim - 2])
            {
                if constexpr (dim == 2)
                {
                    for (size_t i = 0; i < intervals.size(); ++i)
                    {
                        lca.add_interval_back(intervals[i], yz + outer_stencil);
                    }
                }
                else
                {
                    for (const size_t& i : idx)
                    {
                        lca.add_interval_back(intervals[i], inner_stencil[i] + outer_stencil);
                    }
                }
            } // end for
        }

        template <std::size_t dim, class TInterval, class MeshType, size_t max_size>
        size_t make_graduation(CellArray<dim, TInterval, max_size>& ca,
                               [[maybe_unused]] const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
                               const std::array<bool, dim>& is_periodic,
                               const std::array<int, dim>& nb_cells_finest_level,
                               const size_t grad_width = 1)
        {
            using ca_type    = CellArray<dim, TInterval, max_size>;
            using coord_type = typename ca_type::lca_type::coord_type;

            const size_t max_level = ca.max_level();
            const size_t min_level = ca.min_level();

            std::vector<TInterval> add_p_interval;
            std::vector<coord_type> add_p_inner_stencil;
            std::vector<size_t> add_p_idx;

            std::array<ArrayOfIntervalAndPoint<TInterval, coord_type>, max_size> remove_m_all;

            ca_type ca_add_p;
            ca_type ca_remove_p;
            ca_type new_ca;

            size_t nit;
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            for (nit = 0; mpi::all_reduce(world, new_ca != ca, std::logical_or()); ++nit)
#else
            for (nit = 0; new_ca != ca; ++nit)
#endif // SAMURAI_WITH_MPI
            {
                // test if mesh is correctly graduated.
                // We first build a set of non-graduated cells
                // Then, if the non-graduated is not taged as keep, we coarsen it
                ca_add_p.clear();
                ca_remove_p.clear();
                list_intervals_to_remove(grad_width, ca, mpi_neighbourhood, is_periodic, nb_cells_finest_level, remove_m_all);

                add_p_interval.clear();
                add_p_inner_stencil.clear();
                add_p_idx.clear();
                for (size_t level = min_level; level != max_level + 1; ++level)
                {
#ifdef SAMURAI_WITH_MPI
                    remove_m_all[level].remove_overlapping_intervals();
#else
                    remove_m_all[level].sort_intervals();
#endif // SAMURAI_WITH_MPI
                    const size_t imax = remove_m_all[level].size();
                    for (size_t i = 0; i != imax; ++i)
                    {
                        const auto& x_interval = remove_m_all[level][i].first;
                        const auto& yz         = remove_m_all[level][i].second;
                        ca_remove_p[level].add_interval_back(x_interval, yz);
                        if constexpr (dim == 1)
                        {
                            ca_add_p[level + 1].add_interval_back(2 * x_interval, 2 * yz);
                        }
                        else
                        {
                            nestedLoop<dim - 1, 0, dim - 2>(0,
                                                            2,
                                                            [&](const auto& inner_stencil)
                                                            {
                                                                add_p_interval.push_back(2 * x_interval);
                                                                if constexpr (dim > 2)
                                                                {
                                                                    add_p_inner_stencil.emplace_back(2 * yz + inner_stencil);
                                                                    add_p_idx.push_back(add_p_interval.size() - 1); // std::iota on
                                                                                                                    // the fly
                                                                }
                                                            });
                        }
                        if (dim != 1 and (i + 1 == imax or yz[dim - 2] != remove_m_all[level].get_coord(i + 1)[dim - 2]))
                        {
                            add_list_of_interval_back(add_p_interval, coord_type(2 * yz), add_p_inner_stencil, add_p_idx, ca_add_p[level + 1]);
                            add_p_interval.clear();
                            add_p_inner_stencil.clear();
                            add_p_idx.clear();
                        }
                    } // end for remove_m_all
                } // end for level
                // We then create new_ca as ca U ca_add
                new_ca.clear();
                for (std::size_t level = min_level; level != max_level + 1; ++level)
                {
                    auto set = difference(union_(ca[level], ca_add_p[level]), ca_remove_p[level]);
                    set(
                        [&](const auto& x_interval, const auto& yz)
                        {
                            new_ca[level].add_interval_back(x_interval, yz);
                        });
                }
                //
                std::swap(new_ca, ca);
            }

            return nit - 1;
        }

        template <std::size_t dim, class TInterval, size_t max_size, class Tag>
        CellArray<dim, TInterval, max_size> update_cell_array_from_tag(const CellArray<dim, TInterval, max_size>& old_ca, const Tag& tag)
        {
            using size_type        = unsigned int;
            using value_t          = typename TInterval::value_t;
            using unsigned_value_t = typename std::make_unsigned_t<value_t>;
            using ca_type          = CellArray<dim, TInterval, max_size>;
            using coord_type       = typename ca_type::lca_type::coord_type;

            const auto& mesh = tag.mesh();

            const size_t start_level = old_ca.min_level();
            const size_t end_level   = old_ca.max_level() + 1;

            // create the ensemble of cells to coarsen
            ca_type ca_add_m;
            ca_type ca_remove_m;
            ca_type ca_add_p;
            ca_type ca_remove_p;

            std::vector<TInterval> add_p_interval;
            std::vector<coord_type> add_p_inner_stencil;
            std::vector<size_t> add_p_idx;

            for (size_t level = start_level; level != end_level; ++level)
            {
                const auto begin = old_ca[level].cbegin();
                const auto end   = old_ca[level].cend();
                for (auto it = begin; it != end; ++it)
                {
                    const auto& x_interval = *it;
                    const auto& yz         = it.index();
                    const bool is_yz_even  = dim == 1 or xt::all(xt::equal(yz % 2, 0));

                    for (value_t x = x_interval.start; x < x_interval.end; ++x)
                    {
                        const size_type itag         = static_cast<size_type>(x_interval.index) + static_cast<unsigned_value_t>(x);
                        const bool refine            = tag[itag] & static_cast<int>(CellFlag::refine);
                        const bool coarsenAndNotKeep = tag[itag] & static_cast<int>(CellFlag::coarsen)
                                                   and not(tag[itag] & static_cast<int>(CellFlag::keep));
                        if (refine and level < mesh.max_level())
                        {
                            ca_remove_p[level].add_point_back(x, yz);
                            if constexpr (dim == 1)
                            {
                                ca_add_p[level + 1].add_interval_back({2 * x, 2 * x + 2}, {});
                            }
                            else
                            {
                                nestedLoop<dim - 1, 0, dim - 2>(0,
                                                                2,
                                                                [&](const auto& inner_stencil)
                                                                {
                                                                    add_p_interval.push_back({2 * x, 2 * x + 2});
                                                                    if constexpr (dim > 2)
                                                                    {
                                                                        add_p_inner_stencil.emplace_back(2 * yz + inner_stencil);
                                                                        add_p_idx.push_back(add_p_interval.size() - 1); // std::iota
                                                                                                                        // on the fly
                                                                    }
                                                                });
                            }
                        }
                        else if (coarsenAndNotKeep and level > mesh.min_level())
                        {
                            if (x % 2 == 0 and is_yz_even) // should be modified when using load balancing.
                            {
                                ca_add_m[level - 1].add_point_back(x >> 1, yz >> 1);
                            }
                            ca_remove_m[level].add_point_back(x, yz);
                        }
                    } // end for each x
                    if (dim != 1 and (it + 1 == end or (it + 1).index()[dim - 2] != yz[dim - 2]))
                    {
                        add_list_of_interval_back(add_p_interval, coord_type(2 * yz), add_p_inner_stencil, add_p_idx, ca_add_p[level + 1]);
                        add_p_interval.clear();
                        add_p_inner_stencil.clear();
                        add_p_idx.clear();
                    }
                } // end for each interval
            } // end for each level
            CellArray<dim, TInterval, max_size> new_ca;
            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                auto set = difference(union_(old_ca[level], ca_add_m[level], ca_add_p[level]),
                                      union_(ca_remove_m[level], ca_remove_p[level]));
                set(
                    [&](const auto& x_interval, const auto& yz)
                    {
                        new_ca[level].add_interval_back(x_interval, yz);
                    });
            }
            return new_ca;
        }

    }
}
