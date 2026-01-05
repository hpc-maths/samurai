// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once
#include <fmt/format.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xmasked_view.hpp>

#include "../array_of_interval_and_point.hpp"
#include "../cell_flag.hpp"
#include "../mesh.hpp"
#include "../print.hpp"
#include "../stencil.hpp"
#include "../subset/node.hpp"
#include "../subset/utils.hpp"
#include "utils.hpp"

namespace samurai
{
    namespace detail
    {
        template <class T>
        inline T start_shift_neg(T value, T shift)
        {
            return shift >= 0 ? value >> shift : value << -shift;
        }

        template <std::size_t d, class Translation, std::size_t dim>
        void get_periodic_directions(const Translation& translation,
                                     int delta,
                                     std::array<bool, dim> is_periodic,
                                     std::vector<DirectionVector<dim>>& directions,
                                     DirectionVector<dim>& current)
        {
            auto next = [&]()
            {
                if constexpr (d == dim - 1)
                {
                    directions.push_back(current);
                }
                else
                {
                    get_periodic_directions<d + 1>(translation, delta, is_periodic, directions, current);
                }
            };

            if (is_periodic[d])
            {
                current[d] = start_shift_neg(-translation[d], delta);
                next();

                current[d] = start_shift_neg(translation[d], delta);
                next();
            }
            current[d] = 0;
            next();
        }

        template <class Translation, std::size_t dim>
        auto get_periodic_directions(const Translation& translation, int delta, const std::array<bool, dim>& is_periodic)
        {
            DirectionVector<dim> current{};
            std::vector<DirectionVector<dim>> directions;
            get_periodic_directions<0>(translation, delta, is_periodic, directions, current);
            directions.pop_back();
            return directions;
        }

        template <std::size_t... Is>
        auto build_union(const auto& fine_lca, const auto& directions, std::index_sequence<Is...>)
        {
            return union_(fine_lca, translate(fine_lca, directions[Is])...);
        }
    }

    ///////////////////////
    // graduate operator //
    ///////////////////////

    template <std::size_t dim, class TInterval>
    class graduate_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(graduate_op)

        template <std::size_t d, class T, class Stencil>
        inline void operator()(Dim<d>, T& tag, const Stencil& s) const
        {
            using namespace xt::placeholders;

            auto tag_func = [&](auto& i_f)
            {
                auto mask = tag(level, i_f - s[0], index - view(s, xt::range(1, _))) & static_cast<int>(CellFlag::refine);
                auto i_c  = i_f >> 1;
                apply_on_masked(tag(level - 1, i_c, index >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });

                auto mask2 = tag(level, i_f - s[0], index - view(s, xt::range(1, _))) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_c, index >> 1),
                                mask2,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::keep);
                                });
            };

            if (auto i_even = i.even_elements(); i_even.is_valid())
            {
                tag_func(i_even);
            }

            if (auto i_odd = i.odd_elements(); i_odd.is_valid())
            {
                tag_func(i_odd);
            }
        }
    };

    template <class T, class Stencil>
    inline auto graduate(T& tag, const Stencil& s)
    {
        return make_field_operator_function<graduate_op>(tag, s);
    }

    template <class Tag, class Stencil>
    void graduation(Tag& tag, const Stencil& stencil)
    {
        auto& mesh      = tag.mesh();
        using mesh_t    = typename Tag::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;

        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level > 0; --level)
        {
            /**
             *
             *        |-----|-----| |-----|-----|
             *                                    --------------->
             *                                                             K
             *        |===========|-----------| |===========|-----------|
             */

            auto ghost_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);

            ghost_subset.apply_op(tag_to_keep<0>(tag));

            /**
             *                 R                                 K     R     K
             *        |-----|-----|=====|   ---------------> |-----|-----|=====|
             *
             */

            auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            auto ghost_width = mesh.cfg().graduation_width();
            assert(ghost_width < 10 && "Graduation not implemented for ghost_width higher than 10");
            // maximum ghost width is set to 9
            static_for<1, 10>::apply(
                [&](auto static_ghost_width_)
                {
                    static constexpr int static_ghost_width = static_cast<int>(static_ghost_width_());
                    if (ghost_width == static_ghost_width)
                    {
                        subset_2.apply_op(tag_to_keep<static_ghost_width>(tag, CellFlag::refine));
                    }
                });

            /**
             *      K     C                          K     K
             *   |-----|-----|   -------------->  |-----|-----|
             *
             *   |-----------|
             *
             */

            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(level - 1);
            keep_subset.apply_op(keep_children_together(tag));

            /**
             * Case 1
             * ======
             *                   R     K R     K
             *                |-----|-----|   --------------> |-----|-----| C or
             * K                                                 R
             *   |-----------| |-----------|
             *
             * Case 2
             * ======
             *                   K     K K     K
             *                |-----|-----|   --------------> |-----|-----| C K
             *   |-----------| |-----------|
             *
             */
            assert(stencil.shape()[1] == Tag::dim);
            for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
            {
                auto s      = xt::view(stencil, i);
                auto subset = intersection(translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level - 1]).on(level);
                subset.apply_op(graduate(tag, s));
            }
        }
    }

    template <class Mesh, std::size_t neighbourhood_width = 1>
    bool is_graduated(const Mesh& mesh, const Stencil<1 + 2 * Mesh::dim * neighbourhood_width, Mesh::dim> stencil = star_stencil<Mesh::dim>())
    {
        bool cond = true;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                {
                    auto s   = xt::view(stencil, is);
                    auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                    set(
                        [&cond](const auto&, const auto&)
                        {
                            cond = false;
                        });
                    if (!cond)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <size_t dim, typename TInterval, typename MeshType, size_t max_size, typename TCoord>
    void list_interval_to_refine_for_graduation(
        const size_t grad_width,
        const CellArray<dim, TInterval, max_size>& ca,
        const LevelCellArray<dim, TInterval>& domain,
        [[maybe_unused]] const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
        const std::array<bool, dim>& is_periodic,
        const std::array<int, dim>& nb_cells_finest_level,
        std::array<ArrayOfIntervalAndPoint<TInterval, TCoord>, CellArray<dim, TInterval, max_size>::max_size>& out)
    {
        const size_t max_level      = ca.max_level() + 1;
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
        const auto list_overlapping_intervals = [&](const auto& lhs_ca, const auto& rhs_ca)
        {
            auto apply_refine = [&](const auto& union_func, auto coarse_level, auto& isIntersectionEmpty)
            {
                for (int width = 1; isIntersectionEmpty and width != max_width; ++width)
                {
                    auto refine_subset = intersection(nestedExpand(union_func, 2 * width), rhs_ca[coarse_level]).on(coarse_level);
                    refine_subset(
                        [&](const auto& x_interval, const auto& yz)
                        {
                            out[coarse_level].push_back(x_interval, yz);
                            isIntersectionEmpty = false;
                        });
                }
            };

            for (size_t fine_level = max_level; fine_level > min_fine_level; --fine_level)
            {
                const int delta_l = int(domain.level() - fine_level);
                auto directions   = detail::get_periodic_directions(nb_cells_finest_level, delta_l, is_periodic);
                auto& fine_lca    = lhs_ca[fine_level];
                for (size_t coarse_level = fine_level - 2; coarse_level > min_level - 1; --coarse_level)
                {
                    bool isIntersectionEmpty = true;
                    switch (directions.size())
                    {
                        case 0:
                            apply_refine(fine_lca, coarse_level, isIntersectionEmpty);
                            break;
                        case 2:
                            apply_refine(detail::build_union(fine_lca, directions, std::make_index_sequence<2>()),
                                         coarse_level,
                                         isIntersectionEmpty);
                            break;
                        case 8:
                            apply_refine(detail::build_union(fine_lca, directions, std::make_index_sequence<8>()),
                                         coarse_level,
                                         isIntersectionEmpty);
                            break;
                        case 26:
                            apply_refine(detail::build_union(fine_lca, directions, std::make_index_sequence<26>()),
                                         coarse_level,
                                         isIntersectionEmpty);
                            break;
                        default:
                            samurai::io::eprint("Warning: Unsupported number of periodic directions ({}) .\n", directions.size());
                    }
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

    template <size_t dim, typename TInterval, size_t max_size, typename TCoord>
    void list_interval_to_refine_for_contiguous_boundary_cells(
        const int max_stencil_radius,
        const CellArray<dim, TInterval, max_size>& ca,
        const LevelCellArray<dim, TInterval>& domain,
        const std::array<bool, dim>& is_periodic,
        std::array<ArrayOfIntervalAndPoint<TInterval, TCoord>, CellArray<dim, TInterval, max_size>::max_size>& out)
    {
        if (max_stencil_radius == 1)
        {
            return;
        }

        const size_t max_level = ca.max_level();
        const size_t min_level = ca.min_level();

        // We want to avoid a flux being computed with ghosts outside of the domain if the cell doesn't touch the boundary,
        // because we only want to apply the B.C. on the cells that touch the boundary.
        // For details and figures, see https://github.com/hpc-maths/samurai/pull/320

        for_each_cartesian_direction<dim>(
            [&](const auto direction_idx, const auto& translation)
            {
                if (not is_periodic[direction_idx])
                {
                    // 1. Jump level --> level-1
                    // Case where the boundary is at level L and the jump is going down to L-1:
                    //     We want to have enough contiguous boundary cells to ensure that the stencil at the lower level
                    //     won't go outside the domain.
                    //     To ensure max_stencil_radius at L-1, we need 2*max_stencil_radius at level L.
                    //     However, since we project the B.C. in the first outside ghost at level L-1, we can reduce the number of
                    //     contiguous cells by 1 at level L-1. This makes, at level L, 2*(max_stencil_radius - 2) contiguous cells.
                    //     (One cell is a real cell, the other is a ghost cell outside of the domain, which makes max_stencil_radius - 2
                    //     ghosts cells inside the domain).

                    int n_contiguous_boundary_cells = std::max(max_stencil_radius, 2 * (max_stencil_radius - 2));

                    if (n_contiguous_boundary_cells > 1)
                    {
                        for (size_t level = max_level; level != min_level; --level)
                        {
                            auto boundaryCells = difference(ca[level], translate(self(domain).on(level), -translation)).on(level);

                            for (int i = 2; i <= n_contiguous_boundary_cells; i += 2)
                            {
                                // Here, the set algebra doesn't work, so we put the translation in a LevelCellArray before computing
                                // the intersection. When the problem is fixed, remove the two following lines and uncomment the line
                                // below.
                                LevelCellArray<dim, TInterval> translated_boundary(translate(boundaryCells, -i * translation));
                                auto refine_subset = intersection(translated_boundary, ca[level - 1]).on(level - 1);
                                // auto refine_subset = intersection(translate(boundaryCells, -i*translation), ca[level-1]).on(level-1);

                                refine_subset(
                                    [&](const auto& x_interval, const auto& yz)
                                    {
                                        out[level - 1].push_back(x_interval, yz);
                                    });
                            }
                        }
                    }

                    // 2. Jump level --> level+1
                    // Case where the boundary is at level L and jump is going up:
                    //    If the number of boundary contiguous cells is >= ceil(max_stencil_radius/2), then there is nothing to do,
                    //    since the half stencil at L+1 will not go out of the domain. Here, we just test if max_stencil_radius > 2 by
                    //    simplicity, but at some point it would be nice to implement the real test. Otherwise, ensuring
                    //    max_stencil_radius contiguous cells at level L+1 is enough.
                    if (max_stencil_radius > 2)
                    {
                        for (size_t level = max_level - 1; level != min_level - 1; --level)
                        {
                            auto boundaryCells = difference(ca[level], translate(self(domain).on(level), -translation));
                            for (int i = 1; i != max_stencil_radius; ++i)
                            {
                                auto refine_subset = translate(
                                                         intersection(translate(boundaryCells, -i * translation), ca[level + 1]).on(level),
                                                         i * translation)
                                                         .on(level);
                                refine_subset(
                                    [&](const auto& x_interval, const auto& yz)
                                    {
                                        out[level].push_back(x_interval, yz);
                                    });
                            }
                        }
                    }
                }
            });
    }

    template <size_t dim, typename TInterval, typename MeshType, size_t max_size, typename TCoord>
    void list_intervals_to_refine(const std::size_t grad_width,
                                  const int max_stencil_radius,
                                  const CellArray<dim, TInterval, max_size>& ca,
                                  const LevelCellArray<dim, TInterval>& domain,
                                  [[maybe_unused]] const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
                                  const std::array<bool, dim>& is_periodic,
                                  const std::array<int, dim>& nb_cells_finest_level,
                                  std::array<ArrayOfIntervalAndPoint<TInterval, TCoord>, CellArray<dim, TInterval, max_size>::max_size>& out)
    {
        list_interval_to_refine_for_graduation(grad_width, ca, domain, mpi_neighbourhood, is_periodic, nb_cells_finest_level, out);
        if (!domain.empty())
        {
            list_interval_to_refine_for_contiguous_boundary_cells(max_stencil_radius, ca, domain, is_periodic, out);
        }
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
                           const LevelCellArray<dim, TInterval>& domain,
                           [[maybe_unused]] const std::vector<MPI_Subdomain<MeshType>>& mpi_neighbourhood,
                           const std::array<bool, dim>& is_periodic,
                           const size_t grad_width      = 1,
                           const int max_stencil_radius = 1 // half of width of the numerical scheme's stencil.
    )
    {
        using ca_type    = CellArray<dim, TInterval, max_size>;
        using coord_type = typename ca_type::lca_type::coord_type;

        const size_t max_level = ca.max_level();
        const size_t min_level = ca.min_level();

        std::array<int, dim> nb_cells_finest_level;

        if (std::any_of(is_periodic.begin(),
                        is_periodic.end(),
                        [](const bool& b)
                        {
                            return b;
                        }))
        {
            const auto& min_indices = domain.min_indices();
            const auto& max_indices = domain.max_indices();

            for (size_t d = 0; d != max_indices.size(); ++d)
            {
                nb_cells_finest_level[d] = max_indices[d] - min_indices[d];
            }
        }

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
            // Then, if the non-graduated is not tagged as keep, we coarsen it
            ca_add_p.clear();
            ca_remove_p.clear();
            list_intervals_to_refine(grad_width, max_stencil_radius, ca, domain, mpi_neighbourhood, is_periodic, nb_cells_finest_level, remove_m_all);

            add_p_interval.clear();
            add_p_inner_stencil.clear();
            add_p_idx.clear();
            for (size_t level = min_level; level != max_level + 1; ++level)
            {
                remove_m_all[level].remove_overlapping_intervals();
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

    template <std::size_t dim, class TInterval, size_t max_size>
    size_t make_graduation(CellArray<dim, TInterval, max_size>& ca, const size_t grad_width = 1)
    {
        struct DummyMesh
        {
        };

        std::vector<MPI_Subdomain<DummyMesh>> mpi_neighbourhood;
        std::array<bool, dim> is_periodic;
        LevelCellArray<dim, TInterval> domain;

        is_periodic.fill(false);
        return make_graduation(ca, domain, mpi_neighbourhood, is_periodic, grad_width);
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
            auto set = difference(union_(old_ca[level], ca_add_m[level], ca_add_p[level]), union_(ca_remove_m[level], ca_remove_p[level]));
            set(
                [&](const auto& x_interval, const auto& yz)
                {
                    new_ca[level].add_interval_back(x_interval, yz);
                });
        }
        return new_ca;
    }

}
