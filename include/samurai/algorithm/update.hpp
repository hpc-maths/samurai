// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>

#include <xtensor/xfixed.hpp>

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"
#include "../concepts.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../subset/node.hpp"
#include "../timers.hpp"
#include "graduation.hpp"
#include "utils.hpp"

#ifndef NDEBUG
#include "../io/hdf5.hpp"
#endif

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <xtensor/xmasked_view.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    template <class Field, class... Fields>
    void update_ghost(Field& field, Fields&... fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        auto& mesh            = field.mesh();
        std::size_t max_level = mesh.max_level();

        update_outer_ghosts(max_level, field, fields...);
        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::proj_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, fields...));
            update_outer_ghosts(level - 1, field, fields...);
        }

        update_outer_ghosts(0, field, fields...);
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= max_level; ++level)
        {
            auto set_at_level = intersection(mesh[mesh_id_t::pred_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level);
            set_at_level.apply_op(variadic_prediction<pred_order, false>(field, fields...));
        }
    }

    template <class Field>
    void update_ghost_mro(Field& field)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;
        auto& mesh                       = field.mesh();

        std::size_t max_level = mesh.max_level();

        update_outer_ghosts(max_level, field);
        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(projection(field));
            update_outer_ghosts(level - 1, field);
        }

        update_outer_ghosts(0, field);
        for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= max_level; ++level)
        {
            // We eliminate the overleaves from the computation since they
            // are done separately
            // auto expr =
            // difference(intersection(difference(mesh[mesh_id_t::all_cells][level],
            //                                                union_(mesh[mesh_id_t::cells][level],
            //                                                       mesh[mesh_id_t::proj_cells][level])),
            //                                     mesh.domain()),
            //                        difference(mesh[mesh_id_t::overleaves][level],
            //                                   union_(mesh[mesh_id_t::union_cells][level],
            //                                          mesh[mesh_id_t::cells_and_ghosts][level])))
            //             .on(level);

            auto expr = intersection(
                difference(mesh[mesh_id_t::all_cells][level], union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level])),
                self(mesh.domain()).on(level));

            expr.apply_op(prediction<pred_order, false>(field));
        }
    }

    template <class Field>
    void project_bc(std::size_t proj_level, const DirectionVector<Field::dim>& direction, int layer, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        assert(layer > 0 && layer <= Field::mesh_t::config::max_stencil_width);

        auto& mesh = field.mesh();

        auto domain = self(mesh.domain()).on(proj_level);

        auto& inner = mesh.get_union()[proj_level];
        // auto inner = self(mesh[mesh_id_t::cells][proj_level + 1]).on(proj_level);

        // We want only 1 layer (the further one),
        // so we remove all closer layers by making the difference with the domain translated by (layer - 1) * direction
        auto outside_layer     = difference(translate(inner, layer * direction), translate(domain, (layer - 1) * direction));
        auto projection_ghosts = intersection(outside_layer, mesh[mesh_id_t::reference][proj_level]).on(proj_level);

        if (mesh.domain().is_box())
        {
            project_bc(projection_ghosts, proj_level, direction, layer, field);
        }
        else
        {
            // We don't want to fill by projection the ghosts that have been/will be filled by the B.C. in other directions.
            // This can happen when there is a hole in the domain.

            std::size_t n_bc_ghosts = Field::mesh_t::config::max_stencil_width;
            if (!field.get_bc().empty())
            {
                n_bc_ghosts = field.get_bc().front()->stencil_size() / 2;
            }

            auto bc_ghosts_in_other_directions  = domain_boundary_outer_layer(mesh, proj_level, n_bc_ghosts);
            auto projection_ghosts_no_bc_ghosts = difference(projection_ghosts, bc_ghosts_in_other_directions);

            project_bc(projection_ghosts_no_bc_ghosts, proj_level, direction, layer, field);
        }
    }

    template <class Subset, class Field>
    void project_bc(Subset& projection_ghosts,
                    std::size_t proj_level,
                    [[maybe_unused]] const DirectionVector<Field::dim>& direction,
                    [[maybe_unused]] int layer,
                    Field& field)
    {
        using mesh_id_t  = typename Field::mesh_t::mesh_id_t;
        using interval_t = typename Field::mesh_t::interval_t;
        using lca_t      = typename Field::mesh_t::lca_type;

        auto& mesh = field.mesh();
        lca_t proj_ghost_lca(proj_level, mesh.origin_point(), mesh.scaling_factor());

        projection_ghosts(
            [&](const auto& i, const auto& index)
            {
                field(proj_level, i, index) = 0; // Initialize the sums to 0 to compute the average

                interval_t i_cell = {i.start, i.start + 1};

                for (auto ii = i.start; ii < i.end; ++ii, i_cell += 1)
                {
                    proj_ghost_lca.add_point_back(ii, index); // this LCA stores only the current ghost we need to fill
                    int n_children = 0;

                    // We loop over the upper levels to find children.
                    // 99% of the time, children are found at level+1, but if there is no children there, we search at level+2
                    // (this can actually happen in the lid-driven cavity)
                    for (auto children_level = proj_level + 1; children_level <= proj_level + 2; ++children_level)
                    {
                        // We retrieve the children of the current ghost by intersecting it with the upper level
                        auto children = intersection(self(proj_ghost_lca).on(children_level), mesh[mesh_id_t::reference][children_level]);
                        // We iterate over the children and add their values to the current ghost in order to compute the average
                        children(
                            [&](const auto& i_child, const auto& index_child)
                            {
                                for (auto ii_child = i_child.start; ii_child < i_child.end; ++ii_child)
                                {
#ifdef SAMURAI_CHECK_NAN
                                    if (xt::any(xt::isnan(field(children_level, {ii_child, ii_child + 1}, index_child))))
                                    {
                                        std::cerr << std::endl;
#ifdef SAMURAI_WITH_MPI
                                        mpi::communicator world;
                                        std::cerr << "[" << world.rank() << "] ";
#endif
                                        std::cerr << "NaN found in field(" << children_level << "," << ii_child << "," << index_child
                                                  << ") during projection of the B.C. into the cell at (" << proj_level << ", " << ii << ", "
                                                  << index << ")   (dir = " << direction << ", layer = " << layer << ")" << std::endl;
#ifndef NDEBUG
                                        save(fs::current_path(), "update_ghosts", {true, true}, mesh, field);
#endif
                                        std::exit(1);
                                    }
#endif
                                    field(proj_level, i_cell, index) += field(children_level, {ii_child, ii_child + 1}, index_child);
                                    n_children++;
                                }
                            });
                        // If we found children, we break the loop. Otherwise, we continue to search at the next level
                        if (n_children > 0)
                        {
                            break;
                        }
                    }
                    if (n_children > 0)
                    {
                        // We divide the sum by the number of children to get the average
                        field(proj_level, i_cell, index) /= n_children;
                    }
#ifndef NDEBUG
#ifndef SAMURAI_WITH_MPI
                    else
                    {
                        // I'm not sure if this can happen in normal conditions, so I put this error message in debug mode only.
                        // However, it can happen in normal conditions if the domain has a hole, so we don't raise an error in that case.
                        if (mesh.domain().is_box())
                        {
                            std::cerr << "No children found for the ghost at level " << proj_level << ", i = " << ii
                                      << ", index = " << index << " during projection of the B.C. into the cell at level " << proj_level
                                      << ", i=" << i_cell << ", index=" << index << std::endl;
                            save(fs::current_path(), "update_ghosts", {true, true}, mesh, field);
                            assert(false);
                        }
                    }
#endif
#endif
                    proj_ghost_lca.clear();
                }
            });
    }

    /**
     * Project the B.C. from level+1 to level:
     * For projection ghost, we compute the average of its children,
     * and we do that layer by layer.
     * For instance, if max_stencil_width = 3, then 3 fine boundary ghosts overlap 2 coarse ghosts.
     * Note that since we want to project the B.C. two levels down, it is done in two steps:
     * - the B.C. is projected onto the lower ghosts
     * - those lower ghosts are projected onto the even lower ghosts
     */
    template <class Field>
    void project_bc(std::size_t level, Field& field)
    {
        auto& mesh = field.mesh();

        if (level < mesh.max_level() && level >= (mesh.min_level() > 0 ? mesh.min_level() - 1 : 0))
        {
            static constexpr std::size_t max_stencil_width = Field::mesh_t::config::max_stencil_width;
            int max_coarse_layer = static_cast<int>(max_stencil_width % 2 == 0 ? max_stencil_width / 2 : (max_stencil_width + 1) / 2);

            for_each_cartesian_direction<Field::dim>(
                [&](auto direction_index, const auto& direction)
                {
                    if (!mesh.is_periodic(direction_index))
                    {
                        for (int layer = 1; layer <= max_coarse_layer; ++layer)
                        {
                            project_bc(level, direction, layer, field);
                        }
                    }
                });
        }
    }

    template <class Field, class... Fields>
    void project_bc(std::size_t level, Field& field, Fields&... other_fields)
    {
        project_bc(level, field);
        project_bc(level, other_fields...);
    }

    template <class Field>
    void predict_bc(std::size_t pred_level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        std::size_t n_bc_ghosts = Field::mesh_t::config::max_stencil_width;
        if (!field.get_bc().empty())
        {
            n_bc_ghosts = field.get_bc().front()->stencil_size() / 2;
        }

        // auto& cells = mesh[mesh_id_t::cells][pred_level - 1];
        //  auto cells                     = domain_boundary(mesh, pred_level - 1, direction);
        // auto bc_ghosts = difference(translate(cells, n_bc_ghosts * direction), self(mesh.domain()).on(pred_level - 1));
        auto bc_ghosts = domain_boundary_outer_layer(mesh, pred_level - 1, direction, n_bc_ghosts);

        auto outside_prediction_ghosts = intersection(bc_ghosts, mesh[mesh_id_t::reference][pred_level]).on(pred_level);

        if (mesh.domain().is_box())
        {
            predict_bc(outside_prediction_ghosts, pred_level, direction, field);
        }
        else
        {
            // We don't want to fill by prediction the ghosts that have been/will be filled by the B.C. in other directions.
            // This can happen when there is a hole in the domain.

            auto bc_ghosts_in_other_directions  = domain_boundary_outer_layer(mesh, pred_level, n_bc_ghosts);
            auto prediction_ghosts_no_bc_ghosts = difference(outside_prediction_ghosts, bc_ghosts_in_other_directions);

            predict_bc(prediction_ghosts_no_bc_ghosts, pred_level, direction, field);
        }
    }

    template <class Subset, class Field>
    void
    predict_bc(Subset& prediction_ghosts, std::size_t pred_level, [[maybe_unused]] const DirectionVector<Field::dim>& direction, Field& field)
    {
        using interval_t = typename Field::mesh_t::interval_t;

        prediction_ghosts(
            [&](const auto& i, const auto& index)
            {
                interval_t i_cell = {i.start, i.start + 1};
                for (auto ii = i.start; ii < i.end; ++ii, i_cell += 1)
                {
#ifdef SAMURAI_CHECK_NAN
                    if (xt::any(xt::isnan(field(pred_level - 1, i_cell >> 1, index >> 1))))
                    {
                        std::cerr << std::endl;
#ifdef SAMURAI_WITH_MPI
                        mpi::communicator world;
                        std::cerr << "[" << world.rank() << "] ";
#endif
                        std::cerr << "NaN found in field(" << (pred_level - 1) << "," << (i_cell >> 1) << "," << (index >> 1)
                                  << ") during prediction of the B.C. into the cell at (" << pred_level << ", " << ii << ", " << index
                                  << ") " << std::endl;
#ifndef NDEBUG
                        samurai::save(fs::current_path(), "update_ghosts", {true, true}, field.mesh(), field);
#endif
                        std::exit(1);
                    }
#endif
                    field(pred_level, i_cell, index) = field(pred_level - 1, i_cell >> 1, index >> 1);
                }
            });
    }

    /**
     * This function projects the outer corner two levels down.
     */
    template <class Field>
    void project_corner_below(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        if (level == 0)
        {
            return;
        }

        auto& mesh = field.mesh();

        for (std::size_t delta_l = 1; delta_l <= 2; ++delta_l) // lower level (1 or 2)
        {
            auto proj_level = level - delta_l;

            auto fine_inner_corner = self(mesh.corner(direction)).on(level);
            auto fine_outer_corner = intersection(translate(fine_inner_corner, direction), mesh[mesh_id_t::reference][level]);
            auto projection_ghost  = intersection(fine_outer_corner.on(proj_level), mesh[mesh_id_t::reference][proj_level]);

            projection_ghost(
                [&](const auto& i, const auto& index)
                {
                    using index_t = std::decay_t<decltype(index)>;

                    auto i_child = (1 << delta_l) * i;
                    i_child.start += direction[0] == -1 ? ((1 << delta_l) - 1) : 0;
                    i_child.end         = i_child.start + 1;
                    i_child.step        = 1;
                    index_t index_child = (1 << delta_l) * index;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        index_child[d] += direction[d + 1] == -1 ? ((1 << delta_l) - 1) : 0;
                    }
                    field(proj_level, i, index) = field(level, i_child, index_child);
                });
            if (proj_level == 0)
            {
                break;
            }
        }
    }

    template <class Field>
    void update_outer_ghosts(std::size_t level, Field& field)
    {
        static_assert(Field::mesh_t::config::prediction_order <= 1);

        constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        // Project outer corners two levels down
        if constexpr (dim > 1)
        {
            if (level <= mesh.max_level() && level >= mesh.min_level())
            {
                for_each_diagonal_direction<dim>(
                    [&](auto& direction)
                    {
                        auto d = find_direction_index(direction);
                        if (!mesh.is_periodic(d))
                        {
                            update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                            project_corner_below(level, direction, field); // project to level-1 and level-2
                        }
                    });
            }
        }

        // Apply the B.C. at the same level as the cells and project below
        for_each_cartesian_direction<dim>(
            [&](auto direction_index, const auto& direction)
            {
                if (!mesh.is_periodic(direction_index))
                {
                    if (level < mesh.max_level())
                    {
                        // Project the B.C. from level+1 to level:
                        // For projection ghost, we compute the average of its children,
                        // and we do that layer by layer.
                        // For instance, if max_stencil_width = 3, then 3 fine boundary ghosts overlap 2 coarse ghosts.
                        // Note that since we want to project the B.C. two levels down, it is done in two steps:
                        // - the B.C. is projected onto the lower ghosts
                        // - those lower ghosts are projected onto the even lower ghosts

                        std::size_t n_bc_ghosts = Field::mesh_t::config::max_stencil_width;
                        if (!field.get_bc().empty())
                        {
                            n_bc_ghosts = field.get_bc().front()->stencil_size() / 2;
                        }
                        int max_coarse_layer = static_cast<int>(n_bc_ghosts % 2 == 0 ? n_bc_ghosts / 2 : (n_bc_ghosts + 1) / 2);
                        for (int layer = 1; layer <= max_coarse_layer; ++layer)
                        {
                            project_bc(level, direction, layer, field); // project from level+1 to level
                        }
                    }
                    if (level >= mesh.min_level())
                    {
                        // Apply the B.C. at the same level as the cells
                        apply_field_bc(level, direction, field);
                    }
                    if (level < mesh.max_level() && level >= mesh.min_level())
                    {
                        // Predict the B.C. to level+1 (prediction of order 0, which is the same as a projection)
                        predict_bc(level + 1, direction, field);
                    }

                    // // If the B.C. doesn't fill all the ghost layers, we use polynomial extrapolation
                    // // to fill the remaining layers
                    // update_further_ghosts_by_polynomial_extrapolation(level, direction, field);
                }
            });

        // If the B.C. doesn't fill all the ghost layers, we use polynomial extrapolation
        // to fill the remaining layers
        if (level >= mesh.min_level())
        {
            for_each_cartesian_direction<dim>(
                [&](auto direction_index, const auto& direction)
                {
                    if (!mesh.is_periodic(direction_index))
                    {
                        update_further_ghosts_by_polynomial_extrapolation(level, direction, field);
                    }
                });
        }
    }

    /**
     * Updates the outer ghosts:
     * - The outer corners are updated by polynomial extrapolation (and projected below)
     * - The B.C. are applied at the same level as the cells (and projected below)
     */
    template <class Field>
    void update_outer_ghosts(Field& field)
    {
        auto& mesh = field.mesh();

        for (std::size_t level = mesh.max_level(); level >= (mesh.min_level() > 0 ? mesh.min_level() - 1 : 0); --level)
        {
            update_outer_ghosts(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_outer_ghosts(Field& field, Fields&... fields)
    {
        update_outer_ghosts(field);
        update_outer_ghosts(fields...);
    }

    template <class Field, class... Fields>
    void update_outer_ghosts(std::size_t level, Field& field, Fields&... fields)
    {
        update_outer_ghosts(level, field);
        update_outer_ghosts(level, fields...);
    }

    template <class Field>
    void update_ghost_mr_if_needed(Field& field)
    {
        if (!field.ghosts_updated())
        {
            update_ghost_mr(field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_mr_if_needed(Field& field, Fields&... other_fields)
    {
        update_ghost_mr_if_needed(field);
        update_ghost_mr_if_needed(other_fields...);
    }

    template <class Field, class... Fields>
    void update_ghost_mr(Field& field, Fields&... other_fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        times::timers.start("ghost update");

        auto& mesh            = field.mesh();
        auto max_level        = mesh.max_level();
        std::size_t min_level = 0;

        update_outer_ghosts(max_level, field, other_fields...);

        for (std::size_t level = max_level; level > min_level; --level)
        {
            update_ghost_periodic(level, field, other_fields...);
            update_ghost_subdomains(level, field, other_fields...);

            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, other_fields...));

            update_outer_ghosts(level - 1, field, other_fields...);
        }

        if (min_level > 0 && min_level != max_level)
        {
            update_ghost_periodic(min_level - 1, field, other_fields...);
            update_ghost_subdomains(min_level - 1, field, other_fields...);
            update_outer_ghosts(min_level - 1, field, other_fields...);
        }
        update_ghost_periodic(min_level, field, other_fields...);
        update_ghost_subdomains(min_level, field, other_fields...);

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            auto expr        = intersection(pred_ghosts, mesh.subdomain(), mesh[mesh_id_t::all_cells][level - 1]).on(level);

            expr.apply_op(variadic_prediction<pred_order, false>(field, other_fields...));
            update_ghost_periodic(level, field, other_fields...);
            update_ghost_subdomains(level, field, other_fields...);
        }
        // save(fs::current_path(), "update_ghosts", {true, true}, mesh, field);

        field.ghosts_updated() = true;
        ((other_fields.ghosts_updated() = true), ...);

        times::timers.stop("ghost update");
    }

    inline void update_ghost_mr()
    {
    }

    template <class... T>
    inline void update_ghost_mr(std::tuple<T...>& fields)
    {
        std::apply(
            [](T&... tupleArgs)
            {
                update_ghost_mr(tupleArgs...);
            },
            fields);
    }

    template <class... T>
    inline void update_ghost_mr(Field_tuple<T...>& fields)
    {
        update_ghost_mr(fields.elements());
    }

    template <bool to_send, class Field>
    auto outer_subdomain_corner(std::size_t level, Field& field, const typename Field::mesh_t::mpi_subdomain_t& neighbour)
    {
        using mesh_id_t  = typename Field::mesh_t::mesh_id_t;
        using lca_t      = typename Field::mesh_t::lca_type;
        using interval_t = typename Field::mesh_t::interval_t;
        using coord_t    = typename lca_t::coord_type;

        static constexpr std::size_t ghost_width = Field::mesh_t::config::ghost_width;

        ArrayOfIntervalAndPoint<interval_t, coord_t> interval_list;

        auto& mesh = field.mesh();
        for_each_cartesian_direction<Field::dim>(
            [&](auto bdry_direction_index, const auto& bdry_direction)
            {
                if (!mesh.is_periodic(bdry_direction_index))
                {
                    auto domain = self(mesh.domain()).on(level);
                    auto& mesh1 = to_send ? mesh : neighbour.mesh;
                    auto& mesh2 = to_send ? neighbour.mesh : mesh;

                    auto my_boundary_ghosts = difference(
                        intersection(mesh1[mesh_id_t::reference][level],
                                     translate(domain, ghost_width * bdry_direction),
                                     translate(self(mesh1.subdomain()).on(level), ghost_width * bdry_direction)),
                        domain);

                    auto neighbour_outer_corner = intersection(my_boundary_ghosts, mesh2[mesh_id_t::reference][level]);
                    neighbour_outer_corner(
                        [&](const auto& i, const auto& index)
                        {
                            interval_list.push_back(i, index);
                        });
                }
            });

        interval_list.sort_intervals();

        lca_t lca(level);
        for (std::size_t k = 0; k < interval_list.size(); ++k)
        {
            const auto& [i, index] = interval_list[k];
            lca.add_interval_back(i, index);
        }

        return lca;
    }

    template <class Field>
    void update_ghost_subdomains([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        using mesh_t    = typename Field::mesh_t;
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh = field.mesh();
        mpi::communicator world;
        std::vector<std::vector<value_t>> to_send(mesh.mpi_neighbourhood().size());

        std::size_t i_neigh = 0;
        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                  neighbour.mesh[mesh_id_t::reference][level],
                                                  mesh.subdomain())
                                         .on(level);
                out_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(field(level, i, index).begin(), field(level, i, index).end(), std::back_inserter(to_send[i_neigh]));
                    });
                auto subdomain_corners = outer_subdomain_corner<true>(level, field, neighbour);
                for_each_interval(
                    subdomain_corners,
                    [&](const auto, const auto& i, const auto& index)
                    {
                        std::copy(field(level, i, index).begin(), field(level, i, index).end(), std::back_inserter(to_send[i_neigh]));
                    });

                req.push_back(world.isend(neighbour.rank, neighbour.rank, to_send[i_neigh++]));
            }
        }

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                std::vector<value_t> to_recv;
                std::ptrdiff_t count = 0;

                world.recv(neighbour.rank, world.rank(), to_recv);
                auto in_interface = intersection(neighbour.mesh[mesh_id_t::reference][level],
                                                 mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh.subdomain())
                                        .on(level);
                in_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(to_recv.begin() + count,
                                  to_recv.begin() + count + static_cast<ptrdiff_t>(i.size() * Field::n_comp),
                                  field(level, i, index).begin());
                        count += static_cast<ptrdiff_t>(i.size() * Field::n_comp);
                    });
                auto subdomain_corners = outer_subdomain_corner<false>(level, field, neighbour);
                for_each_interval(subdomain_corners,
                                  [&](const auto, const auto& i, const auto& index)
                                  {
                                      std::copy(to_recv.begin() + count,
                                                to_recv.begin() + count + static_cast<ptrdiff_t>(i.size() * Field::n_comp),
                                                field(level, i, index).begin());
                                      count += static_cast<ptrdiff_t>(i.size() * Field::n_comp);
                                  });
            }
        }
        mpi::wait_all(req.begin(), req.end());
#endif
    }

    template <class Field, class... Fields>
    void update_ghost_subdomains(std::size_t level, Field& field, Fields&... other_fields)
    {
        update_ghost_subdomains(level, field);
        update_ghost_subdomains(level, other_fields...);
    }

    template <class Field>
    void update_ghost_subdomains([[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto& mesh     = field.mesh();
        auto max_level = mesh.max_level();
        for (std::size_t level = 0; level <= max_level; ++level)
        {
            update_ghost_subdomains(level, field);
        }
#endif
    }

    template <class Field>
    void update_tag_subdomains([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& tag, [[maybe_unused]] bool erase = false)
    {
#ifdef SAMURAI_WITH_MPI
        using mesh_t    = typename Field::mesh_t;
        using value_t   = typename Field::value_type;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh = tag.mesh();
        mpi::communicator world;
        std::vector<std::vector<value_t>> to_send(mesh.mpi_neighbourhood().size());

        std::size_t i_neigh = 0;
        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                  neighbour.mesh[mesh_id_t::reference][level],
                                                  mesh.subdomain())
                                         .on(level);
                out_interface(
                    [&](const auto& i, const auto& index)
                    {
                        std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(to_send[i_neigh]));
                    });

                req.push_back(world.isend(neighbour.rank, neighbour.rank, to_send[i_neigh++]));
            }
        }

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
            {
                std::vector<value_t> to_recv;
                std::ptrdiff_t count = 0;

                world.recv(neighbour.rank, world.rank(), to_recv);

                auto in_interface = intersection(mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh[mesh_id_t::reference][level],
                                                 neighbour.mesh.subdomain())
                                        .on(level);
                in_interface(
                    [&](const auto& i, const auto& index)
                    {
                        xt::xtensor<value_t, 1> neigh_tag = xt::empty_like(tag(level, i, index));
                        std::copy(to_recv.begin() + count, to_recv.begin() + count + static_cast<std::ptrdiff_t>(i.size()), neigh_tag.begin());
                        if (erase)
                        {
                            tag(level, i, index) = neigh_tag;
                        }
                        else
                        {
                            tag(level, i, index) |= neigh_tag;
                        }
                        count += static_cast<std::ptrdiff_t>(i.size());
                    });
            }
        }
        mpi::wait_all(req.begin(), req.end());

#endif
    }

    template <class Field>
    void check_duplicate_cells([[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        // static constexpr std::size_t dim = Field::dim;
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::cells].min_level();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();
        mpi::communicator world;

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (world.rank() > neighbour.rank)
            {
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto out_interface = intersection(mesh[mesh_id_t::cells][level], neighbour.mesh[mesh_id_t::cells][level]);
                    out_interface(
                        [&](const auto& i, const auto& index)
                        {
                            // delete cell
                            std::cout << fmt::format("fall intersection between {} {} on level {} in {} {}",
                                                     world.rank(),
                                                     neighbour.rank,
                                                     level,
                                                     i,
                                                     index[0])
                                      << std::endl;
                        });
                }
            }
        }
#endif
    }

    template <class Field>
    void keep_only_one_coarse_tag([[maybe_unused]] Field& tag)
    {
#ifdef SAMURAI_WITH_MPI
        constexpr std::size_t dim = Field::dim;
        using mesh_t              = typename Field::mesh_t;
        using mesh_id_t           = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh            = tag.mesh();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();
        mpi::communicator world;

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (world.rank() > neighbour.rank)
            {
                auto min_level = std::max<std::size_t>(1, mesh[mesh_id_t::reference].min_level());

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto out_interface = intersection(mesh[mesh_id_t::cells][level], neighbour.mesh.subdomain()).on(level - 1);
                    out_interface(
                        [&](const auto& i, const auto& index)
                        {
                            if constexpr (dim == 1)
                            {
                                auto mask1 = (tag(level, 2 * i) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1) & static_cast<int>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1) & static_cast<int>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1), mask) = 0;
                            }
                            if constexpr (dim == 2)
                            {
                                auto j     = index[0];
                                auto mask1 = (tag(level, 2 * i, 2 * j) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i, 2 * j) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i, 2 * j), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) = 0;
                            }
                            if constexpr (dim == 3)
                            {
                                auto j     = index[0];
                                auto k     = index[1];
                                auto mask1 = (tag(level, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i, 2 * j, 2 * k), mask)             = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j, 2 * k), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1, 2 * k), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j, 2 * k + 1), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = 0;
                            }
                        });
                }
            }
        }
#endif
    }

    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
        iterate_over_periodic_ghosts(level,
                                     field,
                                     [&](const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
                                     {
                                         field(level, i_ghosts, index_ghosts) = field(level, i_cells, index_cells);
                                     });
    }

    template <class Field, class Func>
    void iterate_over_periodic_ghosts(std::size_t level, Field& field, Func&& copy_values)
    {
#ifdef SAMURAI_WITH_MPI
        using field_value_t = typename Field::value_type;
#endif
        using mesh_id_t        = typename Field::mesh_t::mesh_id_t;
        using config           = typename Field::mesh_t::config;
        using lca_type         = typename Field::mesh_t::lca_type;
        using interval_value_t = typename Field::interval_t::value_t;
        using box_t            = Box<interval_value_t, Field::dim>;

        constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        const auto& domain      = mesh.domain();
        const auto& min_indices = domain.min_indices();
        const auto& max_indices = domain.max_indices();

        const auto& mesh_ref = mesh[mesh_id_t::reference];

        const std::size_t delta_l = domain.level() - level;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> max_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;

        for (std::size_t d = 0; d < dim; ++d)
        {
            min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
            max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
            shift[d]      = 0;
        }
#ifdef SAMURAI_WITH_MPI
        std::vector<mpi::request> req;
        req.reserve(mesh.mpi_neighbourhood().size());
        mpi::communicator world;

        std::vector<std::vector<field_value_t>> field_data_out(mesh.mpi_neighbourhood().size());
        std::vector<field_value_t> field_data_in;
#endif // SAMURAI_WITH_MPI
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                shift[d]                  = (max_indices[d] - min_indices[d]) >> delta_l;
                const auto shift_interval = shift[0];
                const auto shift_index    = xt::view(shift, xt::range(1, _));

                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        copy_values(i, index, i - shift_interval, index - shift_index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        copy_values(i, index, i + shift_interval, index + shift_index);
                    });
#ifdef SAMURAI_WITH_MPI
                size_t neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                    field_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& field_data = field(level, i - shift_interval, index - shift_index);
                            std::copy(field_data.begin(), field_data.end(), std::back_inserter(field_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& field_data = field(level, i + shift_interval, index + shift_index);
                            std::copy(field_data.begin(), field_data.end(), std::back_inserter(field_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, field_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                    world.recv(mpi_neighbor.rank, world.rank(), field_data_in);
                    auto it       = field_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(it, it + std::ssize(field(level, i, index)), field(level, i, index).begin());
                            it += std::ssize(field(level, i, index));
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(it, it + std::ssize(field(level, i, index)), field(level, i, index).begin());
                            it += std::ssize(field(level, i, index));
                        });
                }
                mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
                /* reset variables for next iterations. */
                shift[d]      = 0;
                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
            }
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(std::size_t level, Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(level, field);
        update_ghost_periodic(level, other_fields...);
    }

    template <class Field>
    void update_ghost_periodic(Field& field)
    {
        using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::reference].min_level();
        std::size_t max_level = mesh[mesh_id_t::reference].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_ghost_periodic(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(field);
        update_ghost_periodic(other_fields...);
    }

    template <class Tag>
    void update_tag_periodic(std::size_t level, Tag& tag)
    {
#ifdef SAMURAI_WITH_MPI
        using tag_value_type = typename Tag::value_type;
#endif
        using mesh_id_t           = typename Tag::mesh_t::mesh_id_t;
        using config              = typename Tag::mesh_t::config;
        using lca_type            = typename Tag::mesh_t::lca_type;
        using interval_value_t    = typename Tag::interval_t::value_t;
        using box_t               = Box<interval_value_t, Tag::dim>;
        constexpr std::size_t dim = Tag::dim;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> max_corner;

        auto& mesh           = tag.mesh();
        const auto& mesh_ref = mesh[mesh_id_t::reference];

        auto& domain     = mesh.domain();
        auto min_indices = domain.min_indices();
        auto max_indices = domain.max_indices();

        const std::size_t delta_l = domain.level() - level;

        for (std::size_t d = 0; d < dim; ++d)
        {
            shift[d]      = 0;
            min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
            max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
        }
#ifdef SAMURAI_WITH_MPI
        using tag_value_type = typename Tag::value_type;
        std::vector<mpi::request> req;
        req.reserve(mesh.mpi_neighbourhood().size());
        mpi::communicator world;

        std::vector<std::vector<tag_value_type>> tag_data_out(mesh.mpi_neighbourhood().size());
        std::vector<tag_value_type> tag_data_in;
#endif // SAMURAI_WITH_MPI
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                shift[d]                  = (max_indices[d] - min_indices[d]) >> delta_l;
                const auto shift_interval = shift[0];
                const auto shift_index    = xt::view(shift, xt::range(1, _));

                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level, i, index) |= tag(level, i - shift_interval, index - shift_index);
                        tag(level, i - shift_interval, index - shift_index) |= tag(level, i, index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level, i, index) |= tag(level, i + shift_interval, index + shift_index);
                        tag(level, i + shift_interval, index + shift_index) |= tag(level, i, index);
                    });
#ifdef SAMURAI_WITH_MPI
                // first  pass tag(level, i, index) |= tag(level, i - shift_interval, index - shift_index);
                size_t neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    tag_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& tag_data = tag(level, i - shift_interval, index - shift_index);
                            std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& tag_data = tag(level, i + shift_interval, index + shift_index);
                            std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, tag_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    world.recv(mpi_neighbor.rank, world.rank(), tag_data_in);
                    auto it       = tag_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            for (tag_value_type& tag_xyz : tag(level, i, index))
                            {
                                tag_xyz |= *it;
                                ++it;
                            }
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            for (tag_value_type& tag_xyz : tag(level, i, index))
                            {
                                tag_xyz |= *it;
                                ++it;
                            }
                        });
                }
                mpi::wait_all(req.begin(), req.end());
                // second pass tag(level, i - shift_interval, index - shift_index) |= tag(level, i, index);
                neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    tag_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, tag_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    world.recv(mpi_neighbor.rank, world.rank(), tag_data_in);
                    auto it       = tag_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            auto tag_data = tag(level, i - shift_interval, index - shift_index);
                            for (auto tag_it = tag_data.begin(); tag_it != tag_data.end(); ++tag_it, ++it)
                            {
                                *tag_it |= *it;
                            }
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            auto tag_data = tag(level, i + shift_interval, index + shift_index);
                            for (auto tag_it = tag_data.begin(); tag_it != tag_data.end(); ++tag_it, ++it)
                            {
                                *tag_it |= *it;
                            }
                        });
                }
                mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
                /* reset variables for next iterations. */
                shift[d]      = 0;
                min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
            }
        }
    }

    template <class Field>
    void update_overleaves_mr(Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        update_outer_ghosts(field);
        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            // These are the overleaves which are nothing else
            // because when this procedure is called all the rest
            // should be already with the right value.
            auto overleaves_to_predict = difference(difference(mesh[mesh_id_t::overleaves][level], mesh[mesh_id_t::cells_and_ghosts][level]),
                                                    mesh[mesh_id_t::proj_cells][level]);

            overleaves_to_predict.apply_op(prediction<1, false>(field));
        }
    }

    namespace detail
    {
        template <class PredictionOp, class Mesh, class Field>
        void update_field(PredictionOp&& prediction_op, Mesh& new_mesh, Field& field)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            Field new_field("new_f", new_mesh);
#ifdef SAMURAI_CHECK_NAN
            new_field.fill(std::nan(""));
#else
            new_field.fill(0);
#endif

            auto& mesh = field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::reference][level], new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(new_field, field));
            }

            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_coarsen.apply_op(projection(new_field, field));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_refine.apply_op(std::forward<PredictionOp>(prediction_op)(new_field, field));
            }

            swap(field, new_field);
        }
    }

    template <class PredictionFn, class Mesh>
        requires IsMesh<Mesh>
    void update_fields(PredictionFn&&, Mesh&)
    {
    }

    template <class Mesh>
        requires IsMesh<Mesh>
    void update_fields(Mesh&)
    {
    }

    template <class PredictionFn, class Mesh, class Fields, std::size_t... Is>
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Fields& fields, std::index_sequence<Is...>)
    {
        (detail::update_field(std::forward<PredictionFn>(prediction_fn), new_mesh, std::get<Is>(fields)), ...);
    }

    template <class Mesh, class Fields, std::size_t... Is>
    void update_fields(Mesh& new_mesh, Fields& fields, std::index_sequence<Is...>)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        (detail::update_field(std::forward<prediction_fn_t>(default_config::default_prediction_fn), new_mesh, std::get<Is>(fields)), ...);
    }

    template <class PredictionFn, class Mesh, class... T>
        requires IsMesh<Mesh> && (IsField<T> && ...)
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Field_tuple<T...>& fields)
    {
        update_fields(std::forward<PredictionFn>(prediction_fn), new_mesh, fields.elements(), std::make_index_sequence<sizeof...(T)>{});
    }

    template <class Mesh, class... T>
        requires IsMesh<Mesh> && (IsField<T> && ...)
    void update_fields(Mesh& new_mesh, Field_tuple<T...>& fields)
    {
        update_fields(new_mesh, fields.elements(), std::make_index_sequence<sizeof...(T)>{});
    }

    template <class Mesh, class Field, class... Fields>
        requires IsMesh<Mesh> && IsField<Field> && (IsField<Fields> && ...)
    void update_fields(Mesh& new_mesh, Field& field, Fields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        detail::update_field(std::forward<prediction_fn_t>(default_config::default_prediction_fn), new_mesh, field);
        update_fields(new_mesh, fields...);
    }

    template <class PredictionFn, class Mesh, class Field, class... Fields>
        requires IsMesh<Mesh> && IsField<Field> && (IsField<Fields> && ...)
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Field& field, Fields&... fields)
    {
        detail::update_field(std::forward<PredictionFn>(prediction_fn), new_mesh, field);
        update_fields(std::forward<PredictionFn>(prediction_fn), new_mesh, fields...);
    }

    template <class Tag, class... Fields>
    bool update_field(Tag& tag, Fields&... fields)
    {
        static constexpr std::size_t dim = Tag::dim;
        using mesh_t                     = typename Tag::mesh_t;
        using size_type                  = typename Tag::size_type;
        using mesh_id_t                  = typename Tag::mesh_t::mesh_id_t;
        using cl_type                    = typename Tag::mesh_t::cl_type;

        auto& mesh = tag.mesh();

        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              auto itag = static_cast<size_type>(interval.start + interval.index);
                              for (auto i = interval.start; i < interval.end; ++i)
                              {
                                  if (tag[itag] & static_cast<int>(CellFlag::refine))
                                  {
                                      if (level < mesh.max_level())
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](const auto& stencil)
                                              {
                                                  auto new_index = 2 * index + stencil;
                                                  cl[level + 1][new_index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  else if (tag[itag] & static_cast<int>(CellFlag::keep))
                                  {
                                      cl[level][index].add_point(i);
                                  }
                                  else if (tag[itag] & static_cast<int>(CellFlag::coarsen))
                                  {
                                      if (level > mesh.min_level())
                                      {
                                          cl[level - 1][index >> 1].add_point(i >> 1);
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  itag++;
                              }
                          });

        mesh_t new_mesh = {cl, mesh};

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif
        {
            return true;
        }

        update_fields(new_mesh, fields...);
        tag.mesh().swap(new_mesh);
        return false;
    }
}
