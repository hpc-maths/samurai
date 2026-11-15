// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <ranges>

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"
#include "outer_ghost_sets.hpp"
#include "update_periodic.hpp" // detail::has_ghost_tables

#ifndef NDEBUG
#include "../io/hdf5.hpp"
#endif

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace detail
    {
        // Calls f with the tables of the ghosts outside the (box) domain at `level`
        // when the mesh holds them, and returns what f returns: true when the
        // tables did the work.
        template <class Mesh, class Func>
        bool with_outer_ghost_tables(const Mesh& mesh, std::size_t level, Func&& f)
        {
            if constexpr (has_ghost_tables<Mesh>)
            {
                const auto& outer = mesh.ghost_tables().outer;
                return level < outer.size() && f(outer[level]);
            }
            else
            {
                return false;
            }
        }

        // project_bc on a field: the sum of the children from 0, divided by their number
        template <class Field>
        struct FieldBCProjection
        {
            using value_t = typename Field::interval_t::value_t;
            using index_t = typename Field::mesh_t::lca_type::coord_type;

            Field& field;
            std::size_t proj_level;
            value_t ii{};
            index_t index{};

            void ghost(value_t i, const auto& idx)
            {
                ii                                     = i;
                index                                  = idx;
                field(proj_level, {ii, ii + 1}, index) = 0;
            }

            void child(std::size_t children_level, value_t ii_child, const auto& index_child)
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
                              << ") during projection of the B.C. into the cell at (" << proj_level << ", " << ii << ", " << index << ")"
                              << std::endl;
#ifndef NDEBUG
                    save(fs::current_path(), "update_ghosts", {true, true}, field.mesh(), field);
#endif
                    std::exit(1);
                }
#endif
                field(proj_level, {ii, ii + 1}, index) += field(children_level, {ii_child, ii_child + 1}, index_child);
            }

            void end(int n_children)
            {
                if (n_children > 0)
                {
                    field(proj_level, {ii, ii + 1}, index) /= n_children;
                }
#ifndef NDEBUG
#ifndef SAMURAI_WITH_MPI
                else
                {
                    // It can happen in normal conditions if the domain has a hole, so only a box domain raises an error.
                    if (field.mesh().domain().is_box())
                    {
                        std::cerr << "No children found for the ghost at level " << proj_level << ", i = " << ii << ", index = " << index
                                  << " during projection of the B.C." << std::endl;
                        save(fs::current_path(), "update_ghosts", {true, true}, field.mesh(), field);
                        assert(false);
                    }
                }
#endif
#endif
            }
        };

        // project_corner_below on a field
        template <class Field>
        struct FieldCornerCopy
        {
            Field& field;
            std::size_t level;

            void copy(std::size_t proj_level, auto ii, const auto& index, auto i_child, const auto& index_child)
            {
                field(proj_level, {ii, ii + 1}, index) = field(level, {i_child, i_child + 1}, index_child);
            }
        };
    }

    template <class Field>
    void project_bc(std::size_t proj_level, const DirectionVector<Field::dim>& direction, int layer, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        assert(layer > 0 && layer <= mesh.max_stencil_radius());

        if (detail::with_outer_ghost_tables(mesh,
                                            proj_level,
                                            [&](const auto& outer)
                                            {
                                                const auto& tables = outer.projection[detail::cartesian_direction_index(direction)];
                                                if (static_cast<std::size_t>(layer) > tables.size())
                                                {
                                                    return false;
                                                }
                                                detail::run_bc_projection(field, tables[static_cast<std::size_t>(layer - 1)]);
                                                return true;
                                            }))
        {
            return;
        }

        const auto& domain = mesh.domain(proj_level);

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

            int n_bc_ghosts = mesh.max_stencil_radius();
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
        detail::FieldBCProjection<Field> visitor{field, proj_level};
        detail::visit_bc_projection(field.mesh(), projection_ghosts, proj_level, visitor);
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
            static constexpr std::size_t max_stencil_width = Field::mesh_t::config_t::max_stencil_width;
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

        int n_bc_ghosts = mesh.max_stencil_radius();
        if (!field.get_bc().empty())
        {
            n_bc_ghosts = field.get_bc().front()->stencil_size() / 2;
        }

        if (detail::with_outer_ghost_tables(mesh,
                                            pred_level,
                                            [&](const auto& outer)
                                            {
                                                const auto& tables = outer.prediction[detail::cartesian_direction_index(direction)];
                                                if (static_cast<std::size_t>(n_bc_ghosts) > tables.size())
                                                {
                                                    return false;
                                                }
                                                for (int layer = 1; layer <= n_bc_ghosts; ++layer)
                                                {
                                                    detail::run_copies(field, tables[static_cast<std::size_t>(layer - 1)]);
                                                }
                                                return true;
                                            }))
        {
            return;
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
        detail::FieldCornerCopy<Field> visitor{field, level};
        detail::visit_corner_copies(field.mesh(), level, direction, visitor);
    }

    // Field projection of the corner ghosts at `level` to level-1 and level-2,
    // in every non-periodic diagonal direction.
    template <class Field>
    void project_outer_corners_below(std::size_t level, Field& field)
    {
        constexpr std::size_t dim = Field::dim;
        auto& mesh                = field.mesh();
        if constexpr (dim > 1)
        {
            if (level <= mesh.max_level() && level >= mesh.min_level())
            {
                if (detail::with_outer_ghost_tables(mesh,
                                                    level,
                                                    [&](const auto& outer)
                                                    {
                                                        detail::run_copies(field, outer.corners);
                                                        return true;
                                                    }))
                {
                    return;
                }
                for_each_diagonal_direction<dim>(
                    [&](auto& direction)
                    {
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            if (direction[d] != 0 && mesh.is_periodic(d))
                            {
                                return;
                            }
                        }
                        project_corner_below(level, direction, field);
                    });
            }
        }
    }

    template <class Field, class... Fields>
    void project_outer_corners_below(std::size_t level, Field& field, Fields&... fields)
    {
        project_outer_corners_below(level, field);
        (project_outer_corners_below(level, fields), ...);
    }

    template <class Field>
    void update_outer_ghosts(std::size_t level, Field& field)
    {
        static_assert(Field::mesh_t::config_t::prediction_stencil_radius <= 1);

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
                        // Skip if any non-zero component of the direction is periodic:
                        // a periodic direction has no real boundary, so no corner ghost to fill.
                        bool any_periodic = false;
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            if (direction[d] != 0 && mesh.is_periodic(d))
                            {
                                any_periodic = true;
                                break;
                            }
                        }
                        if (!any_periodic)
                        {
                            update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                            // A boundary condition that owns the diagonal directions overwrites the
                            // extrapolation just applied (no FV condition does; an LBM reflection with
                            // diagonal velocities does). It runs BEFORE project_corner_below so that the
                            // value carried down to the coarser levels is the corrected one.
                            apply_field_bc_diagonal(level, direction, field);
                        }
                    });
            }
        }
        project_outer_corners_below(level, field);

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

                        int n_bc_ghosts = mesh.max_stencil_radius();
                        if (!field.get_bc().empty())
                        {
                            n_bc_ghosts = static_cast<int>(field.get_bc().front()->stencil_size()) / 2;
                        }
                        int max_coarse_layer = n_bc_ghosts % 2 == 0 ? n_bc_ghosts / 2 : (n_bc_ghosts + 1) / 2;
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

        const std::size_t lowest_level = mesh.min_level() > 0 ? mesh.min_level() - 1 : 0;
        for (const std::size_t level : std::views::iota(lowest_level, mesh.max_level() + 1) | std::views::reverse)
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
}
