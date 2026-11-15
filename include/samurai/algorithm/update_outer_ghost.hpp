// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <ranges>
#include <span>

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"
#include "../entry_table.hpp"

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
        // A ghost outside the domain and its children in the reference mesh.
        struct BcProjectionEntry
        {
            std::size_t ghost;
            PoolRange children;
        };

        // The ghosts of `level` in the layer `layer` outside the domain in
        // `direction`, facing the union: those project_bc fills with the mean
        // of their children, at level + 1 or, when they have none there, at
        // level + 2. When the domain is not a box, the ghosts within
        // n_bc_ghosts of the boundary in another direction are left out: the
        // B.C. fills them (n_bc_ghosts is 0 on a box).
        template <std::size_t dim>
        struct bc_projection_ghosts
        {
            using entry_type = BcProjectionEntry;

            std::size_t level;
            std::size_t direction;
            std::size_t layer;
            std::size_t n_bc_ghosts;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;
                using lca_t     = typename Mesh::lca_type;

                const auto dir          = direction_from_code<dim>(direction);
                const auto signed_layer = static_cast<int>(layer);
                const auto& domain      = mesh.domain(level);
                const auto& inner       = mesh.get_union()[level];

                // only the layer `layer`: the closer ones are removed with the domain translated by (layer - 1) * direction
                auto outside_layer     = difference(translate(inner, signed_layer * dir), translate(domain, (signed_layer - 1) * dir));
                auto projection_ghosts = intersection(outside_layer, mesh[mesh_id_t::reference][level]).on(level);

                lca_t ghost_lca(level, mesh.origin_point(), mesh.scaling_factor());
                auto record = [&](const auto& i, const auto& index)
                {
                    const auto first = memory_offset(mesh, {level, i.start, index});
                    for (auto ii = i.start; ii < i.end; ++ii)
                    {
                        BcProjectionEntry entry{
                            first + static_cast<std::size_t>(ii - i.start),
                            {emit.pool().size(), 0}
                        };
                        ghost_lca.add_point_back(ii, index); // only the current ghost
                        for (auto children_level = level + 1; children_level <= level + 2 && entry.children.count == 0; ++children_level)
                        {
                            auto children = intersection(self(ghost_lca).on(children_level), mesh[mesh_id_t::reference][children_level]);
                            children(
                                [&](const auto& i_child, const auto& index_child)
                                {
                                    const auto first_child = memory_offset(mesh, {children_level, i_child.start, index_child});
                                    for (std::size_t k = 0; k < i_child.size(); ++k)
                                    {
                                        emit.pool().push_back(first_child + k);
                                        ++entry.children.count;
                                    }
                                });
                        }
                        ghost_lca.clear();
                        emit(entry);
                    }
                };

                if (n_bc_ghosts == 0)
                {
                    projection_ghosts(record);
                }
                else
                {
                    auto bc_ghosts_in_other_directions = domain_boundary_outer_layer(mesh, level, static_cast<int>(n_bc_ghosts));
                    difference(projection_ghosts, bc_ghosts_in_other_directions)(record);
                }
            }
        };

        // The sum of the children from 0, divided by their number.
        template <class Field>
        void project_children(Field& field, const BcProjectionEntry& entry, std::span<const std::size_t> pool)
        {
            using value_t            = typename Field::value_type;
            constexpr std::size_t nc = Field::n_comp;
            auto* data               = field.data();
            const auto children      = pool.subspan(entry.children.first, entry.children.count);
            for (std::size_t c = 0; c < nc; ++c)
            {
                value_t sum = 0;
                for (const auto child : children)
                {
#ifdef SAMURAI_CHECK_NAN
                    if (std::isnan(data[child * nc + c]))
                    {
                        std::cerr << "NaN found in the cell of storage index " << child
                                  << " during projection of the B.C. into the ghost of storage index " << entry.ghost << std::endl;
                        std::exit(1);
                    }
#endif
                    sum += data[child * nc + c];
                }
                if (!children.empty())
                {
                    sum /= static_cast<int>(children.size());
                }
                data[entry.ghost * nc + c] = sum;
            }
        }
    }

    template <class Field>
    void project_bc(std::size_t proj_level, const DirectionVector<Field::dim>& direction, int layer, Field& field)
    {
        const auto& mesh = field.mesh();

        assert(layer > 0 && layer <= mesh.max_stencil_radius());

        // On a domain with a hole, the ghosts the B.C. fills in other directions are not projected.
        int n_bc_ghosts = 0;
        if (!mesh.domain().is_box())
        {
            n_bc_ghosts = field.get_bc().empty() ? mesh.max_stencil_radius() : field.get_bc().front()->stencil_size() / 2;
        }

        for_each_entry(mesh,
                       detail::bc_projection_ghosts<Field::dim>{proj_level,
                                                                direction_code<Field::dim>(direction),
                                                                static_cast<std::size_t>(layer),
                                                                static_cast<std::size_t>(n_bc_ghosts)},
                       [&](const auto& entry, std::span<const std::size_t> pool)
                       {
#if !defined(NDEBUG) && !defined(SAMURAI_WITH_MPI)
                           // It can happen in normal conditions if the domain has a hole, so only a box domain raises an error.
                           assert(entry.children.count > 0 || !mesh.domain().is_box());
#endif
                           detail::project_children(field, entry, pool);
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
        using coord_t   = typename Field::mesh_t::lca_type::all_coord_type;

        static constexpr std::size_t dim = Field::dim;

        if (level == 0)
        {
            return;
        }

        auto& mesh = field.mesh();

        for (std::size_t delta_l = 1; delta_l <= 2; ++delta_l) // lower level (1 or 2)
        {
            auto proj_level = level - delta_l;

            // The source is the corner ghost at `level`, whether the corner has cells at `level`
            // (ghost filled by polynomial extrapolation) or not (ghost filled by the projection
            // from level+1, which has already been done since the levels are processed from fine
            // to coarse). This cascade fills the corner ghosts at every level below the corner
            // cells, even when max_level - min_level > 2.
            auto fine_inner_corner = self(mesh.corner(direction)).on(level);
            auto fine_outer_corner = intersection(translate(fine_inner_corner, direction), mesh[mesh_id_t::reference][level]);
            auto projection_ghost  = intersection(fine_outer_corner.on(proj_level), mesh[mesh_id_t::reference][proj_level]);

            // Each coarse corner ghost is copied from its corner-most child at `level`, the cell at
            // the reconstructed fixed-parity position (ii << delta_l) + offset. projection_ghost keeps
            // a coarse ghost as soon as one of its children lies in the reference array, but on an
            // adapted mesh whose refinement reaches the domain boundary the corner-most child may be
            // absent while another child is present. Reading the absent child would index the cell
            // array with the not-found offset (out of bounds). The child is therefore looked up per
            // cell and copied only when it exists; where it exists it is the single relevant child, so
            // the value matches the former unconditional read.
            const auto& ref_level = mesh[mesh_id_t::reference][level];
            projection_ghost(
                [&](const auto& i, const auto& index)
                {
                    using index_t = std::decay_t<decltype(index)>;

                    index_t index_child = index << delta_l;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        index_child[d] += direction[d + 1] == -1 ? ((1 << delta_l) - 1) : 0;
                    }

                    coord_t child_coord;
                    auto i_child = (i.start << delta_l) + (direction[0] == -1 ? ((1 << delta_l) - 1) : 0);
                    for (auto ii = i.start; ii < i.end; ++ii, i_child += (1 << delta_l))
                    {
                        child_coord[0] = i_child;
                        for (std::size_t d = 1; d < dim; ++d)
                        {
                            child_coord[d] = index_child[d - 1];
                        }

                        if (find(ref_level, child_coord) >= 0)
                        {
                            field(proj_level, {ii, ii + 1}, index) = field(level, {i_child, i_child + 1}, index_child);
                        }
                    }
                });
            if (proj_level == 0)
            {
                break;
            }
        }
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
