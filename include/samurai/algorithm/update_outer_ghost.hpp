// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cmath>
#include <ranges>
#include <span>

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"
#include "../entry_table.hpp"
#include "cell_block.hpp"

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
            bool walk(const Mesh& mesh, Emit& emit) const;

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

        // Same entries by walking the union cells on the face of a box domain,
        // moved `layer` cells out.
        template <std::size_t dim>
        template <class Mesh, class Emit>
        bool bc_projection_ghosts<dim>::walk(const Mesh& mesh, Emit& emit) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            const auto domain = n_bc_ghosts == 0 ? box_domain_block(mesh, level, true) : std::nullopt;
            if (!domain)
            {
                return false;
            }
            const auto dir                 = direction_from_code<dim>(direction);
            const auto& ref                = mesh[mesh_id_t::reference];
            const DirectionVector<dim> out = static_cast<int>(layer) * dir;
            for_each_interval_in_block(mesh.get_union()[level],
                                       face_layer(*domain, dir, 0),
                                       [&](auto start, auto end, const auto& coords, auto)
                                       {
                                           for_each_interval_in_block(
                                               ref[level],
                                               translate(row_block(coords, start, end), out),
                                               [&](auto ghost_start, auto ghost_end, const auto& ghost_coords, auto shift)
                                               {
                                                   auto ghost = ghost_coords;
                                                   for (auto x = ghost_start; x < ghost_end; ++x)
                                                   {
                                                       ghost[0] = x;
                                                       BcProjectionEntry entry{
                                                           static_cast<std::size_t>(x + shift),
                                                           {emit.pool().size(), 0}
                                                       };
                                                       // the children at level + 1 or, when there are none, at level + 2
                                                       for (std::size_t delta_l = 1; delta_l <= 2 && entry.children.count == 0; ++delta_l)
                                                       {
                                                           for_each_interval_in_block(
                                                               ref[level + delta_l],
                                                               refine(cell_block(ghost), delta_l),
                                                               [&](auto child_start, auto child_end, const auto&, auto child_shift)
                                                               {
                                                                   for (auto c = child_start; c < child_end; ++c)
                                                                   {
                                                                       emit.pool().push_back(static_cast<std::size_t>(c + child_shift));
                                                                       ++entry.children.count;
                                                                   }
                                                               });
                                                       }
                                                       emit(entry);
                                                   }
                                               });
                                       });
            return true;
        }

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

    namespace detail
    {
        // The ghosts of `level` below the layer `layer` of B.C. ghosts at
        // level - 1 outside the domain in `direction`, each after its parent:
        // those predict_bc copies from their parent. The layers 1 to n together
        // are the ghosts it fills with n B.C. ghosts. When the domain is not a
        // box, the ghosts within n_bc_ghosts of the boundary in another
        // direction are left out: the B.C. fills them (n_bc_ghosts is 0 on a box).
        template <std::size_t dim>
        struct bc_prediction_ghosts
        {
            using entry_type = Cells<2>;

            std::size_t level;
            std::size_t direction;
            std::size_t layer;
            std::size_t n_bc_ghosts;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                const auto dir    = direction_from_code<dim>(direction);
                const auto upto   = domain_boundary_outer_layer(mesh, level - 1, dir, static_cast<int>(layer));
                const auto before = domain_boundary_outer_layer(mesh, level - 1, dir, static_cast<int>(layer) - 1);
                auto ghosts       = intersection(difference(upto, before), mesh[mesh_id_t::reference][level]).on(level);

                auto record = [&](const auto& i, const auto& index)
                {
                    for (auto ii = i.start; ii < i.end; ++ii)
                    {
                        emit(entry_type{memory_offset(mesh, {level - 1, ii >> 1, index >> 1}), memory_offset(mesh, {level, ii, index})});
                    }
                };
                if (n_bc_ghosts == 0)
                {
                    ghosts(record);
                }
                else
                {
                    auto bc_ghosts_in_other_directions = domain_boundary_outer_layer(mesh, level, static_cast<int>(n_bc_ghosts));
                    difference(ghosts, bc_ghosts_in_other_directions)(record);
                }
            }

            // Same entries by walking the ghosts whose parent is a cell at
            // level - 1 on the face of a box domain, moved `layer` cells out.
            template <class Mesh, class Emit>
            bool walk(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                const auto coarse_domain = n_bc_ghosts == 0 ? box_domain_block(mesh, level - 1, false) : std::nullopt;
                if (!coarse_domain)
                {
                    return false;
                }
                const auto dir                  = direction_from_code<dim>(direction);
                const auto& ref                 = mesh[mesh_id_t::reference];
                const DirectionVector<dim> back = -static_cast<int>(layer) * dir;
                bool found                      = true;
                for_each_interval_in_block(
                    ref[level],
                    refine(face_layer(*coarse_domain, dir, static_cast<int>(layer)), 1),
                    [&](auto start, auto end, const auto& coords, auto shift)
                    {
                        auto parent_coords = coords;
                        std::transform(coords.begin(),
                                       coords.end(),
                                       parent_coords.begin(),
                                       [](auto c)
                                       {
                                           return c >> 1;
                                       });
                        RowLookup parents(ref[level - 1], parent_coords);
                        for_each_interval_in_block(
                            mesh[mesh_id_t::cells][level - 1],
                            translate(row_block(parent_coords, start >> 1, ((end - 1) >> 1) + 1), back),
                            [&](auto cells_start, auto cells_end, const auto&, auto)
                            {
                                const auto first = std::max(start, (cells_start - back[0]) << 1);
                                const auto last  = std::min(end, (cells_end - back[0]) << 1);
                                for (auto x = first; x < last; ++x)
                                {
                                    const auto parent = parents(x >> 1);
                                    found             = found && parent >= 0;
                                    emit(entry_type{static_cast<std::size_t>(parent), static_cast<std::size_t>(x + shift)});
                                }
                            });
                    });
                return found;
            }
        };

        // The corner ghosts at level - 1 and level - 2 below the corner ghosts
        // of `level` in the diagonal `direction`, each after the cell it is
        // copied from: its corner-most child at `level`, when it exists.
        template <std::size_t dim>
        struct corner_ghosts_below
        {
            using entry_type = Cells<2>;

            std::size_t level;
            std::size_t direction;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;
                using coord_t   = typename Mesh::lca_type::all_coord_type;

                if (level == 0)
                {
                    return;
                }
                const auto dir = direction_from_code<dim>(direction);
                for (std::size_t delta_l = 1; delta_l <= 2; ++delta_l) // lower level (1 or 2)
                {
                    auto proj_level = level - delta_l;

                    // The source is the corner ghost at `level`, whether the corner has cells at `level`
                    // (ghost filled by polynomial extrapolation) or not (ghost filled by the projection
                    // from level+1, which has already been done since the levels are processed from fine
                    // to coarse). This cascade fills the corner ghosts at every level below the corner
                    // cells, even when max_level - min_level > 2.
                    auto fine_inner_corner = self(mesh.corner(dir)).on(level);
                    auto fine_outer_corner = intersection(translate(fine_inner_corner, dir), mesh[mesh_id_t::reference][level]);
                    auto projection_ghost  = intersection(fine_outer_corner.on(proj_level), mesh[mesh_id_t::reference][proj_level]);

                    // Each coarse corner ghost is copied from its corner-most child at `level`, the cell at
                    // the reconstructed fixed-parity position (ii << delta_l) + offset. projection_ghost keeps
                    // a coarse ghost as soon as one of its children lies in the reference array, but on an
                    // adapted mesh whose refinement reaches the domain boundary the corner-most child may be
                    // absent while another child is present. The child is therefore looked up per cell and
                    // copied only when it exists.
                    const auto& ref_level = mesh[mesh_id_t::reference][level];
                    projection_ghost(
                        [&](const auto& i, const auto& index)
                        {
                            using index_t = std::decay_t<decltype(index)>;

                            index_t index_child = index << delta_l;
                            for (std::size_t d = 0; d < dim - 1; ++d)
                            {
                                index_child[d] += dir[d + 1] == -1 ? ((1 << delta_l) - 1) : 0;
                            }

                            coord_t child_coord;
                            auto i_child = (i.start << delta_l) + (dir[0] == -1 ? ((1 << delta_l) - 1) : 0);
                            for (auto ii = i.start; ii < i.end; ++ii, i_child += (1 << delta_l))
                            {
                                child_coord[0] = i_child;
                                for (std::size_t d = 1; d < dim; ++d)
                                {
                                    child_coord[d] = index_child[d - 1];
                                }
                                if (find(ref_level, child_coord) >= 0)
                                {
                                    emit(entry_type{memory_offset(mesh, {level, i_child, index_child}),
                                                    memory_offset(mesh, {proj_level, ii, index})});
                                }
                            }
                        });
                    if (proj_level == 0)
                    {
                        break;
                    }
                }
            }

            // Same entries by walking the reference mesh around the corner block.
            template <class Mesh, class Emit>
            bool walk(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                if (level == 0)
                {
                    return true;
                }
                const auto dir    = direction_from_code<dim>(direction);
                const auto corner = corner_block(mesh, dir, level);
                if (!corner)
                {
                    return false;
                }
                const auto& ref        = mesh[mesh_id_t::reference];
                const auto outer_ghost = translate(*corner, dir);
                for (std::size_t delta_l = 1; delta_l <= 2; ++delta_l)
                {
                    const auto proj_level = level - delta_l;
                    for_each_interval_in_block(
                        ref[proj_level],
                        coarsen(outer_ghost, delta_l),
                        [&](auto start, auto end, const auto& coords, auto shift)
                        {
                            auto ghost = coords;
                            for (auto x = start; x < end; ++x)
                            {
                                ghost[0] = x;
                                // a coarse ghost has a child in the outer corner ghosts at `level`
                                bool has_child = false;
                                for_each_interval_in_block(ref[level],
                                                           intersect(refine(cell_block(ghost), delta_l), outer_ghost),
                                                           [&](auto, auto, const auto&, auto)
                                                           {
                                                               has_child = true;
                                                           });
                                if (!has_child)
                                {
                                    continue;
                                }
                                // copied from its corner-most child when it exists
                                auto child = ghost;
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    child[d] = (ghost[d] << delta_l) + (dir[d] == -1 ? ((1 << delta_l) - 1) : 0);
                                }
                                const auto src = cell_index(ref[level], child);
                                if (src >= 0)
                                {
                                    emit(entry_type{static_cast<std::size_t>(src), static_cast<std::size_t>(x + shift)});
                                }
                            }
                        });
                    if (proj_level == 0)
                    {
                        break;
                    }
                }
                return true;
            }
        };
    }

    // Prediction of order 0 of the B.C. ghosts at pred_level - 1 to their
    // children at pred_level: each ghost takes the value of its parent.
    template <class Field>
    void predict_bc(std::size_t pred_level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        const auto& mesh = field.mesh();

        int n_bc_ghosts = mesh.max_stencil_radius();
        if (!field.get_bc().empty())
        {
            n_bc_ghosts = field.get_bc().front()->stencil_size() / 2;
        }
        // On a domain with a hole, the ghosts the B.C. fills in other directions are not predicted.
        const auto excluded = mesh.domain().is_box() ? std::size_t{0} : static_cast<std::size_t>(n_bc_ghosts);
        const auto dir      = direction_code<Field::dim>(direction);

        for (int layer = 1; layer <= n_bc_ghosts; ++layer)
        {
            for_each_entry(mesh,
                           detail::bc_prediction_ghosts<Field::dim>{pred_level, dir, static_cast<std::size_t>(layer), excluded},
                           [&](const auto& cells)
                           {
                               detail::extrapolate_cells(field, cells);
                           });
        }
    }

    /**
     * This function projects the outer corner two levels down.
     */
    template <class Field>
    void project_corner_below(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        for_each_entry(field.mesh(),
                       detail::corner_ghosts_below<Field::dim>{level, direction_code<Field::dim>(direction)},
                       [&](const auto& cells)
                       {
                           detail::extrapolate_cells(field, cells);
                       });
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
