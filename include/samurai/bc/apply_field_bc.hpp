// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm/outer_ghost_sets.hpp"
#include "../boundary.hpp"
#include "../field/concepts.hpp"
#include "../static_dispatch.hpp"
#include "extrapolation_sets.hpp"
#include "polynomial_extrapolation.hpp"
#include <algorithm>

#include <fmt/format.h>
#include <stdexcept>

namespace samurai
{
    namespace detail
    {
        /**
         * The polynomial extrapolation fills the outer ghosts one layer at a time, with a
         * stencil that ends on the ghost to fill. That stencil grows with the layer until
         * it reaches the largest implemented size, then slides outward instead, resting on
         * the ghosts the shallower layers have already filled. It has to keep covering the
         * boundary cell, which caps the number of layers it can fill.
         */
        inline void check_ghost_width_for_polynomial_extrapolation(int ghost_width, std::size_t max_ghost_layers)
        {
            if (ghost_width > static_cast<int>(max_ghost_layers))
            {
                throw std::runtime_error(
                    fmt::format("The outer ghosts are filled by polynomial extrapolation, which reaches {} ghost layers at "
                                "most, but the mesh has a ghost width of {}.\n"
                                "To fix this issue, lower the ghost width (it is half of mesh_config.max_stencil_size()), or raise "
                                "max_stencil_size_implemented_PE in bc/polynomial_extrapolation.hpp and add the extrapolation "
                                "coefficients that go with it.",
                                max_ghost_layers,
                                ghost_width));
            }
        }
    }

    template <class Field, class Subset, std::size_t stencil_size, class Vector>
    void apply_bc_on_subset(Bc<Field>& bc,
                            Field& field,
                            Subset& subset,
                            const StencilAnalyzer<stencil_size, Field::dim>& stencil,
                            const Vector& direction)
    {
        auto bc_function = bc.get_apply_function(std::integral_constant<std::size_t, stencil_size>(), direction);
        if (bc.get_value_type() == BCVType::constant)
        {
            auto value = bc.constant_value();
            for_each_stencil(field.mesh(),
                             subset,
                             stencil,
                             [&, value](auto& cells)
                             {
                                 bc_function(field, cells, value);
                             });
        }
        else if (bc.get_value_type() == BCVType::function)
        {
            assert(stencil.has_origin);
            for_each_stencil(field.mesh(),
                             subset,
                             stencil,
                             [&](auto& cells)
                             {
                                 auto& cell_in    = cells[stencil.origin_index];
                                 auto face_coords = cell_in.face_center(direction);
                                 auto value       = bc.value(direction, cell_in, face_coords);
                                 bc_function(field, cells, value);
                             });
        }
        else
        {
            throw std::runtime_error("Unknown BC type");
        }
    }

    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        auto& region            = bc.get_region();
        auto& region_directions = region.first;
        auto& region_lca        = region.second;
        auto stencil_0          = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());

        for (std::size_t d = 0; d < region_directions.size(); ++d)
        {
            if (region_directions[d] != direction)
            {
                continue;
            }

            bool is_periodic = false;
            for (std::size_t i = 0; i < dim; ++i)
            {
                if (direction(i) != 0 && field.mesh().is_periodic(i))
                {
                    is_periodic = true;
                    break;
                }
            }
            if (!is_periodic)
            {
                bool is_cartesian_direction = is_cartesian(direction);

                if (is_cartesian_direction)
                {
                    auto stencil          = convert_for_direction(stencil_0, direction);
                    auto stencil_analyzer = make_stencil_analyzer(stencil);

                    // Inner cells in the boundary region
                    auto bdry_cells = intersection(mesh[mesh_id_t::cells][level], region_lca[d]).on(level);
                    if (level >= mesh.min_level()) // otherwise there is no cells
                    {
                        apply_bc_on_subset(bc, field, bdry_cells, stencil_analyzer, direction);
                    }
                }
            }
        }
    }

    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, Field& field)
    {
        static_nested_loop<Field::dim, -1, 2>(
            [&](auto& direction)
            {
                if (xt::any(xt::not_equal(direction, 0))) // direction != {0, ..., 0}
                {
                    apply_bc_impl<Field, stencil_size>(bc, level, direction, field);
                }
            });
    }

    /**
     * Is the diagonal @a direction covered by a boundary region declaring @a region_directions?
     *
     * Either the diagonal itself is declared (the default @c Everywhere region enumerates every one
     * of the 3^dim - 1 directions), or every Cartesian component of it is declared - so a wall put
     * on {left, top, bottom} owns the top-left and bottom-left corners, which are corners *of that
     * wall*, but not the bottom-right one, where the neighbouring face carries another condition.
     * A corner between two different boundary conditions is deliberately left alone.
     */
    template <std::size_t dim, class Directions>
    bool diagonal_direction_is_declared(const Directions& region_directions, const DirectionVector<dim>& direction)
    {
        auto declared = [&](const DirectionVector<dim>& d)
        {
            return std::any_of(region_directions.begin(),
                               region_directions.end(),
                               [&](const auto& rd)
                               {
                                   return rd == d;
                               });
        };

        if (declared(direction))
        {
            return true;
        }

        for (std::size_t d = 0; d < dim; ++d)
        {
            if (direction[d] != 0)
            {
                DirectionVector<dim> component;
                component.fill(0);
                component[d] = direction[d];
                if (!declared(component))
                {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Apply a boundary condition on the outer ghosts in a DIAGONAL (non-Cartesian) direction: the
     * domain corners in 2D, the domain edges and vertices in 3D.
     *
     * Only the boundary conditions that ask for it are applied here - see
     * @c Bc::fills_diagonal_directions(). No finite-volume condition does: an FV flux stencil never
     * reads a diagonal ghost, so those ghosts keep being filled by
     * @c update_outer_corners_by_polynomial_extrapolation, unchanged. A lattice-Boltzmann reflection
     * whose velocity set contains a diagonal velocity does: such a scheme streams across the corner,
     * so the corner ghost must carry the wall reflection and not an extrapolation of a distribution
     * function, which is meaningless there.
     *
     * The stencil needs no rotation. A reflection has @c stencil_size == 2, so the diagonal stencil is
     * just {inner, inner + direction}, exact for any direction - unlike @c convert_for_direction(),
     * which builds the rotation taking e1 to @a direction and hence only works for a Cartesian one.
     */
    template <class Field, std::size_t stencil_size>
    void apply_diagonal_bc_impl(Bc<Field>& bc, std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        if constexpr (dim == 1 || stencil_size != 2)
        {
            // 1D has no diagonal direction, and only the 2-point stencil of a reflection is
            // implemented: a wider diagonal stencil would need the general lattice-symmetry
            // machinery, which this does not add.
            return;
        }
        else
        {
            auto& mesh = field.mesh();

            if (level < mesh.min_level() || level > mesh.max_level())
            {
                return;
            }

            for (std::size_t d = 0; d < dim; ++d)
            {
                if (direction[d] != 0 && mesh.is_periodic(d))
                {
                    return; // a periodic axis has no real boundary in that direction
                }
            }

            if (!diagonal_direction_is_declared<dim>(bc.get_region().first, direction))
            {
                return;
            }

            Stencil<2, dim> stencil;
            xt::view(stencil, 0)  = 0;
            xt::view(stencil, 1)  = direction;
            auto stencil_analyzer = make_stencil_analyzer(stencil);

            // mesh.corner(direction) holds the inner corner cells of that diagonal, precomputed on
            // the mesh (the corner extrapolation uses it too). Restricted to the cells that exist at
            // this level: on an adapted mesh the corner is not covered at every level, and iterating
            // ghosts that do not exist is an out-of-bounds access.
            auto corner_cells = intersection(self(mesh.corner(direction)).on(level), mesh[mesh_id_t::cells][level]).on(level);

            apply_bc_on_subset(bc, field, corner_cells, stencil_analyzer, direction);
        }
    }

    /**
     * Apply, in the diagonal @a direction, those boundary conditions of @a field that fill diagonal
     * directions. A no-op for every finite-volume condition.
     */
    template <class Field>
        requires field_like<Field>
    void apply_field_bc_diagonal(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t max_stencil_size_implemented_BC = Bc<Field>::max_stencil_size_implemented;

        for (auto& bc : field.get_bc())
        {
            if (!bc->fills_diagonal_directions())
            {
                continue;
            }

            static_for<1, max_stencil_size_implemented_BC + 1>::apply(
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t i = decltype(integral_constant_i)::value;

                    if (bc->stencil_size() == i)
                    {
                        apply_diagonal_bc_impl<Field, i>(*bc.get(), level, direction, field);
                    }
                });
        }
    }

    template <class Field>
        requires field_like<Field>
    void apply_field_bc(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t max_stencil_size_implemented_BC = Bc<Field>::max_stencil_size_implemented;

        for (auto& bc : field.get_bc())
        {
            // Dispatch on the runtime stencil size (in [1, max_stencil_size_implemented_BC]) to the
            // corresponding compile-time instantiation of apply_bc_impl.
            dispatch_static<1, max_stencil_size_implemented_BC>(static_cast<std::size_t>(bc->stencil_size()),
                                                                [&](auto integral_constant_i)
                                                                {
                                                                    static constexpr std::size_t i = decltype(integral_constant_i)::value;
                                                                    apply_bc_impl<Field, i>(*bc.get(), level, direction, field);
                                                                });
        }
    }

    template <class Field>
        requires field_like<Field>
    void apply_field_bc(Field& field, const DirectionVector<Field::dim>& direction)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            apply_field_bc(level, direction, field);
        }
    }

    template <class Field>
        requires field_like<Field>
    void apply_field_bc(Field& field, std::size_t direction_index)
    {
        DirectionVector<Field::dim> direction;
        direction.fill(0);

        direction[direction_index] = 1;
        apply_field_bc(field, direction);

        direction[direction_index] = -1;
        apply_field_bc(field, direction);
    }

    template <class Field>
        requires field_like<Field>
    void apply_field_bc(std::size_t level, Field& field, std::size_t direction_index)
    {
        DirectionVector<Field::dim> direction;
        direction.fill(0);

        direction[direction_index] = 1;
        apply_field_bc(level, direction, field);

        direction[direction_index] = -1;
        apply_field_bc(level, direction, field);
    }

    template <class Field>
        requires field_like<Field>
    void apply_field_bc(Field& field)
    {
        for_each_cartesian_direction<Field::dim>(
            [&](const auto& direction)
            {
                apply_field_bc(field, direction);
            });
    }

    template <class Field, class... Fields>
        requires(field_like<Field> && (field_like<Fields> && ...))
    void apply_field_bc(Field& field, Fields&... other_fields)
    {
        apply_field_bc(field, other_fields...);
    }

    namespace detail
    {
        // Polynomial extrapolation of a field on the stencils a traversal of
        // extrapolation_sets.hpp reports.
        template <class Field>
        struct FieldExtrapolation
        {
            Field& field;

            template <class Cells>
            void stencil(const Cells& cells)
            {
                PolynomialExtrapolation<Field, std::tuple_size_v<Cells>>::extrapolate(field, cells);
            }
        };
    }

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        if constexpr (Field::dim > 1)
        {
            if (detail::with_outer_ghost_tables(field.mesh(),
                                                level,
                                                [&](const auto& outer)
                                                {
                                                    const auto code = detail::direction_code(direction);
                                                    if (code >= outer.corner_extrapolation.size()
                                                        || !outer.corner_extrapolation[code].recorded)
                                                    {
                                                        return false;
                                                    }
                                                    detail::run_extrapolation(field, outer.corner_extrapolation[code].sequence);
                                                    return true;
                                                }))
            {
                return;
            }
            detail::FieldExtrapolation<Field> visitor{field};
            detail::visit_corner_extrapolation(field.mesh(), level, direction, visitor);
        }
    }

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, Field& field)
    {
        static constexpr std::size_t dim = Field::dim;

        if constexpr (dim == 1)
        {
            return; // No outer corners in 1D
        }

        for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                bool is_periodic = false;
                for (std::size_t i = 0; i < dim; ++i)
                {
                    if (direction(i) != 0 && field.mesh().is_periodic(i))
                    {
                        is_periodic = true;
                        break;
                    }
                }
                if (!is_periodic)
                {
                    update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                }
            });
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        int ghost_width = field.mesh().ghost_width();

        detail::check_ghost_width_for_polynomial_extrapolation(ghost_width, detail::max_ghost_layers_implemented_PE);

        // 1. The ghosts further than those the B.C. fills, where there are boundary cells.
        // 2. The ghosts further than those the projection of the B.C. fills, where there are ghosts below boundary cells.
        int ghost_layers_filled_by_bc = 0;
        for (auto& bc : field.get_bc())
        {
            ghost_layers_filled_by_bc = std::max(ghost_layers_filled_by_bc, bc->stencil_size() / 2);
        }
        const int ghost_layers_filled_by_projection_bc = 1;

        if (detail::with_outer_ghost_tables(
                field.mesh(),
                level,
                [&](const auto& outer)
                {
                    const auto d = detail::cartesian_direction_index(direction);
                    if (d >= outer.extrapolation_on_cells.size() || outer.extrapolation_on_cells[d].size() == 0)
                    {
                        return false;
                    }
                    // the layers past those this B.C. fills, recorded the first time a field asks for them
                    for (int layer = ghost_layers_filled_by_bc + 1; layer <= ghost_width; ++layer)
                    {
                        const auto& sequence = outer.extrapolation_on_cells[d].get(
                            layer,
                            [&](auto& to_record)
                            {
                                detail::record_extrapolation_on_cells(field.mesh(), level, direction, layer, to_record);
                            });
                        detail::run_extrapolation(field, sequence);
                    }
                    for (int layer = ghost_layers_filled_by_projection_bc + 1; layer <= ghost_width; ++layer)
                    {
                        detail::run_extrapolation(field, outer.extrapolation_on_ghosts[d][static_cast<std::size_t>(layer - 1)]);
                    }
                    return true;
                }))
        {
            return;
        }

        detail::FieldExtrapolation<Field> visitor{field};
        detail::visit_extrapolation_on_cells(field.mesh(), level, direction, ghost_layers_filled_by_bc + 1, ghost_width, visitor);
        detail::visit_extrapolation_on_ghosts(field.mesh(), level, direction, ghost_layers_filled_by_projection_bc + 1, ghost_width, visitor);
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(Field& field, const DirectionVector<Field::dim>& direction)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            update_further_ghosts_by_polynomial_extrapolation(level, direction, field);
        }
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(Field& field)
    {
        for_each_cartesian_direction<Field::dim>(
            [&](const auto& direction)
            {
                update_further_ghosts_by_polynomial_extrapolation(field, direction);
            });
    }

    template <class Field, class... Fields>
    void update_further_ghosts_by_polynomial_extrapolation(Field& field, Fields&... other_fields)
    {
        update_further_ghosts_by_polynomial_extrapolation(field);
        update_further_ghosts_by_polynomial_extrapolation(other_fields...);
    }
}
