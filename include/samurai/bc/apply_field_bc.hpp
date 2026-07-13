// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../boundary.hpp"
#include "../field/concepts.hpp"
#include "polynomial_extrapolation.hpp"

#include <stdexcept>

namespace samurai
{
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
     * Apply polynomial extrapolation on the outside ghosts close to boundary cells
     * @param bc The PolynomialExtrapolation boundary condition
     * @param level Level where to apply the polynomial extrapolation
     * @param field Field to apply the extrapolation on
     * @param direction Direction of the boundary
     * @param bdry_cells subset corresponding to boundary cells where to apply the extrapolation on (center of the BC stencil)
     */
    template <std::size_t stencil_size, class Field, class Subset>
    void
    apply_extrapolation_bc_cells(Bc<Field>& bc, std::size_t level, Field& field, const DirectionVector<Field::dim>& direction, Subset& bdry_cells)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        auto stencil_0        = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());
        auto stencil          = convert_for_direction(stencil_0, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        //  We need to check that the furthest ghost exists. It's not always the case for large stencils!
        if constexpr (stencil_size == 2)
        {
            auto cells = intersection(mesh[mesh_id_t::cells][level], bdry_cells).on(level);

            apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
        }
        else
        {
            auto translated_outer_nghbr = translate(mesh[mesh_id_t::reference][level], -(stencil_size / 2) * direction); // can be removed?
            auto cells                  = intersection(translated_outer_nghbr, mesh[mesh_id_t::cells][level], bdry_cells).on(level);

            apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
        }
    }

    template <std::size_t layers, class Mesh>
    auto translated_outer_neighbours(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        static_assert(layers <= 5, "not implemented for layers > 10");

        // Technically, if mesh.domain().is_box(), then we can only test that the furthest layer of ghosts exists
        // (i.e. the set return by the case stencil_size == 2 below).
        // On the other hand, if the domain has holes, we have to check that all the intermediary ghost layers exist.
        // Since we can't easily make the distinction in a static way, we always check that all the ghost layers exist.

        if constexpr (layers == 1)
        {
            return translate(mesh[mesh_id_t::reference][level], -layers * direction);
        }
        else if constexpr (layers == 2)
        {
            // clang-format off
            return intersection(translate(mesh[mesh_id_t::reference][level], -(layers    ) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 1) * direction));
            // clang-format on
        }
        else if constexpr (layers == 3)
        {
            // clang-format off
            return intersection(translate(mesh[mesh_id_t::reference][level], -(layers    ) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 1) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 2) * direction));
            // clang-format on
        }
        else if constexpr (layers == 4)
        {
            // clang-format off
            return intersection(translate(mesh[mesh_id_t::reference][level], -(layers    ) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 1) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 2) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 3) * direction));
            // clang-format on
        }
        else if constexpr (layers == 5)
        {
            // clang-format off
            return intersection(translate(mesh[mesh_id_t::reference][level], -(layers    ) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 1) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 2) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 3) * direction),
                                translate(mesh[mesh_id_t::reference][level], -(layers - 4) * direction));
            // clang-format on
        }
    }

    /**
     * Apply polynomial extrapolation on the outside ghosts close to inner ghosts at the boundary
     * (i.e. inner ghosts in the boundary region that have neighbouring ghosts outside the domain)
     * @param bc The PolynomialExtrapolation boundary condition
     * @param level Level where to apply the polynomial extrapolation
     * @param field Field to apply the extrapolation on
     * @param direction Direction of the boundary
     * @param subset subset corresponding to inner ghosts where to apply the extrapolation on (center of the BC stencil)
     */
    template <std::size_t stencil_size, class Field, class Subset>
    void apply_extrapolation_bc_ghosts(Bc<Field>& bc,
                                       std::size_t level,
                                       Field& field,
                                       const DirectionVector<Field::dim>& direction,
                                       Subset& inner_ghosts_location)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        auto stencil_0        = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());
        auto stencil          = convert_for_direction(stencil_0, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        auto translated_outer_nghbr           = translated_outer_neighbours<stencil_size / 2>(mesh, level, direction);
        auto potential_inner_cells_and_ghosts = intersection(translated_outer_nghbr, inner_ghosts_location).on(level);
        auto inner_cells_and_ghosts           = intersection(potential_inner_cells_and_ghosts, mesh.get_union()[level]).on(level);
        // auto inner_cells_and_ghosts        = intersection(potential_inner_cells_and_ghosts, mesh[mesh_id_t::cells][level + 1]).on(level);
        auto inner_ghosts_with_outer_nghbr = difference(inner_cells_and_ghosts, mesh[mesh_id_t::cells][level]).on(level);
        apply_bc_on_subset(bc, field, inner_ghosts_with_outer_nghbr, stencil_analyzer, direction);
    }

    template <class Field>
        requires field_like<Field>
    void apply_field_bc(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t max_stencil_size_implemented_BC = Bc<Field>::max_stencil_size_implemented;

        for (auto& bc : field.get_bc())
        {
            static_for<1, max_stencil_size_implemented_BC + 1>::apply( // for (int i=1; i<=max_stencil_size_implemented; i++)
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t i = decltype(integral_constant_i)::value;

                    if (bc->stencil_size() == i)
                    {
                        apply_bc_impl<Field, i>(*bc.get(), level, direction, field);
                    }
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

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        if constexpr (Field::dim == 1)
        {
            return; // No outer corners in 1D
        }

        static constexpr std::size_t max_stencil_size_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;

        int ghost_width        = field.mesh().ghost_width();
        const auto& domain     = detail::get_mesh(field.mesh());
        const auto& corner_lca = field.mesh().corner(direction);

        assert(static_cast<std::size_t>(2 * ghost_width) <= max_stencil_size_PE); // otherwise we don't have the implementation for such a
                                                                                  // large stencil size in polynomial extrapolation

        // Step 1: Fill the diagonal ghost cells layer by layer using stencil sizes 2, 4, ..., 2*ghost_width
        for (int ghost_layer = 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            int stencil_s = 2 * ghost_layer;
            static_for<2, max_stencil_size_PE + 1>::apply(
                [&](auto stencil_size_)
                {
                    static constexpr int stencil_size = static_cast<int>(stencil_size_());
                    if constexpr (stencil_size % 2 == 0)
                    {
                        if (stencil_s == stencil_size)
                        {
                            PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);
                            auto corner = self(corner_lca).on(level);
                            apply_extrapolation_bc_cells<stencil_size>(bc, level, field, direction, corner);
                        }
                    }
                });
        }

        // Step 2: Fill off-diagonal ghost cells by copying the diagonal ghost value.
        //
        // For layer k (k=1..ghost_width), the source is the diagonal cell at
        //   source_at_k = corner + k*direction   (already filled by Step 1).
        // All other cells in the corner block with first-dim offset k are targets.
        // A target's offset from source_at_k is:
        //   delta = sum_{p=1}^{num_nonzero-1} (g_p - (k-1)) * e_dirs[p],
        // where g_p in {0,...,ghost_width-1} and not all g_p == k-1.
        //
        // Example (direction=(-1,-1,-1), ghost_width=2):
        //   k=1: source=(-1,-1,-1). Fill: (-1,-2,-1), (-1,-1,-2), (-1,-2,-2).
        //   k=2: source=(-2,-2,-2). Fill: (-2,-1,-1), (-2,-1,-2), (-2,-2,-1).

        // Collect the non-zero direction dimensions in order.
        std::size_t num_nonzero = 0;
        std::array<std::size_t, Field::dim> nonzero_dirs;
        for (std::size_t d = 0; d < Field::dim; ++d)
        {
            if (direction[d] != 0)
            {
                nonzero_dirs[num_nonzero] = d;
                ++num_nonzero;
            }
        }

        if (num_nonzero < 2)
        {
            return; // No off-diagonal ghosts for Cartesian directions
        }

        auto corner_at_level = self(corner_lca).on(level);

        // Build unit direction vectors for each non-zero dimension.
        std::array<DirectionVector<Field::dim>, Field::dim> e_dirs;
        for (std::size_t idx = 0; idx < num_nonzero; ++idx)
        {
            e_dirs[idx].fill(0);
            e_dirs[idx][nonzero_dirs[idx]] = direction[nonzero_dirs[idx]];
        }

        // Total number of offset combos for the non-first dimensions: ghost_width^(num_nonzero-1).
        // For each layer k, enumerate all (g_1,...,g_{n-1}) in {0,...,ghost_width-1}^{n-1}.
        // The target cell offset from source_at_k is:
        //   delta = sum_{p=1}^{n-1} (g_p - (k-1)) * e_dirs[p].
        // Skip when all g_p == k-1 (that is the source diagonal cell itself).
        std::size_t num_combos = 1;
        for (std::size_t p = 1; p < num_nonzero; ++p)
        {
            num_combos *= static_cast<std::size_t>(ghost_width);
        }

        // Restrict the corner to the cells that actually exist at this level. This is the same
        // condition as in Step 1, where apply_extrapolation_bc_cells() intersects the corner
        // with mesh[cells][level] before applying the extrapolation.
        //
        // Why this restriction is necessary: on an adapted mesh, the domain corner is not
        // necessarily covered by cells at every level. For instance, in the lid-driven cavity
        // with min_level=3 and max_level=6, the velocity singularities refine the corners down
        // to level 6, so at levels 3 to 5 the corner region holds no cells. At such a level:
        //   - Step 1 did nothing (its subsets are empty after intersection with the cells), so
        //     the diagonal ghosts hold no valid source value to copy from;
        //   - worse, the corner ghost cells may not even exist in the mesh at this level (ghost
        //     cells are only allocated around existing cells). Iterating over them would then
        //     query intervals that are not in the cell array, which is an out-of-bounds access
        //     (get_interval() returns m_cells[0][size_t(-1)]) and a crash in practice.
        //
        // The corner ghosts at the levels where the corner has no cells are not filled here:
        // they are filled by projecting the value from the finer levels, level by level from
        // fine to coarse (see project_corner_below() in algorithm/update.hpp, called right
        // after this function in update_outer_ghosts()).
        using mesh_id_t   = typename Field::mesh_t::mesh_id_t;
        auto corner_cells = intersection(corner_at_level, field.mesh()[mesh_id_t::cells][level]).on(level);

        for (int k = 1; k <= ghost_width; ++k)
        {
            auto source_at_k = translate(corner_cells, k * direction);

            for (std::size_t combo = 0; combo < num_combos; ++combo)
            {
                DirectionVector<Field::dim> delta;
                delta.fill(0);
                bool is_source  = true;
                std::size_t tmp = combo;
                for (std::size_t p = 1; p < num_nonzero; ++p)
                {
                    int g_p = static_cast<int>(tmp % static_cast<std::size_t>(ghost_width));
                    tmp /= static_cast<std::size_t>(ghost_width);
                    int d_p = g_p - (k - 1);
                    delta += d_p * e_dirs[p];
                    if (d_p != 0)
                    {
                        is_source = false;
                    }
                }

                if (is_source)
                {
                    continue;
                }

                Stencil<2, Field::dim> stencil_copy;
                xt::view(stencil_copy, 0) = 0;
                xt::view(stencil_copy, 1) = delta;
                auto analyzer_copy        = make_stencil_analyzer(stencil_copy);

                for_each_stencil(field.mesh(),
                                 source_at_k,
                                 analyzer_copy,
                                 [&](const auto& cells)
                                 {
                                     field[cells[1]] = field[cells[0]];
                                 });
            }
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

        auto domain = self(field.mesh().domain()).on(level);

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
        int ghost_width                                              = field.mesh().ghost_width();
        static constexpr std::size_t max_stencil_size_implemented_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;

        // 1. We fill the ghosts that are further than those filled by the B.C. (where there are boundary cells)

        int ghost_layers_filled_by_bc = 0;
        for (auto& bc : field.get_bc())
        {
            ghost_layers_filled_by_bc = std::max(ghost_layers_filled_by_bc, bc->stencil_size() / 2);
        }

        // We populate the ghosts sequentially from the closest to the farthest.
        for (int ghost_layer = ghost_layers_filled_by_bc + 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            int stencil_s = 2 * ghost_layer;
            static_for<2, max_stencil_size_implemented_PE + 1>::apply(
                [&](auto stencil_size_)
                {
                    static constexpr int stencil_size = static_cast<int>(stencil_size_());

                    if constexpr (stencil_size % 2 == 0) // (because PolynomialExtrapolation is only implemented for even stencil_size)
                    {
                        if (stencil_s == stencil_size)
                        {
                            auto& domain = detail::get_mesh(field.mesh());
                            PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);

                            auto boundary_cells = domain_boundary(field.mesh(), level, direction);
                            apply_extrapolation_bc_cells<stencil_size>(bc, level, field, direction, boundary_cells);
                        }
                    }
                });
        }

        // 2. We fill the ghosts that are further than those filled by the projection of the B.C. (where there are ghost cells below
        // boundary cells)

        const std::size_t ghost_layers_filled_by_projection_bc = 1;

        for (int ghost_layer = ghost_layers_filled_by_projection_bc + 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            int stencil_s = 2 * ghost_layer;
            static_for<2, max_stencil_size_implemented_PE + 1>::apply(
                [&](auto stencil_size_)
                {
                    static constexpr int stencil_size = static_cast<int>(stencil_size_());

                    if constexpr (stencil_size % 2 == 0) // (because PolynomialExtrapolation is only implemented for even stencil_size)
                    {
                        if (stencil_s == stencil_size)
                        {
                            auto& domain = detail::get_mesh(field.mesh());
                            PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);

                            auto domain2         = self(field.mesh().domain()).on(level);
                            auto boundary_ghosts = difference(domain2, translate(domain2, -direction));
                            apply_extrapolation_bc_ghosts<stencil_size>(bc, level, field, direction, boundary_ghosts);
                        }
                    }
                });
        }
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
