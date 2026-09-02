// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <utility>

#include "../boundary.hpp"
#include "../field/concepts.hpp"
#include "../static_dispatch.hpp"
#include "polynomial_extrapolation.hpp"
#include <algorithm>

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

    namespace detail
    {
        template <class Mesh, std::size_t stencil_size, std::size_t... Is>
        auto
        cells_holding_stencil(const Mesh& mesh, std::size_t level, const Stencil<stencil_size, Mesh::dim>& stencil, std::index_sequence<Is...>)
        {
            using mesh_id_t       = typename Mesh::mesh_id_t;
            const auto& reference = mesh[mesh_id_t::reference][level];
            auto shifted_back     = [&](std::size_t i)
            {
                DirectionVector<Mesh::dim> shift = -xt::view(stencil, i);
                return translate(reference, shift);
            };
            if constexpr (stencil_size == 1)
            {
                return shifted_back(0);
            }
            else
            {
                return intersection(shifted_back(Is)...);
            }
        }
    }

    /**
     * The cells of @a level around which the mesh holds every cell of @a stencil: a stencil
     * centred on any of them reads and writes cells that exist. A real cell always qualifies,
     * its ghost layers being what the mesh is built to hold; a projection ghost under a refined
     * boundary does not always, its neighbours further inside the domain being held at the finer
     * level only.
     */
    template <class Mesh, std::size_t stencil_size>
    auto cells_holding_stencil(const Mesh& mesh, std::size_t level, const Stencil<stencil_size, Mesh::dim>& stencil)
    {
        return detail::cells_holding_stencil(mesh, level, stencil, std::make_index_sequence<stencil_size>{});
    }

    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
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

                    // The cells of this level in the boundary region around which the whole
                    // stencil exists. Every real cell's does; a projection ghost's under a
                    // refined boundary not always - its outermost ghost, and the cells inside
                    // the domain the condition reads, may be held at the finer level only.
                    auto bdry_cells = intersection(region_lca[d], cells_holding_stencil(mesh, level, stencil)).on(level);
                    apply_bc_on_subset(bc, field, bdry_cells, stencil_analyzer, direction);
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
     * reads a diagonal ghost, and those ghosts keep the polynomial extrapolation of
     * @c update_outer_corners_by_polynomial_extrapolation as a fallback. A lattice-Boltzmann reflection
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
            // the mesh (the corner extrapolation uses it too). Restricted to the cells this level
            // holds with their corner ghost: on an adapted mesh the corner is not covered at
            // every level, and iterating ghosts that do not exist is an out-of-bounds access.
            auto corner_cells = intersection(self(mesh.corner(direction)).on(level), cells_holding_stencil(mesh, level, stencil)).on(level);

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
        const auto& mesh = field.mesh();

        auto stencil_0        = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());
        auto stencil          = convert_for_direction(stencil_0, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        // The cells of bdry_cells around which the mesh holds the whole stencil, which is not
        // always the case for a large stencil, nor for a projection ghost.
        auto cells = intersection(cells_holding_stencil(mesh, level, stencil), bdry_cells).on(level);

        apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
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
            dispatch_static<1, max_stencil_size_implemented_BC>(bc->stencil_size(),
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

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        if constexpr (Field::dim == 1)
        {
            return; // No outer corners in 1D
        }

        static constexpr std::size_t max_stencil_size_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;
        // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly on the
        // ghost layer (stencil_size = 2 * ghost_layer) instead of on the stencil size, to avoid instantiating
        // the unused odd-stencil_size candidates.
        static constexpr std::size_t max_ghost_layers_PE = max_stencil_size_PE / 2;

        int ghost_width        = field.mesh().ghost_width();
        const auto& domain     = detail::get_mesh(field.mesh());
        const auto& corner_lca = field.mesh().corner(direction);

        assert(static_cast<std::size_t>(2 * ghost_width) <= max_stencil_size_PE); // otherwise we don't have the implementation for such a
                                                                                  // large stencil size in polynomial extrapolation

        // Step 1: Fill the diagonal ghost cells layer by layer using stencil sizes 2, 4, ..., 2*ghost_width
        for (int ghost_layer = 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            dispatch_static<1, max_ghost_layers_PE>(static_cast<std::size_t>(ghost_layer),
                                                    [&](auto ghost_layer_)
                                                    {
                                                        static constexpr int stencil_size = 2 * static_cast<int>(ghost_layer_());
                                                        PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);
                                                        auto corner = self(corner_lca).on(level);
                                                        apply_extrapolation_bc_cells<stencil_size>(bc, level, field, direction, corner);
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

        // Restrict the corner to the cells this level holds with their whole corner block, as
        // Step 1 does through apply_extrapolation_bc_cells(). On an adapted mesh the domain
        // corner is not covered by cells at every level, and a coarse level under a refined
        // corner holds it as a projection ghost whose corner block may be narrower than the
        // ghost width; iterating ghosts that do not exist is an out-of-bounds access.
        using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
        const auto& reference = field.mesh()[mesh_id_t::reference][level];
        auto corner_cells     = intersection(corner_at_level, reference, translate(reference, -ghost_width * direction)).on(level);

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
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        int ghost_width                                              = field.mesh().ghost_width();
        static constexpr std::size_t max_stencil_size_implemented_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;
        // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly on the
        // ghost layer (stencil_size = 2 * ghost_layer) instead of on the stencil size, to avoid instantiating
        // the unused odd-stencil_size candidates.
        static constexpr std::size_t max_ghost_layers_implemented_PE = max_stencil_size_implemented_PE / 2;

        // The layers further than those filled by the B.C., around the boundary cells of this level.

        int ghost_layers_filled_by_bc = 0;
        for (auto& bc : field.get_bc())
        {
            ghost_layers_filled_by_bc = std::max(ghost_layers_filled_by_bc, bc->stencil_size() / 2);
        }

        // We populate the ghosts sequentially from the closest to the farthest.
        for (int ghost_layer = ghost_layers_filled_by_bc + 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            dispatch_static<1, max_ghost_layers_implemented_PE>(
                static_cast<std::size_t>(ghost_layer),
                [&](auto ghost_layer_)
                {
                    static constexpr int stencil_size = 2 * static_cast<int>(ghost_layer_());

                    auto& domain = detail::get_mesh(field.mesh());
                    PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);

                    // The cells this level holds inside the domain and touching its boundary.
                    const auto& mesh        = field.mesh();
                    const auto& domain_at_l = mesh.domain(level);
                    auto boundary_cells     = difference(intersection(mesh[mesh_id_t::reference][level], domain_at_l),
                                                     translate(self(domain_at_l), -direction));
                    apply_extrapolation_bc_cells<stencil_size>(bc, level, field, direction, boundary_cells);
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
