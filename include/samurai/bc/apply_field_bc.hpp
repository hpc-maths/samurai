// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../boundary.hpp"
#include "../field/concepts.hpp"
#include "../static_dispatch.hpp"
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

    /**
     * Apply polynomial extrapolation on the outside ghosts close to boundary cells
     * @param bc The PolynomialExtrapolation boundary condition holding the coefficients
     * @param level Level where to apply the polynomial extrapolation
     * @param field Field to apply the extrapolation on
     * @param direction Direction of the boundary
     * @param bdry_cells subset corresponding to boundary cells where to apply the extrapolation on (center of the BC stencil)
     * @param stencil_in_x The stencil along the first axis, ending on the ghost to fill
     * @param ghost_layer The outer ghost layer that stencil ends on, counted from the boundary
     */
    template <std::size_t stencil_size, class Field, class Subset>
    void apply_extrapolation_bc_cells(Bc<Field>& bc,
                                      std::size_t level,
                                      Field& field,
                                      const DirectionVector<Field::dim>& direction,
                                      Subset& bdry_cells,
                                      const Stencil<stencil_size, Field::dim>& stencil_in_x,
                                      int ghost_layer)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        auto stencil          = convert_for_direction(stencil_in_x, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        //  We need to check that the furthest ghost exists. It's not always the case for large stencils!
        if (ghost_layer == 1)
        {
            auto cells = intersection(mesh[mesh_id_t::cells][level], bdry_cells).on(level);

            apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
        }
        else
        {
            auto translated_outer_nghbr = translate(mesh[mesh_id_t::reference][level], -ghost_layer * direction); // can be removed?
            auto cells                  = intersection(translated_outer_nghbr, mesh[mesh_id_t::cells][level], bdry_cells).on(level);

            apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
        }
    }

    template <class Mesh, std::size_t stencil_size, std::size_t... Is>
    auto stencil_origins_in_reference_impl(const Mesh& mesh,
                                           std::size_t level,
                                           const Stencil<stencil_size, Mesh::dim>& stencil,
                                           std::index_sequence<Is...>)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        // One copy of the reference cells per stencil cell, translated back by its offset.
        return intersection(translate(mesh[mesh_id_t::reference][level], DirectionVector<Mesh::dim>(-xt::view(stencil, Is)))...);
    }

    /**
     * The cells of @a level on which @a stencil can be laid: every cell it reaches, origin included, is in the reference mesh.
     *
     * Checking the outer ghosts alone is not enough. A coarse ghost that covers finer cells can be in the reference mesh
     * without its inner neighbour along the stencil. This happens at a re-entrant corner of a domain with a hole: the
     * ghost is only there to compute the details of the finer cells on the other side of the corner.
     */
    template <class Mesh, std::size_t stencil_size>
    auto stencil_origins_in_reference(const Mesh& mesh, std::size_t level, const Stencil<stencil_size, Mesh::dim>& stencil)
    {
        static_assert(stencil_size >= 2, "intersection requires at least two sets");

        return stencil_origins_in_reference_impl(mesh, level, stencil, std::make_index_sequence<stencil_size>{});
    }

    template <class Mesh, std::size_t stencil_size, std::size_t... Is>
    auto stencil_origins_with_known_cells_impl(const Mesh& mesh,
                                               std::size_t level,
                                               const Stencil<stencil_size, Mesh::dim>& stencil,
                                               std::index_sequence<Is...>)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        // Known values: the cells, and the projections of the finer cells.
        auto known = union_(mesh[mesh_id_t::cells][level], mesh.get_union()[level]);
        if constexpr (sizeof...(Is) == 1)
        {
            return translate(known, DirectionVector<Mesh::dim>(-xt::view(stencil, 0)));
        }
        else
        {
            return intersection(translate(known, DirectionVector<Mesh::dim>(-xt::view(stencil, Is)))...);
        }
    }

    /**
     * The cells of @a level on which the first @a n_known cells of @a stencil hold known values: cells of @a level, or
     * projections of finer cells. The other ghosts of the level are filled by prediction after the outer ghosts.
     */
    template <std::size_t n_known, class Mesh, std::size_t stencil_size>
    auto stencil_origins_with_known_cells(const Mesh& mesh, std::size_t level, const Stencil<stencil_size, Mesh::dim>& stencil)
    {
        static_assert(n_known >= 1 && n_known <= stencil_size);

        return stencil_origins_with_known_cells_impl(mesh, level, stencil, std::make_index_sequence<n_known>{});
    }

    /**
     * Apply polynomial extrapolation on the outside ghosts close to inner ghosts at the boundary
     * (i.e. inner ghosts in the boundary region that have neighbouring ghosts outside the domain)
     * @tparam ghost_layer Outer ghost layer to fill, counted from the boundary
     * @param bc The PolynomialExtrapolation boundary condition holding the coefficients
     * @param level Level where to apply the polynomial extrapolation
     * @param field Field to apply the extrapolation on
     * @param direction Direction of the boundary
     * @param inner_ghosts_location subset corresponding to inner ghosts where to apply the extrapolation on (center of the BC stencil)
     * @param stencil_in_x The stencil along the first axis, ending on the ghost to fill
     */
    template <std::size_t ghost_layer, std::size_t stencil_size, class Field, class Subset>
    void apply_extrapolation_bc_ghosts(Bc<Field>& bc,
                                       std::size_t level,
                                       Field& field,
                                       const DirectionVector<Field::dim>& direction,
                                       Subset& inner_ghosts_location,
                                       const Stencil<stencil_size, Field::dim>& stencil_in_x)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static_assert(ghost_layer < stencil_size, "the stencil must still reach the boundary cell");

        auto& mesh = field.mesh();

        auto stencil          = convert_for_direction(stencil_in_x, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        auto stencil_origins = stencil_origins_in_reference(mesh, level, stencil);
        // The stencil reads the stencil_size - ghost_layer cells inside the domain. Inside the domain, a ghost that is not
        // the projection of finer cells is predicted after the outer ghosts: it does not hold its value yet.
        auto inner_cells_known                = stencil_origins_with_known_cells<stencil_size - ghost_layer>(mesh, level, stencil);
        auto potential_inner_cells_and_ghosts = intersection(stencil_origins, inner_cells_known, inner_ghosts_location).on(level);
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

    /**
     * Fill the diagonal ghost of @a ghost_layer in the corner @a direction with the value of the
     * inner diagonal cell it mirrors, the one at the offset 1 - ghost_layer from the corner cell.
     *
     * This is what the growing extrapolation stencil of that layer computes (see Step 1 of
     * @c update_outer_corners_by_polynomial_extrapolation), written as the copy it reduces to,
     * so that the layers beyond the largest implemented stencil size need no new coefficients.
     */
    template <class Field>
    void mirror_corner_ghost_layer(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field, int ghost_layer)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh             = field.mesh();
        const auto& corner_lca = mesh.corner(direction);

        // From the cell being mirrored, the ghost it fills lies 2 * ghost_layer - 1 cells further out.
        DirectionVector<Field::dim> to_ghost = (2 * ghost_layer - 1) * direction;

        // Cells to read from: the inner diagonal cell at the offset 1 - ghost_layer from the corner
        // cell. It has to be a cell of this level, and the ghost it mirrors on to has to exist in
        // the mesh. Same two conditions as the extrapolation applies in Step 1.
        auto corner_at_level = self(corner_lca).on(level);
        auto mirrored_cells  = intersection(translate(corner_at_level, (1 - ghost_layer) * direction),
                                           mesh[mesh_id_t::cells][level],
                                           translate(mesh[mesh_id_t::reference][level], -to_ghost))
                                  .on(level);

        Stencil<2, Field::dim> stencil_copy;
        xt::view(stencil_copy, 0) = 0;
        xt::view(stencil_copy, 1) = to_ghost;
        auto analyzer_copy        = make_stencil_analyzer(stencil_copy);

        for_each_stencil(mesh,
                         mirrored_cells,
                         analyzer_copy,
                         [&](const auto& cells)
                         {
                             field[cells[1]] = field[cells[0]];
                         });
    }

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        if constexpr (Field::dim == 1)
        {
            return; // No outer corners in 1D
        }

        static constexpr std::size_t dim = Field::dim;

        // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly on the
        // ghost layer (stencil_size = 2 * ghost_layer) instead of on the stencil size, to avoid instantiating
        // the unused odd-stencil_size candidates.
        static constexpr std::size_t max_stencil_size_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;
        static constexpr std::size_t max_ghost_layers_PE = max_stencil_size_PE / 2;

        int ghost_width        = field.mesh().ghost_width();
        const auto& domain     = detail::get_mesh(field.mesh());
        const auto& corner_lca = field.mesh().corner(direction);

        // Step 1: Fill the diagonal ghost cells layer by layer.
        //
        // Along the diagonal, the stencil of the layer k reads the k inner cells and the k - 1
        // diagonal ghosts that the shallower layers have already filled, and the extrapolation
        // collapses: it writes the value of the inner cell at the offset 1 - k. The corner block
        // is therefore the mirror image of the inner diagonal about the corner. The growing line
        // stencil spells that out as long as its size is implemented; past that size the mirror
        // is applied directly, which is the very same value and needs no new coefficients.
        for (int ghost_layer = 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            auto corner = self(corner_lca).on(level);

            if (static_cast<std::size_t>(2 * ghost_layer) <= max_stencil_size_PE)
            {
                dispatch_static<1, max_ghost_layers_PE>(
                    static_cast<std::size_t>(ghost_layer),
                    [&](auto ghost_layer_)
                    {
                        static constexpr std::size_t layer        = decltype(ghost_layer_)::value;
                        static constexpr std::size_t stencil_size = 2 * layer;

                        PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);
                        apply_extrapolation_bc_cells(bc, level, field, direction, corner, line_stencil<dim, 0, stencil_size>(), ghost_layer);
                    });
            }
            else
            {
                mirror_corner_ghost_layer(level, direction, field, ghost_layer);
            }
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
        // The corner ghosts at the levels where the corner has no cells are not filled here: the
        // diagonal ghost of the first layer is projected from the finer levels, level by level from
        // fine to coarse (see project_corner_below() in algorithm/update_outer_ghost.hpp), and
        // update_refined_outer_corner() fills the rest of the corner block.
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

    /**
     * Fill the corner block @a direction at a level where the corner cell is refined: it is not a
     * cell of @a level, but its children are cells of the finer levels, so that its value is known
     * by projection.
     *
     * Where the corner is a cell, the corner block is the mirror image of the inner diagonal: the
     * ghost whose offset from the corner cell is k along the first direction gets the value of the
     * inner diagonal cell at the offset 1 - k (see @c update_outer_corners_by_polynomial_extrapolation).
     * The same values are written here, read on the inner cells that are cells or projections of
     * finer cells, except for the first layer: it copies the diagonal ghost of the first layer, which
     * comes from the finer levels (see project_corner_below()). Each ghost is read from its source
     * directly, so that the block may lack its diagonal ghosts in the reference mesh.
     *
     * With a prediction of radius 1, the details only read the diagonal ghost of the first layer.
     * A larger radius reads the rest of the corner block.
     */
    template <class Field>
    void update_refined_outer_corner(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t dim = Field::dim;

        if constexpr (dim > 1)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;

            auto& mesh            = field.mesh();
            const int ghost_width = mesh.ghost_width();

            const auto& cells     = mesh[mesh_id_t::cells][level];
            const auto& reference = mesh[mesh_id_t::reference][level];
            const auto& refined   = mesh.get_union()[level];

            auto refined_corner = difference(intersection(self(mesh.corner(direction)).on(level), refined, reference), cells).on(level);
            // Inner cells whose value is known at this point: the cells, and the projections of the finer cells.
            auto known_cells = intersection(union_(cells, refined), reference);

            // The non-zero directions of the corner, the first one sets the layer.
            std::size_t num_nonzero = 0;
            std::array<std::size_t, dim> nonzero_dirs;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (direction[d] != 0)
                {
                    nonzero_dirs[num_nonzero++] = d;
                }
            }

            // Every ghost of the block: an offset in {1, ..., ghost_width} along each non-zero direction.
            std::size_t num_ghosts = 1;
            for (std::size_t p = 0; p < num_nonzero; ++p)
            {
                num_ghosts *= static_cast<std::size_t>(ghost_width);
            }

            for (std::size_t g = 0; g < num_ghosts; ++g)
            {
                DirectionVector<dim> to_ghost; // from the corner cell
                to_ghost.fill(0);
                int layer = 0;
                for (std::size_t p = 0, rest = g; p < num_nonzero; ++p, rest /= static_cast<std::size_t>(ghost_width))
                {
                    const int offset          = static_cast<int>(rest % static_cast<std::size_t>(ghost_width)) + 1;
                    to_ghost[nonzero_dirs[p]] = offset * direction[nonzero_dirs[p]];
                    if (p == 0)
                    {
                        layer = offset;
                    }
                }

                Stencil<2, dim> stencil_copy;
                xt::view(stencil_copy, 0) = 0;
                auto copy_to_ghosts       = [&](const auto& sources)
                {
                    auto analyzer_copy = make_stencil_analyzer(stencil_copy);
                    for_each_stencil(mesh,
                                     sources,
                                     analyzer_copy,
                                     [&](const auto& stencil_cells)
                                     {
                                         field[stencil_cells[1]] = field[stencil_cells[0]];
                                     });
                };

                // The source: for the first layer, the diagonal ghost filled from the finer levels; beyond, the
                // inner diagonal cell it mirrors.
                const bool first_layer               = layer == 1;
                const DirectionVector<dim> to_source = first_layer ? DirectionVector<dim>(direction)
                                                                   : DirectionVector<dim>((1 - layer) * direction);
                const DirectionVector<dim> delta     = to_ghost - to_source;
                if (xt::all(xt::equal(delta, 0)))
                {
                    continue;
                }
                xt::view(stencil_copy, 1)              = delta;
                const DirectionVector<dim> minus_delta = -delta;
                if (first_layer)
                {
                    copy_to_ghosts(intersection(translate(refined_corner, to_source), reference, translate(reference, minus_delta)).on(level));
                }
                else
                {
                    copy_to_ghosts(
                        intersection(translate(refined_corner, to_source), known_cells, translate(reference, minus_delta)).on(level));
                }
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

    namespace detail
    {
        /**
         * Fills the ghost layers [first_layer, ghost_width] by polynomial extrapolation, from the closest layer to the farthest.
         * For each layer, calls fill_layer(bc, stencil, layer), where layer is a std::integral_constant and stencil is the
         * line stencil along the direction that ends on the ghost of that layer.
         */
        template <class Field, class Func>
        void for_each_extrapolated_ghost_layer(Field& field, int first_layer, int ghost_width, Func&& fill_layer)
        {
            static constexpr std::size_t dim = Field::dim;

            // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly on the
            // ghost layer instead of on the stencil size, to avoid instantiating the unused odd-stencil_size
            // candidates.
            static constexpr std::size_t max_stencil_size_implemented_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;
            static constexpr std::size_t max_ghost_layers_implemented_PE = PolynomialExtrapolation<Field, 2>::max_ghost_layers_implemented_PE;

            for (int ghost_layer = first_layer; ghost_layer <= ghost_width; ++ghost_layer)
            {
                dispatch_static<1, max_ghost_layers_implemented_PE>(
                    static_cast<std::size_t>(ghost_layer),
                    [&](auto ghost_layer_)
                    {
                        static constexpr std::size_t layer = decltype(ghost_layer_)::value;
                        // The stencil ends on the ghost of that layer. It grows with the layer until it
                        // reaches the largest implemented size, then slides outward at constant size.
                        static constexpr std::size_t stencil_size = std::min(2 * layer, max_stencil_size_implemented_PE);
                        static constexpr int stencil_start        = static_cast<int>(layer) - static_cast<int>(stencil_size) + 1;

                        auto& domain = get_mesh(field.mesh());
                        PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);

                        fill_layer(bc, line_stencil_from<dim, 0, stencil_size>(stencil_start), ghost_layer_);
                    });
            }
        }
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t max_ghost_layers_implemented_PE = PolynomialExtrapolation<Field, 2>::max_ghost_layers_implemented_PE;

        int ghost_width = field.mesh().ghost_width();

        detail::check_ghost_width_for_polynomial_extrapolation(ghost_width, max_ghost_layers_implemented_PE);

        // 1. We fill the ghosts that are further than those filled by the B.C. (where there are boundary cells)

        int ghost_layers_filled_by_bc = 0;
        for (auto& bc : field.get_bc())
        {
            ghost_layers_filled_by_bc = std::max(ghost_layers_filled_by_bc, bc->stencil_size() / 2);
        }

        detail::for_each_extrapolated_ghost_layer(
            field,
            ghost_layers_filled_by_bc + 1,
            ghost_width,
            [&](auto& bc, const auto& stencil, auto ghost_layer_)
            {
                static constexpr std::size_t layer = decltype(ghost_layer_)::value;

                auto boundary_cells = domain_boundary(field.mesh(), level, direction);
                apply_extrapolation_bc_cells(bc, level, field, direction, boundary_cells, stencil, static_cast<int>(layer));
            });

        // 2. We fill the ghosts that are further than those filled by the projection of the B.C. (where there are ghost cells below
        // boundary cells)

        const int ghost_layers_filled_by_projection_bc = 1;

        detail::for_each_extrapolated_ghost_layer(
            field,
            ghost_layers_filled_by_projection_bc + 1,
            ghost_width,
            [&](auto& bc, const auto& stencil, auto ghost_layer_)
            {
                static constexpr std::size_t layer = decltype(ghost_layer_)::value;

                const auto& domain2  = field.mesh().domain(level);
                auto boundary_ghosts = difference(domain2, translate(domain2, -direction));
                apply_extrapolation_bc_ghosts<layer>(bc, level, field, direction, boundary_ghosts, stencil);
            });
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
