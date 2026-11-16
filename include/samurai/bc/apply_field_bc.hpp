// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm/cell_block.hpp"
#include "../boundary.hpp"
#include "../entry_table.hpp"
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
        /**
         * For each ghost layer from @a first_layer to @a last_layer, calls f(stencil_size, layer), where stencil_size is a
         * std::integral_constant: the size of the line stencil that ends on the ghost of that layer.
         */
        template <class Func>
        void for_each_extrapolated_ghost_layer(int first_layer, int last_layer, Func&& f)
        {
            // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly on the
            // ghost layer instead of on the stencil size, to avoid instantiating the unused odd-stencil_size
            // candidates.
            for (int ghost_layer = first_layer; ghost_layer <= last_layer; ++ghost_layer)
            {
                dispatch_static<1, max_ghost_layers_implemented_PE>(
                    static_cast<std::size_t>(ghost_layer),
                    [&](auto ghost_layer_)
                    {
                        static constexpr std::size_t layer = decltype(ghost_layer_)::value;
                        // The stencil ends on the ghost of that layer. It grows with the layer until it
                        // reaches the largest implemented size, then slides outward at constant size.
                        static constexpr std::size_t stencil_size = std::min(2 * layer, max_stencil_size_implemented_PE);

                        f(std::integral_constant<std::size_t, stencil_size>{}, layer);
                    });
            }
        }

        // The line stencil of `stencil_size` cells along `direction` that ends on the ghost of the layer `layer`.
        template <std::size_t stencil_size, std::size_t dim>
        auto extrapolation_stencil(std::size_t layer, const DirectionVector<dim>& direction)
        {
            const int stencil_start = static_cast<int>(layer) - static_cast<int>(stencil_size) + 1;
            return convert_for_direction(line_stencil_from<dim, 0, stencil_size>(stencil_start), direction);
        }

        // Emits the storage indices of the cells of `stencil` laid on each cell of `origins`, the ghost to fill last.
        template <class Mesh, class Subset, std::size_t stencil_size, class Emit>
        void emit_stencils(const Mesh& mesh, Subset& origins, const Stencil<stencil_size, Mesh::dim>& stencil, Emit& emit)
        {
            auto stencil_analyzer = make_stencil_analyzer(stencil);
            for_each_stencil(mesh,
                             origins,
                             stencil_analyzer,
                             [&](const auto& cells)
                             {
                                 Cells<stencil_size> indices;
                                 for (std::size_t k = 0; k < stencil_size; ++k)
                                 {
                                     indices[k] = static_cast<std::size_t>(cells[k].index);
                                 }
                                 emit(indices);
                             });
        }

        // Emits the stencils laid on the cells of `level` in `bdry_cells`, for the ghost layer `layer`. Past
        // the first layer, the ghost a stencil ends on has to exist: it is not always the case for large stencils.
        template <class Mesh, class Subset, std::size_t stencil_size, class Emit>
        void emit_stencils_on_cells(const Mesh& mesh,
                                    std::size_t level,
                                    const DirectionVector<Mesh::dim>& direction,
                                    Subset& bdry_cells,
                                    const Stencil<stencil_size, Mesh::dim>& stencil,
                                    std::size_t layer,
                                    Emit& emit)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            if (layer == 1)
            {
                auto cells = intersection(mesh[mesh_id_t::cells][level], bdry_cells).on(level);
                emit_stencils(mesh, cells, stencil, emit);
            }
            else
            {
                auto translated_outer_nghbr = translate(mesh[mesh_id_t::reference][level], -static_cast<int>(layer) * direction);
                auto cells                  = intersection(translated_outer_nghbr, mesh[mesh_id_t::cells][level], bdry_cells).on(level);
                emit_stencils(mesh, cells, stencil, emit);
            }
        }

        //////////////////////////////////////////////////////////////////////
        // Direct walks
        //////////////////////////////////////////////////////////////////////

        // What a set expression checks before laying a stencil.
        enum class CheckedCells
        {
            all,   // every stencil cell is in the reference mesh
            ghost, // the ghost to fill, the last cell, is in the reference mesh
            none,
        };

        template <std::size_t size, std::size_t dim>
        std::array<DirectionVector<dim>, size> stencil_offsets(const Stencil<size, dim>& stencil)
        {
            std::array<DirectionVector<dim>, size> offsets;
            for (std::size_t k = 0; k < size; ++k)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    offsets[k][d] = stencil(k, d);
                }
            }
            return offsets;
        }

        // Emits the stencils laid on the cells [start, end) of the row of `coords`, their cells read in `ref`.
        // Returns false when a stencil cell is missing where the set expression does not check it.
        template <class LCA, std::size_t size, class Emit>
        bool emit_row_stencils(const LCA& ref,
                               const std::array<typename LCA::value_t, LCA::dim>& coords,
                               typename LCA::value_t start,
                               typename LCA::value_t end,
                               const std::array<DirectionVector<LCA::dim>, size>& stencil,
                               CheckedCells checked,
                               Emit& emit)
        {
            auto rows_of_stencil = [&]<std::size_t... k>(std::index_sequence<k...>)
            {
                return std::array<RowLookup<LCA>, size>{RowLookup<LCA>(ref, translate(coords, stencil[k]))...};
            }(std::make_index_sequence<size>{});

            Cells<size> indices;
            for (auto x = start; x < end; ++x)
            {
                bool missing_ghost = false;
                bool missing_other = false;
                for (std::size_t k = 0; k < size; ++k)
                {
                    const auto index = rows_of_stencil[k](x + stencil[k][0]);
                    if (index < 0)
                    {
                        (k == size - 1 ? missing_ghost : missing_other) = true;
                    }
                    else
                    {
                        indices[k] = static_cast<std::size_t>(index);
                    }
                }
                if ((checked == CheckedCells::all && (missing_ghost || missing_other)) || (checked == CheckedCells::ghost && missing_ghost))
                {
                    continue;
                }
                if (missing_ghost || missing_other)
                {
                    return false;
                }
                emit(indices);
            }
            return true;
        }

        // Emits the stencils laid on the cells of `level` in `block`.
        template <class Mesh, std::size_t size, class Emit>
        bool emit_block_stencils(const Mesh& mesh,
                                 std::size_t level,
                                 const CellBlock<typename Mesh::interval_t::value_t, Mesh::dim>& block,
                                 const std::array<DirectionVector<Mesh::dim>, size>& stencil,
                                 CheckedCells checked,
                                 Emit& emit)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            bool done       = true;
            for_each_interval_in_block(
                mesh[mesh_id_t::cells][level],
                block,
                [&](auto start, auto end, const auto& coords, auto)
                {
                    done = done && emit_row_stencils(mesh[mesh_id_t::reference][level], coords, start, end, stencil, checked, emit);
                });
            return done;
        }

        //////////////////////////////////////////////////////////////////////
        // Keys
        //////////////////////////////////////////////////////////////////////

        // The ghosts of the layer `layer` outside the domain in `direction`, at the boundary cells of `level`:
        // the stencils that extrapolate them, laid on these cells.
        template <std::size_t dim, std::size_t stencil_size>
        struct extrapolated_ghosts_at_cells
        {
            using entry_type = Cells<stencil_size>;

            std::size_t level;
            std::size_t direction;
            std::size_t layer;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                const auto dir      = direction_from_code<dim>(direction);
                auto boundary_cells = domain_boundary(mesh, level, dir);
                emit_stencils_on_cells(mesh, level, dir, boundary_cells, extrapolation_stencil<stencil_size>(layer, dir), layer, emit);
            }

            // On a box domain that holds the cells, the boundary cells are those on its face.
            template <class Mesh, class Emit>
            bool walk(const Mesh& mesh, Emit& emit) const
            {
                const auto domain = box_domain_block(mesh, level, false);
                if (!domain)
                {
                    return false;
                }
                const auto dir = direction_from_code<dim>(direction);
                return emit_block_stencils(mesh,
                                           level,
                                           face_layer(*domain, dir, 0),
                                           stencil_offsets(extrapolation_stencil<stencil_size>(layer, dir)),
                                           layer == 1 ? CheckedCells::none : CheckedCells::ghost,
                                           emit);
            }
        };

        // The same layer where the boundary of `level` holds ghosts below finer cells and not cells: the
        // stencils laid on the inner ghosts whose whole stencil is in the reference mesh.
        template <std::size_t dim, std::size_t stencil_size>
        struct extrapolated_ghosts_at_inner_ghosts
        {
            using entry_type = Cells<stencil_size>;

            std::size_t level;
            std::size_t direction;
            std::size_t layer;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                const auto dir       = direction_from_code<dim>(direction);
                const auto stencil   = extrapolation_stencil<stencil_size>(layer, dir);
                const auto& domain   = mesh.domain(level);
                auto boundary_ghosts = difference(domain, translate(domain, -dir));

                auto stencil_origins                  = stencil_origins_in_reference(mesh, level, stencil);
                auto potential_inner_cells_and_ghosts = intersection(stencil_origins, boundary_ghosts).on(level);
                auto inner_cells_and_ghosts           = intersection(potential_inner_cells_and_ghosts, mesh.get_union()[level]).on(level);
                auto inner_ghosts_with_outer_nghbr    = difference(inner_cells_and_ghosts, mesh[mesh_id_t::cells][level]).on(level);
                emit_stencils(mesh, inner_ghosts_with_outer_nghbr, stencil, emit);
            }

            // On a box domain that holds the cells and the union: the union cells on its face that are not cells.
            template <class Mesh, class Emit>
            bool walk(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                const auto domain = box_domain_block(mesh, level, true);
                if (!domain)
                {
                    return false;
                }
                const auto dir     = direction_from_code<dim>(direction);
                const auto stencil = stencil_offsets(extrapolation_stencil<stencil_size>(layer, dir));
                const auto& ref    = mesh[mesh_id_t::reference][level];
                for_each_interval_in_block(mesh.get_union()[level],
                                           face_layer(*domain, dir, 0),
                                           [&](auto start, auto end, const auto& coords, auto)
                                           {
                                               const auto row = coords;
                                               auto x         = start;
                                               auto gap       = [&](auto gap_end)
                                               {
                                                   if (x < gap_end)
                                                   {
                                                       emit_row_stencils(ref, row, x, gap_end, stencil, CheckedCells::all, emit);
                                                   }
                                               };
                                               for_each_interval_in_block(mesh[mesh_id_t::cells][level],
                                                                          row_block(row, start, end),
                                                                          [&](auto cells_start, auto cells_end, const auto&, auto)
                                                                          {
                                                                              gap(cells_start);
                                                                              x = cells_end;
                                                                          });
                                               gap(end);
                                           });
                return true;
            }
        };

        // The diagonal ghost of the layer `layer` in the corner of the diagonal `direction`, at `level`.
        //
        // Along the diagonal, the stencil of the layer k reads the k inner cells and the k - 1 diagonal ghosts
        // that the shallower layers have already filled, and the extrapolation collapses: it writes the value
        // of the inner cell at the offset 1 - k. The growing line stencil spells that out as long as its size,
        // 2 * layer, is implemented. Past that size, stencil_size is 2 and the mirror is applied directly: the
        // ghost is copied from the inner diagonal cell 2 * layer - 1 cells further in.
        template <std::size_t dim, std::size_t stencil_size>
        struct corner_diagonal_ghosts
        {
            using entry_type = Cells<stencil_size>;

            std::size_t level;
            std::size_t direction;
            std::size_t layer;

            bool mirrored() const
            {
                return 2 * layer > max_stencil_size_implemented_PE;
            }

            // From the mirrored inner cell to the ghost.
            DirectionVector<dim> to_ghost(const DirectionVector<dim>& dir) const
            {
                return (2 * static_cast<int>(layer) - 1) * dir;
            }

            // From the corner cell to the mirrored inner cell.
            DirectionVector<dim> to_inner_cell(const DirectionVector<dim>& dir) const
            {
                return (1 - static_cast<int>(layer)) * dir;
            }

            // The stencil of the mirror: the inner cell, then the ghost.
            Stencil<2, dim> mirror_stencil(const DirectionVector<dim>& dir) const
            {
                const auto ghost = to_ghost(dir);
                Stencil<2, dim> stencil;
                xt::view(stencil, 0) = 0;
                xt::view(stencil, 1) = ghost;
                return stencil;
            }

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                const auto dir = direction_from_code<dim>(direction);
                auto corner    = self(mesh.corner(dir)).on(level);
                if constexpr (stencil_size == 2)
                {
                    if (mirrored())
                    {
                        // The inner cell has to be a cell of this level, and the ghost has to exist in the mesh:
                        // the same two conditions as the extrapolation.
                        auto mirrored_cells = intersection(translate(corner, to_inner_cell(dir)),
                                                           mesh[mesh_id_t::cells][level],
                                                           translate(mesh[mesh_id_t::reference][level], -to_ghost(dir)))
                                                  .on(level);
                        emit_stencils(mesh, mirrored_cells, mirror_stencil(dir), emit);
                        return;
                    }
                }
                emit_stencils_on_cells(mesh, level, dir, corner, extrapolation_stencil<stencil_size>(layer, dir), layer, emit);
            }

            template <class Mesh, class Emit>
            bool walk(const Mesh& mesh, Emit& emit) const
            {
                const auto dir    = direction_from_code<dim>(direction);
                const auto corner = corner_block(mesh, dir, level);
                if (!corner)
                {
                    return false;
                }
                const auto checked = layer == 1 ? CheckedCells::none : CheckedCells::ghost;
                if constexpr (stencil_size == 2)
                {
                    if (mirrored())
                    {
                        return emit_block_stencils(mesh,
                                                   level,
                                                   translate(*corner, to_inner_cell(dir)),
                                                   stencil_offsets(mirror_stencil(dir)),
                                                   checked,
                                                   emit);
                    }
                }
                return emit_block_stencils(mesh, level, *corner, stencil_offsets(extrapolation_stencil<stencil_size>(layer, dir)), checked, emit);
            }
        };

        // Calls f(k, delta) for each off-diagonal ghost of a corner block: the ghost is at the offset delta
        // from the diagonal ghost of the layer k, the one at k * direction from the corner cell.
        //
        // With n non-zero components in the direction, delta = sum_{p=1}^{n-1} (g_p - (k - 1)) * e_p, where
        // g_p is in {0, ..., ghost_width - 1}, e_p is the direction restricted to its p-th non-zero
        // component, and not all g_p equal k - 1 (that is the diagonal ghost itself).
        //
        // Example (direction=(-1,-1,-1), ghost_width=2):
        //   k=1: source=(-1,-1,-1). Fill: (-1,-2,-1), (-1,-1,-2), (-1,-2,-2).
        //   k=2: source=(-2,-2,-2). Fill: (-2,-1,-1), (-2,-1,-2), (-2,-2,-1).
        template <std::size_t dim, class Func>
        void for_each_off_diagonal_offset(const DirectionVector<dim>& direction, int ghost_width, Func&& f)
        {
            std::size_t num_nonzero = 0;
            std::array<DirectionVector<dim>, dim> e_dirs;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (direction[d] != 0)
                {
                    e_dirs[num_nonzero].fill(0);
                    e_dirs[num_nonzero][d] = direction[d];
                    ++num_nonzero;
                }
            }
            if (num_nonzero < 2)
            {
                return; // No off-diagonal ghosts for Cartesian directions
            }
            std::size_t num_combos = 1;
            for (std::size_t p = 1; p < num_nonzero; ++p)
            {
                num_combos *= static_cast<std::size_t>(ghost_width);
            }
            for (int k = 1; k <= ghost_width; ++k)
            {
                for (std::size_t combo = 0; combo < num_combos; ++combo)
                {
                    DirectionVector<dim> delta;
                    delta.fill(0);
                    bool is_source  = true;
                    std::size_t tmp = combo;
                    for (std::size_t p = 1; p < num_nonzero; ++p)
                    {
                        const int d_p = static_cast<int>(tmp % static_cast<std::size_t>(ghost_width)) - (k - 1);
                        tmp /= static_cast<std::size_t>(ghost_width);
                        delta += d_p * e_dirs[p];
                        is_source = is_source && d_p == 0;
                    }
                    if (!is_source)
                    {
                        f(k, delta);
                    }
                }
            }
        }

        // The off-diagonal ghosts of the corner of the diagonal `direction`, at `level`: each is a copy of the
        // diagonal ghost of its layer, once corner_diagonal_ghosts are filled.
        //
        // Only where the corner holds cells at this level: on an adapted mesh, the corner is not necessarily
        // covered by cells at every level, and then neither the diagonal ghosts nor the ghosts to fill need
        // exist. The corner ghosts of such a level are filled from the finer levels (see project_corner_below).
        template <std::size_t dim>
        struct corner_off_diagonal_ghosts
        {
            using entry_type = Cells<2>;

            std::size_t level;
            std::size_t direction;

            template <class Mesh, class Emit>
            void produce(const Mesh& mesh, Emit& emit) const
            {
                using mesh_id_t = typename Mesh::mesh_id_t;

                const auto dir    = direction_from_code<dim>(direction);
                auto corner_cells = intersection(self(mesh.corner(dir)).on(level), mesh[mesh_id_t::cells][level]).on(level);
                for_each_off_diagonal_offset<dim>(dir,
                                                  mesh.ghost_width(),
                                                  [&](int k, const auto& delta)
                                                  {
                                                      auto source_at_k = translate(corner_cells, k * dir);
                                                      Stencil<2, dim> stencil;
                                                      xt::view(stencil, 0) = 0;
                                                      xt::view(stencil, 1) = delta;
                                                      emit_stencils(mesh, source_at_k, stencil, emit);
                                                  });
            }

            template <class Mesh, class Emit>
            bool walk(const Mesh& mesh, Emit& emit) const
            {
                const auto dir    = direction_from_code<dim>(direction);
                const auto corner = corner_block(mesh, dir, level);
                if (!corner)
                {
                    return false;
                }
                bool done = true;
                for_each_off_diagonal_offset<dim>(dir,
                                                  mesh.ghost_width(),
                                                  [&](int k, const auto& delta)
                                                  {
                                                      std::array<DirectionVector<dim>, 2> stencil;
                                                      stencil[0] = k * dir;
                                                      stencil[1] = stencil[0] + delta;
                                                      done       = done
                                                          && emit_block_stencils(mesh, level, *corner, stencil, CheckedCells::none, emit);
                                                  });
                return done;
            }
        };

        // Same arithmetic as PolynomialExtrapolation, component by component. With two cells, a copy.
        template <class Field, std::size_t stencil_size>
        void extrapolate_cells(Field& field, const Cells<stencil_size>& cells)
        {
            constexpr std::size_t nc = Field::n_comp;
            auto* data               = field.data();
            for (std::size_t c = 0; c < nc; ++c)
            {
                data[cells[stencil_size - 1] * nc + c] = extrapolated_value<stencil_size>(
                    [&](std::size_t k)
                    {
                        return data[cells[k] * nc + c];
                    });
            }
        }

        // The kernel that extrapolates the entries of a key into `field`.
        template <class Field>
        auto extrapolation_of(Field& field)
        {
            return [&field](const auto& cells)
            {
                extrapolate_cells(field, cells);
            };
        }
    }

    // Fills the corner ghosts of `level` in the diagonal `direction`: the diagonal ones layer by layer, by
    // polynomial extrapolation, then the others, copies of the diagonal ghost of their layer.
    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t dim = Field::dim;

        if constexpr (dim > 1)
        {
            const auto& mesh      = field.mesh();
            const auto dir        = direction_code<dim>(direction);
            const int ghost_width = mesh.ghost_width();

            const auto extrapolate = detail::extrapolation_of(field);

            for (int ghost_layer = 1; ghost_layer <= ghost_width; ++ghost_layer)
            {
                const auto layer = static_cast<std::size_t>(ghost_layer);
                if (2 * layer <= detail::max_stencil_size_implemented_PE)
                {
                    // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly
                    // on the ghost layer (stencil_size = 2 * ghost_layer).
                    dispatch_static<1, detail::max_stencil_size_implemented_PE / 2>(
                        layer,
                        [&](auto ghost_layer_)
                        {
                            static constexpr std::size_t stencil_size = 2 * decltype(ghost_layer_)::value;
                            for_each_entry(mesh, detail::corner_diagonal_ghosts<dim, stencil_size>{level, dir, layer}, extrapolate);
                        });
                }
                else
                {
                    for_each_entry(mesh, detail::corner_diagonal_ghosts<dim, 2>{level, dir, layer}, extrapolate);
                }
            }
            for_each_entry(mesh, detail::corner_off_diagonal_ghosts<dim>{level, dir}, extrapolate);
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
        static constexpr std::size_t dim = Field::dim;

        const auto& mesh      = field.mesh();
        const int ghost_width = mesh.ghost_width();
        const auto dir        = direction_code<dim>(direction);

        detail::check_ghost_width_for_polynomial_extrapolation(ghost_width, detail::max_ghost_layers_implemented_PE);

        const auto extrapolate = detail::extrapolation_of(field);

        // 1. We fill the ghosts that are further than those filled by the B.C. (where there are boundary cells)

        int ghost_layers_filled_by_bc = 0;
        for (auto& bc : field.get_bc())
        {
            ghost_layers_filled_by_bc = std::max(ghost_layers_filled_by_bc, bc->stencil_size() / 2);
        }

        detail::for_each_extrapolated_ghost_layer(
            ghost_layers_filled_by_bc + 1,
            ghost_width,
            [&](auto stencil_size_, std::size_t layer)
            {
                static constexpr std::size_t stencil_size = decltype(stencil_size_)::value;
                for_each_entry(mesh, detail::extrapolated_ghosts_at_cells<dim, stencil_size>{level, dir, layer}, extrapolate);
            });

        // 2. We fill the ghosts that are further than those filled by the projection of the B.C. (where there are ghost cells below
        // boundary cells)

        const int ghost_layers_filled_by_projection_bc = 1;

        detail::for_each_extrapolated_ghost_layer(
            ghost_layers_filled_by_projection_bc + 1,
            ghost_width,
            [&](auto stencil_size_, std::size_t layer)
            {
                static constexpr std::size_t stencil_size = decltype(stencil_size_)::value;
                for_each_entry(mesh, detail::extrapolated_ghosts_at_inner_ghosts<dim, stencil_size>{level, dir, layer}, extrapolate);
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
