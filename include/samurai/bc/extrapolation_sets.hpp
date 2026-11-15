// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

#include "../algorithm/cell_block.hpp"
#include "../boundary.hpp"
#include "../static_dispatch.hpp"
#include "../stencil.hpp"
#include "polynomial_extrapolation_formula.hpp"

// The ghosts outside the domain that update_outer_ghosts fills by polynomial
// extrapolation. They only depend on the mesh, so each traversal reports the
// stencils it applies to a visitor: visitor.stencil(cells), cells being the
// std::array of the stencil cells, the ghost to fill last. The field path
// extrapolates there, the table builder records the cell indices.
namespace samurai
{
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
}

namespace samurai::detail
{
    // Calls f(cells) for each stencil of the layer `ghost_layer` laid on the cells
    // of `bdry_cells`. Past the first layer, the ghost the stencil ends on has to
    // exist: it is not always the case for large stencils.
    template <class Mesh, class Subset, std::size_t stencil_size, class Func>
    void for_each_extrapolation_stencil_on_cells(const Mesh& mesh,
                                                 std::size_t level,
                                                 const DirectionVector<Mesh::dim>& direction,
                                                 Subset& bdry_cells,
                                                 const Stencil<stencil_size, Mesh::dim>& stencil_in_x,
                                                 int ghost_layer,
                                                 Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto stencil_analyzer = make_stencil_analyzer(convert_for_direction(stencil_in_x, direction));

        if (ghost_layer == 1)
        {
            auto cells = intersection(mesh[mesh_id_t::cells][level], bdry_cells).on(level);
            for_each_stencil(mesh, cells, stencil_analyzer, f);
        }
        else
        {
            auto translated_outer_nghbr = translate(mesh[mesh_id_t::reference][level], -ghost_layer * direction);
            auto cells                  = intersection(translated_outer_nghbr, mesh[mesh_id_t::cells][level], bdry_cells).on(level);
            for_each_stencil(mesh, cells, stencil_analyzer, f);
        }
    }

    // Calls f(cells) for each stencil laid on the inner ghosts of
    // `inner_ghosts_location` that have outer neighbours.
    template <std::size_t ghost_layer, class Mesh, class Subset, std::size_t stencil_size, class Func>
    void for_each_extrapolation_stencil_on_ghosts(const Mesh& mesh,
                                                  std::size_t level,
                                                  const DirectionVector<Mesh::dim>& direction,
                                                  Subset& inner_ghosts_location,
                                                  const Stencil<stencil_size, Mesh::dim>& stencil_in_x,
                                                  Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        static_assert(ghost_layer < stencil_size, "the stencil must still reach the boundary cell");

        auto stencil          = convert_for_direction(stencil_in_x, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        auto stencil_origins                  = stencil_origins_in_reference(mesh, level, stencil);
        auto potential_inner_cells_and_ghosts = intersection(stencil_origins, inner_ghosts_location).on(level);
        auto inner_cells_and_ghosts           = intersection(potential_inner_cells_and_ghosts, mesh.get_union()[level]).on(level);
        auto inner_ghosts_with_outer_nghbr    = difference(inner_cells_and_ghosts, mesh[mesh_id_t::cells][level]).on(level);
        for_each_stencil(mesh, inner_ghosts_with_outer_nghbr, stencil_analyzer, f);
    }

    /**
     * For each ghost layer from @a first_layer to @a last_layer, calls fill_layer(stencil, layer), where layer is a
     * std::integral_constant and stencil is the line stencil along the first axis that ends on the ghost of that layer.
     */
    template <std::size_t dim, class Func>
    void for_each_extrapolated_ghost_layer(int first_layer, int last_layer, Func&& fill_layer)
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
                    static constexpr int stencil_start        = static_cast<int>(layer) - static_cast<int>(stencil_size) + 1;

                    fill_layer(line_stencil_from<dim, 0, stencil_size>(stencil_start), ghost_layer_);
                });
        }
    }

    // The ghosts beyond those the B.C. fills, at the boundary cells of `level`
    // in `direction`, layers first_layer to last_layer.
    template <class Mesh, class Visitor>
    void visit_extrapolation_on_cells(const Mesh& mesh,
                                      std::size_t level,
                                      const DirectionVector<Mesh::dim>& direction,
                                      int first_layer,
                                      int last_layer,
                                      Visitor& visitor)
    {
        for_each_extrapolated_ghost_layer<Mesh::dim>(first_layer,
                                                     last_layer,
                                                     [&](const auto& stencil, auto ghost_layer_)
                                                     {
                                                         auto boundary_cells = domain_boundary(mesh, level, direction);
                                                         for_each_extrapolation_stencil_on_cells(
                                                             mesh,
                                                             level,
                                                             direction,
                                                             boundary_cells,
                                                             stencil,
                                                             static_cast<int>(decltype(ghost_layer_)::value),
                                                             [&](const auto& cells)
                                                             {
                                                                 visitor.stencil(cells);
                                                             });
                                                     });
    }

    // The ghosts beyond those project_bc fills, below the boundary ghosts of
    // `level` in `direction`, layers first_layer to last_layer.
    template <class Mesh, class Visitor>
    void visit_extrapolation_on_ghosts(const Mesh& mesh,
                                       std::size_t level,
                                       const DirectionVector<Mesh::dim>& direction,
                                       int first_layer,
                                       int last_layer,
                                       Visitor& visitor)
    {
        for_each_extrapolated_ghost_layer<Mesh::dim>(first_layer,
                                                     last_layer,
                                                     [&](const auto& stencil, auto ghost_layer_)
                                                     {
                                                         static constexpr std::size_t layer = decltype(ghost_layer_)::value;

                                                         const auto& domain   = mesh.domain(level);
                                                         auto boundary_ghosts = difference(domain, translate(domain, -direction));
                                                         for_each_extrapolation_stencil_on_ghosts<layer>(mesh,
                                                                                                         level,
                                                                                                         direction,
                                                                                                         boundary_ghosts,
                                                                                                         stencil,
                                                                                                         [&](const auto& cells)
                                                                                                         {
                                                                                                             visitor.stencil(cells);
                                                                                                         });
                                                     });
    }

    // The corner ghosts of `level` in the diagonal `direction`, in the order
    // update_outer_corners_by_polynomial_extrapolation fills them.
    template <class Mesh, class Visitor>
    void visit_corner_extrapolation(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction, Visitor& visitor)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

        if constexpr (dim == 1)
        {
            return; // No outer corners in 1D
        }

        // PolynomialExtrapolation is only implemented for even stencil_size, so we dispatch directly on the
        // ghost layer (stencil_size = 2 * ghost_layer) instead of on the stencil size, to avoid instantiating
        // the unused odd-stencil_size candidates.
        static constexpr std::size_t max_ghost_layers_PE = max_stencil_size_implemented_PE / 2;

        auto visit = [&](const auto& cells)
        {
            visitor.stencil(cells);
        };

        int ghost_width        = mesh.ghost_width();
        const auto& corner_lca = mesh.corner(direction);

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

            if (static_cast<std::size_t>(2 * ghost_layer) <= max_stencil_size_implemented_PE)
            {
                dispatch_static<1, max_ghost_layers_PE>(static_cast<std::size_t>(ghost_layer),
                                                        [&](auto ghost_layer_)
                                                        {
                                                            static constexpr std::size_t stencil_size = 2 * decltype(ghost_layer_)::value;
                                                            for_each_extrapolation_stencil_on_cells(mesh,
                                                                                                    level,
                                                                                                    direction,
                                                                                                    corner,
                                                                                                    line_stencil<dim, 0, stencil_size>(),
                                                                                                    ghost_layer,
                                                                                                    visit);
                                                        });
            }
            else
            {
                // The ghost of this layer gets the value of the inner diagonal cell at the offset
                // 1 - ghost_layer from the corner cell, 2 * ghost_layer - 1 cells further in. That
                // cell has to be a cell of this level, and the ghost has to exist in the mesh: the
                // same two conditions as the extrapolation above.
                DirectionVector<dim> to_ghost = (2 * ghost_layer - 1) * direction;
                auto mirrored_cells           = intersection(translate(corner, (1 - ghost_layer) * direction),
                                                   mesh[mesh_id_t::cells][level],
                                                   translate(mesh[mesh_id_t::reference][level], -to_ghost))
                                          .on(level);

                Stencil<2, dim> stencil_copy;
                xt::view(stencil_copy, 0) = 0;
                xt::view(stencil_copy, 1) = to_ghost;
                for_each_stencil(mesh, mirrored_cells, make_stencil_analyzer(stencil_copy), visit);
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
        std::array<std::size_t, dim> nonzero_dirs;
        for (std::size_t d = 0; d < dim; ++d)
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
        std::array<DirectionVector<dim>, dim> e_dirs;
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
        // condition as in Step 1, where for_each_extrapolation_stencil_on_cells() intersects the
        // corner with mesh[cells][level].
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
        // fine to coarse (see project_corner_below() in algorithm/update_outer_ghost.hpp, called
        // right after this function in update_outer_ghosts()).
        auto corner_cells = intersection(corner_at_level, mesh[mesh_id_t::cells][level]).on(level);

        for (int k = 1; k <= ghost_width; ++k)
        {
            auto source_at_k = translate(corner_cells, k * direction);

            for (std::size_t combo = 0; combo < num_combos; ++combo)
            {
                DirectionVector<dim> delta;
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

                Stencil<2, dim> stencil_copy;
                xt::view(stencil_copy, 0) = 0;
                xt::view(stencil_copy, 1) = delta;
                for_each_stencil(mesh, source_at_k, make_stencil_analyzer(stencil_copy), visit);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Tables
    //////////////////////////////////////////////////////////////////////

    // Stencils of one size, in application order: the cell indices of each
    // stencil, the ghost to fill last.
    struct ExtrapolationTable
    {
        std::size_t stencil_size = 0;
        std::vector<std::size_t> cells;
    };

    // What a traversal applies, in order: the stencil size changes from one
    // table to the next.
    using ExtrapolationSequence = std::vector<ExtrapolationTable>;

    // The sequences of the ghost layers 1 to size(), each recorded the first time
    // it is asked for: which layers a field extrapolates depends on its B.C., that
    // the mesh does not know. Safe to call from several threads at once: a layer is
    // recorded under a lock, and every later call reads it after one atomic load.
    class LazyExtrapolationLayers
    {
      public:

        LazyExtrapolationLayers() = default;

        explicit LazyExtrapolationLayers(std::size_t n_layers)
            : m_layers(n_layers)
            , m_recorded(n_layers)
        {
        }

        // Copies the layers recorded so far.
        LazyExtrapolationLayers(const LazyExtrapolationLayers& other)
            : m_layers(other.size())
            , m_recorded(other.size())
        {
            for (std::size_t l = 0; l < size(); ++l)
            {
                if (other.m_recorded[l].load(std::memory_order_acquire))
                {
                    m_layers[l] = other.m_layers[l];
                    m_recorded[l].store(true, std::memory_order_relaxed);
                }
            }
        }

        LazyExtrapolationLayers& operator=(const LazyExtrapolationLayers& other)
        {
            LazyExtrapolationLayers recorded_so_far(other);
            swap(recorded_so_far);
            return *this;
        }

        LazyExtrapolationLayers(LazyExtrapolationLayers&&) noexcept            = default;
        LazyExtrapolationLayers& operator=(LazyExtrapolationLayers&&) noexcept = default;
        ~LazyExtrapolationLayers()                                             = default;

        void swap(LazyExtrapolationLayers& other) noexcept
        {
            m_layers.swap(other.m_layers);
            m_recorded.swap(other.m_recorded);
        }

        std::size_t size() const
        {
            return m_layers.size();
        }

        // The sequence of `layer` (from 1), recorded by record(sequence) the first time.
        template <class Record>
        const ExtrapolationSequence& get(int layer, Record&& record) const
        {
            const auto l = static_cast<std::size_t>(layer - 1);
            if (!m_recorded[l].load(std::memory_order_acquire))
            {
                std::lock_guard<std::mutex> lock(recording());
                if (!m_recorded[l].load(std::memory_order_relaxed))
                {
                    record(m_layers[l]);
                    m_recorded[l].store(true, std::memory_order_release);
                }
            }
            return m_layers[l];
        }

      private:

        static std::mutex& recording()
        {
            static std::mutex mutex;
            return mutex;
        }

        mutable std::vector<ExtrapolationSequence> m_layers;
        mutable std::vector<std::atomic<bool>> m_recorded;
    };

    // Appends one stencil to the sequence: a new table when its size changes.
    template <std::size_t size>
    void append_stencil(ExtrapolationSequence& sequence, const std::array<std::size_t, size>& indices)
    {
        if (sequence.empty() || sequence.back().stencil_size != size)
        {
            sequence.push_back({size, {}});
        }
        sequence.back().cells.insert(sequence.back().cells.end(), indices.begin(), indices.end());
    }

    struct ExtrapolationRecorder
    {
        ExtrapolationSequence& sequence;

        template <class Cells>
        void stencil(const Cells& cells)
        {
            constexpr std::size_t size = std::tuple_size_v<Cells>;
            std::array<std::size_t, size> indices;
            for (std::size_t k = 0; k < size; ++k)
            {
                indices[k] = static_cast<std::size_t>(cells[k].index);
            }
            append_stencil(sequence, indices);
        }
    };

    // Same arithmetic as PolynomialExtrapolation, component by component.
    template <std::size_t stencil_size, class value_t>
    void extrapolate_rows(value_t* data, std::size_t nc, const std::vector<std::size_t>& cells)
    {
        for (std::size_t s = 0; s < cells.size(); s += stencil_size)
        {
            const std::size_t* stencil = cells.data() + s;
            for (std::size_t c = 0; c < nc; ++c)
            {
                data[stencil[stencil_size - 1] * nc + c] = extrapolated_value<stencil_size>(
                    [&](std::size_t k)
                    {
                        return data[stencil[k] * nc + c];
                    });
            }
        }
    }

    template <class Field>
    void run_extrapolation(Field& field, const ExtrapolationSequence& sequence)
    {
        constexpr std::size_t nc = Field::n_comp;
        auto* data               = field.data();
        for (const auto& table : sequence)
        {
            switch (table.stencil_size)
            {
                case 2:
                    extrapolate_rows<2>(data, nc, table.cells);
                    break;
                case 4:
                    extrapolate_rows<4>(data, nc, table.cells);
                    break;
                case 6:
                    extrapolate_rows<6>(data, nc, table.cells);
                    break;
                default:
                    break;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Tables from direct walks
    //////////////////////////////////////////////////////////////////////

    // The same stencils as the traversals above, found by walking the rows of
    // the mesh near the face of a box domain instead of building set
    // expressions. Each scan returns false when it meets a stencil cell that is
    // not in the reference mesh where the traversal does not check it: the
    // caller then records the traversal itself.

    // What a traversal checks before laying a stencil.
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

    // Appends the stencils laid on the cells [start, end) of the row of
    // `coords`, their cells read in `ref`.
    template <class LCA, std::size_t size>
    bool record_row_stencils(const LCA& ref,
                             const std::array<typename LCA::value_t, LCA::dim>& coords,
                             typename LCA::value_t start,
                             typename LCA::value_t end,
                             const std::array<DirectionVector<LCA::dim>, size>& stencil,
                             CheckedCells checked,
                             ExtrapolationSequence& sequence)
    {
        auto rows = [&]<std::size_t... k>(std::index_sequence<k...>)
        {
            return std::array<RowLookup<LCA>, size>{RowLookup<LCA>(ref, translate(coords, stencil[k]))...};
        }(std::make_index_sequence<size>{});

        std::array<std::size_t, size> indices;
        for (auto x = start; x < end; ++x)
        {
            bool missing_ghost = false;
            bool missing_other = false;
            for (std::size_t k = 0; k < size; ++k)
            {
                const auto index = rows[k](x + stencil[k][0]);
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
            append_stencil(sequence, indices);
        }
        return true;
    }

    // visit_extrapolation_on_cells of one layer. The cells lie in the domain,
    // so the boundary cells are those on its face.
    template <class Mesh>
    bool scan_extrapolation_on_cells(const Mesh& mesh,
                                     std::size_t level,
                                     const CellBlock<typename Mesh::interval_t::value_t, Mesh::dim>& domain,
                                     const DirectionVector<Mesh::dim>& direction,
                                     int layer,
                                     ExtrapolationSequence& sequence)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        bool done = true;
        for_each_extrapolated_ghost_layer<Mesh::dim>(
            layer,
            layer,
            [&](const auto& stencil_in_x, auto)
            {
                const auto stencil = stencil_offsets(convert_for_direction(stencil_in_x, direction));
                const auto checked = layer == 1 ? CheckedCells::none : CheckedCells::ghost;
                for_each_interval_in_block(
                    mesh[mesh_id_t::cells][level],
                    face_layer(domain, direction, 0),
                    [&](auto start, auto end, const auto& coords, auto)
                    {
                        done = done && record_row_stencils(mesh[mesh_id_t::reference][level], coords, start, end, stencil, checked, sequence);
                    });
            });
        return done;
    }

    // visit_extrapolation_on_ghosts of one layer: the union cells on the face
    // of the domain that are not cells, where the whole stencil is in the
    // reference mesh.
    template <class Mesh>
    void scan_extrapolation_on_ghosts(const Mesh& mesh,
                                      std::size_t level,
                                      const CellBlock<typename Mesh::interval_t::value_t, Mesh::dim>& domain,
                                      const DirectionVector<Mesh::dim>& direction,
                                      int layer,
                                      ExtrapolationSequence& sequence)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        for_each_extrapolated_ghost_layer<Mesh::dim>(
            layer,
            layer,
            [&](const auto& stencil_in_x, auto)
            {
                const auto stencil = stencil_offsets(convert_for_direction(stencil_in_x, direction));
                const auto& ref    = mesh[mesh_id_t::reference][level];
                for_each_interval_in_block(mesh.get_union()[level],
                                           face_layer(domain, direction, 0),
                                           [&](auto start, auto end, const auto& coords, auto)
                                           {
                                               const auto row = coords;
                                               auto x         = start;
                                               auto gap       = [&](auto gap_end)
                                               {
                                                   if (x < gap_end)
                                                   {
                                                       record_row_stencils(ref, row, x, gap_end, stencil, CheckedCells::all, sequence);
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
            });
    }

    // visit_corner_extrapolation.
    template <class Mesh>
    bool
    scan_corner_extrapolation(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction, ExtrapolationSequence& sequence)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

        const auto corner = corner_block(mesh, direction, level);
        if (!corner)
        {
            return false;
        }
        const auto& cells = mesh[mesh_id_t::cells][level];
        const auto& ref   = mesh[mesh_id_t::reference][level];
        bool done         = true;
        auto record       = [&](const auto& block, const auto& stencil, CheckedCells checked)
        {
            for_each_interval_in_block(cells,
                                       block,
                                       [&](auto start, auto end, const auto& coords, auto)
                                       {
                                           done = done && record_row_stencils(ref, coords, start, end, stencil, checked, sequence);
                                       });
        };

        // Step 1: the diagonal ghosts, layer by layer
        const int ghost_width = static_cast<int>(mesh.ghost_width());
        for (int ghost_layer = 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            const auto checked = ghost_layer == 1 ? CheckedCells::none : CheckedCells::ghost;
            if (static_cast<std::size_t>(2 * ghost_layer) <= max_stencil_size_implemented_PE)
            {
                dispatch_static<1, max_stencil_size_implemented_PE / 2>(
                    static_cast<std::size_t>(ghost_layer),
                    [&](auto ghost_layer_)
                    {
                        static constexpr std::size_t stencil_size = 2 * decltype(ghost_layer_)::value;
                        record(*corner, stencil_offsets(convert_for_direction(line_stencil<dim, 0, stencil_size>(), direction)), checked);
                    });
            }
            else
            {
                DirectionVector<dim> to_ghost = (2 * ghost_layer - 1) * direction;
                DirectionVector<dim> mirror   = (1 - ghost_layer) * direction;
                std::array<DirectionVector<dim>, 2> stencil;
                stencil[0].fill(0);
                stencil[1] = to_ghost;
                record(translate(*corner, mirror), stencil, checked);
            }
        }

        // Step 2: the off-diagonal ghosts, copies of the diagonal ones, in the
        // order of visit_corner_extrapolation
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
            return done;
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
                if (is_source)
                {
                    continue;
                }
                std::array<DirectionVector<dim>, 2> stencil;
                stencil[0] = k * direction;
                stencil[1] = stencil[0] + delta;
                record(*corner, stencil, CheckedCells::none);
            }
        }
        return done;
    }

    // The layer `layer` of visit_extrapolation_on_cells, recorded from direct
    // walks on a box domain.
    template <class Mesh>
    void record_extrapolation_on_cells(const Mesh& mesh,
                                       std::size_t level,
                                       const DirectionVector<Mesh::dim>& direction,
                                       int layer,
                                       ExtrapolationSequence& sequence)
    {
        using mesh_id_t   = typename Mesh::mesh_id_t;
        const auto domain = as_block(mesh.domain(level));
        if (!domain || !block_holds(*domain, mesh[mesh_id_t::cells][level])
            || !scan_extrapolation_on_cells(mesh, level, *domain, direction, layer, sequence))
        {
            sequence.clear();
            ExtrapolationRecorder recorder{sequence};
            visit_extrapolation_on_cells(mesh, level, direction, layer, layer, recorder);
        }
    }

    // Index of a diagonal direction among the 3^dim directions.
    template <std::size_t dim>
    std::size_t direction_code(const DirectionVector<dim>& direction)
    {
        std::size_t code = 0;
        std::size_t base = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            code += static_cast<std::size_t>(direction[d] + 1) * base;
            base *= 3;
        }
        return code;
    }
}
