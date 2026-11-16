// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "../entry_table.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../static_algorithm.hpp"
#include "../subset/node.hpp"

// The ghosts update_ghost_mr fills by field projection and by prediction, as
// entry table keys, and the kernels that fill them: those of projection_op_
// and of predict_rows.
namespace samurai::detail
{
    //////////////////////////////////////////////////////////////////////
    // Field projection
    //////////////////////////////////////////////////////////////////////

    // The ghosts at `level` filled with the mean of their children at level + 1.
    // at[0] is the row of ghosts, the others the rows of children whose first
    // cell is the first child, in the order of static_nested_loop.
    template <std::size_t dim>
    struct projection_ghosts
    {
        using entry_type = Rows<1 + (std::size_t{1} << (dim - 1))>;

        std::size_t level;

        template <class Mesh, class Emit>
        void produce(const Mesh& mesh, Emit& emit) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            auto set        = intersection(mesh[mesh_id_t::reference][level + 1], mesh[mesh_id_t::proj_cells][level]).on(level);
            set(
                [&](const auto& i, const auto& index)
                {
                    entry_type rows{i.size(), {}};
                    rows.at[0] = memory_offset(mesh, {level, i.start, index});
                    if constexpr (dim == 1)
                    {
                        rows.at[1] = memory_offset(mesh, {level + 1, 2 * i.start, index});
                    }
                    else
                    {
                        std::size_t s = 1;
                        static_nested_loop<dim - 1, 0, 2>(
                            [&](const auto& stencil)
                            {
                                rows.at[s++] = memory_offset(mesh, {level + 1, 2 * i.start, 2 * index + stencil});
                            });
                    }
                    emit(rows);
                });
        }
    };

    // The arithmetic of projection_op_.
    template <class Field, std::size_t N>
    void project_entry(Field& field, const Rows<N>& rows)
    {
        auto* data = field.data();
        project_rows<Field::dim, Field::n_comp>(data, rows.at[0], data, std::span(rows.at).template subspan<1>(), rows.len);
    }

    //////////////////////////////////////////////////////////////////////
    // Prediction
    //////////////////////////////////////////////////////////////////////

    // One interval [lo, hi) of ghosts predicted from the level below: the first
    // ghost, the parity of the row in each direction above x, and the first
    // coarse column of each source row, z outer and y inner.
    template <std::size_t dim, std::size_t n_src>
    struct PredictionEntry
    {
        std::size_t dest;
        std::ptrdiff_t lo;
        std::ptrdiff_t hi;
        std::array<std::ptrdiff_t, dim - 1> parity;
        std::array<std::size_t, n_src> src;
    };

    // The ghosts at `level` predicted from level - 1.
    template <std::size_t dim, std::size_t radius>
    struct prediction_ghosts
    {
        static constexpr std::size_t n_src = ce_pow(2 * radius + 1, dim - 1);
        using entry_type                   = PredictionEntry<dim, n_src>;

        std::size_t level;

        template <class Mesh, class Emit>
        void produce(const Mesh& mesh, Emit& emit) const
        {
            using mesh_id_t  = typename Mesh::mesh_id_t;
            using value_t    = typename Mesh::interval_t::value_t;
            constexpr auto r = static_cast<value_t>(radius);

            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            auto set         = intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1]).on(level);
            set(
                [&](const auto& i, const auto& index)
                {
                    entry_type entry{};
                    entry.dest  = memory_offset(mesh, {level, i.start, index});
                    entry.lo    = i.start;
                    entry.hi    = i.end;
                    auto parent = index;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        entry.parity[d] = index[d] & 1;
                        parent[d]       = index[d] >> 1;
                    }
                    // the source rows around the parent row, from the column x0
                    const value_t x0 = (i.start >> 1) - r;
                    if constexpr (dim == 1)
                    {
                        entry.src[0] = memory_offset(mesh, {level - 1, x0, parent});
                    }
                    else
                    {
                        std::size_t s = 0;
                        static_nested_loop<dim - 1, -static_cast<int>(radius), static_cast<int>(radius) + 1>(
                            [&](const auto& shift)
                            {
                                // the last direction changes slowest, as predict_rows reads them
                                entry.src[s++] = memory_offset(mesh, {level - 1, x0, parent + shift});
                            });
                    }
                    emit(entry);
                });
        }
    };

    // The arithmetic of predict_rows with one fine row, its coarse rows read from the entry.
    template <std::size_t radius, class Field, std::size_t dim, std::size_t n_src>
    void predict_entry(Field& field, const PredictionEntry<dim, n_src>& entry)
    {
        auto* data = field.data();
        const listed_stencil_rows<2 * radius + 1, std::remove_reference_t<decltype(*data)>> stencil_rows{
            entry.src.data(),
            data,
            (entry.lo >> 1) - static_cast<std::ptrdiff_t>(radius)};
        predict_rows_from<1, radius, Field::n_comp, dim>(data, {entry.dest}, stencil_rows, entry.lo, entry.hi, {}, entry.parity);
    }
}
