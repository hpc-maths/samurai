// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../algorithm/outer_ghost_sets.hpp"
#include "../algorithm/update_periodic.hpp"
#include "../numeric/prediction_coefficients.hpp"
#include "../static_algorithm.hpp"
#include "../subset/node.hpp"
#include "../utils.hpp"

namespace samurai::detail
{
    // One entry per interval of ghosts to fill at a level: the storage index of
    // its first cell, its bounds along x, and the storage index of the first cell
    // of each source row.
    template <std::size_t dim, class value_t>
    struct GhostTable
    {
        std::vector<std::size_t> dest;
        std::vector<value_t> lo;
        std::vector<value_t> hi;
        std::vector<std::uint8_t> parity; // prediction only: parity of the row in y (bit 0) and z (bit 1)
        std::vector<std::size_t> src;     // n_src rows per entry
    };

    // The field projection, the prediction, the periodic copies and the ghosts
    // outside the domain filled from other ghosts of update_ghost_mr as flat
    // tables, built once per mesh from the same set expressions. Running them gives the
    // same values, bit for bit, without traversing the set expressions again.
    template <std::size_t dim, class value_t>
    struct GhostTables
    {
        using table_t = GhostTable<dim, value_t>;

        std::vector<table_t> projection;          // projection[l]: ghosts at l, mean of their children at l + 1
        std::vector<table_t> prediction;          // prediction[l]: ghosts at l, predicted from l - 1
        std::vector<PeriodicGhostTable> periodic; // periodic[l]: periodic ghosts at l
        std::vector<OuterGhostTables> outer;      // outer[l]: ghosts at l outside a box domain, empty otherwise

        // No tables: the mesh was never finalized.
        GhostTables() = default;

        bool empty() const
        {
            return projection.empty();
        }

        template <class Mesh>
            requires(!std::same_as<Mesh, GhostTables>)
        explicit GhostTables(const Mesh& mesh)
        {
            using mesh_id_t      = typename Mesh::mesh_id_t;
            const auto max_level = mesh.max_level();
            constexpr auto r     = static_cast<value_t>(Mesh::config_t::prediction_stencil_radius);
            projection.resize(max_level + 1);
            prediction.resize(max_level + 1);
            periodic.reserve(max_level + 1);
            for (std::size_t level = 0; level <= max_level; ++level)
            {
                periodic.push_back(make_periodic_ghost_table(level, mesh));
            }
            if (mesh.domain().is_box())
            {
                outer.reserve(max_level + 1);
                for (std::size_t level = 0; level <= max_level; ++level)
                {
                    outer.push_back(make_outer_ghost_tables(level, mesh));
                }
            }

            auto offset = [&](std::size_t level, value_t x, const auto& index)
            {
                return static_cast<std::size_t>(memory_offset(mesh, {level, x, index}));
            };

            for (std::size_t level = 1; level <= max_level; ++level)
            {
                auto& tp = projection[level - 1];
                auto ps  = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
                ps(
                    [&](const auto& i, const auto& index)
                    {
                        tp.dest.push_back(offset(level - 1, i.start, index));
                        tp.lo.push_back(i.start);
                        tp.hi.push_back(i.end);
                        if constexpr (dim == 1)
                        {
                            tp.src.push_back(offset(level, 2 * i.start, index));
                        }
                        else
                        {
                            static_nested_loop<dim - 1, 0, 2>(
                                [&](const auto& stencil)
                                {
                                    tp.src.push_back(offset(level, 2 * i.start, 2 * index + stencil));
                                });
                        }
                    });

                auto& tq         = prediction[level];
                auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                              union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
                auto qs          = intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1]).on(level);
                qs(
                    [&](const auto& i, const auto& index)
                    {
                        tq.dest.push_back(offset(level, i.start, index));
                        tq.lo.push_back(i.start);
                        tq.hi.push_back(i.end);
                        std::uint8_t parity = 0;
                        auto parent         = index;
                        for (std::size_t d = 0; d < dim - 1; ++d)
                        {
                            parity |= static_cast<std::uint8_t>((index[d] & 1) << d);
                            parent[d] = index[d] >> 1;
                        }
                        tq.parity.push_back(parity);
                        // first coarse column of the stencil, rows in the order of the contraction: z outer, y inner
                        const value_t x0 = (i.start >> 1) - r;
                        if constexpr (dim == 1)
                        {
                            tq.src.push_back(offset(level - 1, x0, parent));
                        }
                        else if constexpr (dim == 2)
                        {
                            for (value_t ky = -r; ky <= r; ++ky)
                            {
                                auto source = parent;
                                source[0] += ky;
                                tq.src.push_back(offset(level - 1, x0, source));
                            }
                        }
                        else
                        {
                            for (value_t kz = -r; kz <= r; ++kz)
                            {
                                for (value_t ky = -r; ky <= r; ++ky)
                                {
                                    auto source = parent;
                                    source[0] += ky;
                                    source[1] += kz;
                                    tq.src.push_back(offset(level - 1, x0, source));
                                }
                            }
                        }
                    });
            }
        }
    };

    // Same arithmetic as projection_op_.
    template <class Field, class Table>
    void project_ghosts_with_table(Field& field, const Table& table)
    {
        constexpr std::size_t n_src = std::size_t{1} << (Field::dim - 1);
        constexpr std::size_t nc    = Field::n_comp;
        constexpr double inv        = 1.0 / static_cast<double>(1ULL << Field::dim);
        auto* data                  = field.data();
        for (std::size_t e = 0; e < table.dest.size(); ++e)
        {
            const auto len  = static_cast<std::size_t>(table.hi[e] - table.lo[e]);
            const auto* src = &table.src[e * n_src];
            for (std::size_t ii = 0, i_f = 0; ii < len; ++ii, i_f += 2)
            {
                std::array<double, nc> sum;
                sum.fill(0);
                for (std::size_t s = 0; s < n_src; ++s)
                {
                    for (std::size_t n = 0; n < nc; ++n)
                    {
                        sum[n] += data[(src[s] + i_f) * nc + n] + data[(src[s] + i_f + 1) * nc + n];
                    }
                }
                for (std::size_t n = 0; n < nc; ++n)
                {
                    data[(table.dest[e] + ii) * nc + n] = sum[n] * inv;
                }
            }
        }
    }

    // Same arithmetic as predict_rows with one fine row: the directions above x
    // are contracted column by column (y, then z), then x.
    template <class Field, class Table>
    void predict_ghosts_with_table(Field& field, const Table& table)
    {
        constexpr std::size_t radius = Field::mesh_t::config_t::prediction_stencil_radius;
        constexpr std::size_t dim    = Field::dim;
        constexpr std::size_t size   = 2 * radius + 1;
        constexpr std::size_t nc     = Field::n_comp;
        constexpr std::size_t n_src  = ce_pow(size, dim - 1);

        const std::array<std::array<double, size>, 2> c = {
            {prediction_coefficients<radius>(0, 0).c, prediction_coefficients<radius>(1, 0).c}
        };
        auto* data = field.data();
        std::vector<double> buffer;

        for (std::size_t e = 0; e < table.dest.size(); ++e)
        {
            const auto lo         = table.lo[e];
            const auto hi         = table.hi[e];
            const auto c_lo       = lo >> 1;
            const auto c_hi       = ((hi - 1) >> 1) + 1;
            const std::size_t len = (static_cast<std::size_t>(c_hi - c_lo) + 2 * radius) * nc;
            const auto* src       = &table.src[e * n_src];

            // coarse columns [c_lo - r, c_hi + r) summed above x, nc values per column
            const double* columns = nullptr;
            if constexpr (dim == 1)
            {
                // the source row itself, as in predict_rows
                buffer.assign(data + src[0] * nc, data + src[0] * nc + len);
                columns = buffer.data();
            }
            else
            {
                buffer.resize(2 * len);
                double* out    = buffer.data();
                const auto& cy = c[table.parity[e] & 1];
                if constexpr (dim == 2)
                {
                    for (std::size_t k = 0; k < size; ++k)
                    {
                        const auto* source = data + src[k] * nc;
                        for (std::size_t m = 0; m < len; ++m)
                        {
                            out[m] = (k == 0 ? 0. : out[m]) + cy[k] * source[m];
                        }
                    }
                }
                else
                {
                    const auto& cz = c[(table.parity[e] >> 1) & 1];
                    double* sub    = buffer.data() + len;
                    for (std::size_t kz = 0; kz < size; ++kz)
                    {
                        for (std::size_t ky = 0; ky < size; ++ky)
                        {
                            const auto* source = data + src[kz * size + ky] * nc;
                            for (std::size_t m = 0; m < len; ++m)
                            {
                                sub[m] = (ky == 0 ? 0. : sub[m]) + cy[ky] * source[m];
                            }
                        }
                        for (std::size_t m = 0; m < len; ++m)
                        {
                            out[m] = (kz == 0 ? 0. : out[m]) + cz[kz] * sub[m];
                        }
                    }
                }
                columns = out;
            }

            auto* d = data + table.dest[e] * nc;
            for (auto f = lo; f < hi; ++f)
            {
                const auto& cf  = c[static_cast<std::size_t>(f & 1)];
                const double* x = columns + static_cast<std::size_t>((f >> 1) - c_lo) * nc;
                for (std::size_t n = 0; n < nc; ++n)
                {
                    double v = 0.;
                    for (std::size_t s = 0; s < size; ++s)
                    {
                        v += cf[s] * x[s * nc + n];
                    }
                    d[static_cast<std::size_t>(f - lo) * nc + n] = v;
                }
            }
        }
    }
}
