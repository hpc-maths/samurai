// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../entry_table.hpp"
#include "../numeric/prediction_coefficients.hpp"
#include "../static_algorithm.hpp"
#include "../subset/node.hpp"

// The ghosts update_ghost_mr fills by field projection and by prediction, as
// entry table keys, and the kernels that fill them.
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
            auto offset     = [&](std::size_t l, auto x, const auto& index)
            {
                return static_cast<std::size_t>(memory_offset(mesh, {l, x, index}));
            };
            auto set = intersection(mesh[mesh_id_t::reference][level + 1], mesh[mesh_id_t::proj_cells][level]).on(level);
            set(
                [&](const auto& i, const auto& index)
                {
                    entry_type run{i.size(), {}};
                    run.at[0] = offset(level, i.start, index);
                    if constexpr (dim == 1)
                    {
                        run.at[1] = offset(level + 1, 2 * i.start, index);
                    }
                    else
                    {
                        std::size_t s = 1;
                        static_nested_loop<dim - 1, 0, 2>(
                            [&](const auto& stencil)
                            {
                                run.at[s++] = offset(level + 1, 2 * i.start, 2 * index + stencil);
                            });
                    }
                    emit(run);
                });
        }
    };

    // Same arithmetic as projection_op_.
    template <class Field, std::size_t N>
    void project_run(Field& field, const Rows<N>& run)
    {
        constexpr std::size_t nc = Field::n_comp;
        constexpr double inv     = 1.0 / static_cast<double>(1ULL << Field::dim);
        auto* data               = field.data();
        for (std::size_t ii = 0, i_f = 0; ii < run.len; ++ii, i_f += 2)
        {
            std::array<double, nc> sum;
            sum.fill(0);
            for (std::size_t s = 1; s < N; ++s)
            {
                for (std::size_t n = 0; n < nc; ++n)
                {
                    sum[n] += data[(run.at[s] + i_f) * nc + n] + data[(run.at[s] + i_f + 1) * nc + n];
                }
            }
            for (std::size_t n = 0; n < nc; ++n)
            {
                data[(run.at[0] + ii) * nc + n] = sum[n] * inv;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Prediction
    //////////////////////////////////////////////////////////////////////

    // One interval [lo, hi) of ghosts predicted from the level below: the first
    // ghost, the parity of the row in y (bit 0) and z (bit 1), and the first
    // coarse column of each source row, z outer and y inner.
    template <std::size_t n_src>
    struct PredictionEntry
    {
        std::size_t dest;
        std::ptrdiff_t lo;
        std::ptrdiff_t hi;
        std::uint8_t parity;
        std::array<std::size_t, n_src> src;
    };

    // The ghosts at `level` predicted from level - 1.
    template <std::size_t dim, std::size_t radius>
    struct prediction_ghosts
    {
        static constexpr std::size_t n_src = ce_pow(2 * radius + 1, dim - 1);
        using entry_type                   = PredictionEntry<n_src>;

        std::size_t level;

        template <class Mesh, class Emit>
        void produce(const Mesh& mesh, Emit& emit) const
        {
            using mesh_id_t  = typename Mesh::mesh_id_t;
            using value_t    = typename Mesh::interval_t::value_t;
            constexpr auto r = static_cast<value_t>(radius);
            auto offset      = [&](std::size_t l, auto x, const auto& index)
            {
                return static_cast<std::size_t>(memory_offset(mesh, {l, x, index}));
            };
            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            auto set         = intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1]).on(level);
            set(
                [&](const auto& i, const auto& index)
                {
                    entry_type entry{};
                    entry.dest  = offset(level, i.start, index);
                    entry.lo    = i.start;
                    entry.hi    = i.end;
                    auto parent = index;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        entry.parity |= static_cast<std::uint8_t>((index[d] & 1) << d);
                        parent[d] = index[d] >> 1;
                    }
                    const value_t x0 = (i.start >> 1) - r;
                    if constexpr (dim == 1)
                    {
                        entry.src[0] = offset(level - 1, x0, parent);
                    }
                    else if constexpr (dim == 2)
                    {
                        std::size_t s = 0;
                        for (value_t ky = -r; ky <= r; ++ky)
                        {
                            auto source = parent;
                            source[0] += ky;
                            entry.src[s++] = offset(level - 1, x0, source);
                        }
                    }
                    else
                    {
                        std::size_t s = 0;
                        for (value_t kz = -r; kz <= r; ++kz)
                        {
                            for (value_t ky = -r; ky <= r; ++ky)
                            {
                                auto source = parent;
                                source[0] += ky;
                                source[1] += kz;
                                entry.src[s++] = offset(level - 1, x0, source);
                            }
                        }
                    }
                    emit(entry);
                });
        }
    };

    // Same arithmetic as predict_rows with one fine row: the directions above x
    // are contracted column by column (y, then z), then x.
    template <std::size_t dim, std::size_t radius>
    class PredictionKernel
    {
      public:

        static constexpr std::size_t size = 2 * radius + 1;

        template <class Field, class Entry>
        void operator()(Field& field, const Entry& entry)
        {
            constexpr std::size_t nc = Field::n_comp;
            auto* data               = field.data();

            const auto lo         = entry.lo;
            const auto hi         = entry.hi;
            const auto c_lo       = lo >> 1;
            const auto c_hi       = ((hi - 1) >> 1) + 1;
            const std::size_t len = (static_cast<std::size_t>(c_hi - c_lo) + 2 * radius) * nc;
            const auto& src       = entry.src;

            // coarse columns [c_lo - r, c_hi + r) summed above x, nc values per column
            const double* columns = nullptr;
            if constexpr (dim == 1)
            {
                // the source row itself, as in predict_rows
                m_buffer.assign(data + src[0] * nc, data + src[0] * nc + len);
                columns = m_buffer.data();
            }
            else
            {
                m_buffer.resize(2 * len);
                double* out    = m_buffer.data();
                const auto& cy = m_c[entry.parity & 1];
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
                    const auto& cz = m_c[(entry.parity >> 1) & 1];
                    double* sub    = m_buffer.data() + len;
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

            auto* d = data + entry.dest * nc;
            for (auto f = lo; f < hi; ++f)
            {
                const auto& cf  = m_c[static_cast<std::size_t>(f & 1)];
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

      private:

        std::array<std::array<double, size>, 2> m_c = {
            {prediction_coefficients<radius>(0, 0).c, prediction_coefficients<radius>(1, 0).c}
        };
        std::vector<double> m_buffer;
    };
}
