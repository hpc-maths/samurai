// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "../../algorithm.hpp"
#include "../../algorithm/update_ghost_mr.hpp"
#include "../../reconstruction.hpp"
#include "velocity_scheme.hpp"

namespace samurai
{
    /**
     * @class LBMScheme
     *
     * A Lattice Boltzmann scheme, expressed as a compile-time list of elementary
     * velocity schemes (@ref VelocityScheme). It carries two fields on the same
     * adapted mesh: the distributions @a f (the numerical unknowns) and the
     * moments @a m (the physical variables, on which adaptation and I/O are done).
     *
     * A single time step is @c stream then @c collide:
     *
     *     stream(f)  ->  f2m  ->  relax (MRT)  ->  m2f
     *
     * with @c relax the collision @c m_k += s_k (m_k^eq - m_k).
     *
     * @note Step 1: uniform mesh, @c stream is a nearest-neighbour shift and the
     *       collision is done per cell (correctness first). The multi-level
     *       stream (portions / precomputed prediction-maps) comes in step 2.
     */
    template <class Field, class... Blocks>
    class LBMScheme
    {
      public:

        using field_t                        = Field;
        static constexpr std::size_t dim     = Field::dim;
        static constexpr std::size_t n_comp  = Field::n_comp;
        static constexpr std::size_t nblocks = sizeof...(Blocks);

        static_assert(((Blocks::dim == dim) && ...), "all velocity schemes must share the field dimension");
        static_assert((Blocks::q + ...) == n_comp, "the sum of the block sizes must equal the field n_comp");

        LBMScheme(std::string name, double lambda, Blocks... blocks)
            : m_name(std::move(name))
            , m_lambda(lambda)
            , m_blocks(std::move(blocks)...)
        {
        }

        const std::string& name() const
        {
            return m_name;
        }

        double lambda() const
        {
            return m_lambda;
        }

        /**
         * Finest level of the hierarchy, used to compute the level jump j = max_level - level
         * that drives the multi-level stream. If left unset (0), the current mesh finest level
         * is used (correct for a uniform mesh, where j == 0 everywhere).
         */
        void set_max_level(std::size_t max_level)
        {
            m_max_level = max_level;
        }

        /**
         * Initialise the distributions @a f from a moment field @a m: the user sets
         * the conserved moments (s_k == 0) in @a m, this fills the non-conserved
         * moments with their equilibrium value and sets f = M^{-1} m.
         */
        template <class MField>
        void init_equilibrium(field_t& f, const MField& m) const
        {
            for_each_cell(f.mesh(),
                          [&](const auto& cell)
                          {
                              auto mc = m[cell];
                              auto fc = f[cell];
                              for_each_block(
                                  [&](const auto& block, std::size_t offset)
                                  {
                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                      std::array<double, q> mblock, meq;
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          mblock[k] = mc(offset + k);
                                      }
                                      block.equilibrium(meq, mblock);
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          if (block.s[k] != 0.)
                                          {
                                              mblock[k] = meq[k];
                                          }
                                      }
                                      const auto fblock = matvec(block.invM, mblock);
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          fc(offset + k) = fblock[k];
                                      }
                                  });
                          });
        }

        /**
         * One LBM time step. Updates both @a f (distributions) and @a m (moments).
         */
        template <class MField>
        void operator()(field_t& f, MField& m) const
        {
            update_ghost_mr(f);
            auto f_stream = f; // same mesh; stream overwrites every (real) cell
            stream(f, f_stream);
            std::swap(f.array(), f_stream.array());
            collide(f, m);
        }

      private:

        // y = A.x  (A is q x q, row-major); q is compile-time so the loops unroll.
        template <std::size_t q>
        static std::array<double, q> matvec(const std::array<std::array<double, q>, q>& A, const std::array<double, q>& x)
        {
            std::array<double, q> y{};
            for (std::size_t r = 0; r < q; ++r)
            {
                double acc = 0.;
                for (std::size_t c = 0; c < q; ++c)
                {
                    acc += A[r][c] * x[c];
                }
                y[r] = acc;
            }
            return y;
        }

        // Iterate over the blocks, threading each block's component offset in the field.
        template <class F>
        void for_each_block(F&& f) const
        {
            std::apply(
                [&](const auto&... block)
                {
                    std::size_t offset = 0;
                    (
                        [&]
                        {
                            f(block, offset);
                            offset += std::decay_t<decltype(block)>::q;
                        }(),
                        ...);
                },
                m_blocks);
        }

        using index_t         = default_config::value_t;
        using stream_stencil_t = prediction_map<dim, index_t>;

        static constexpr std::size_t pred_order = Field::mesh_t::config_t::prediction_stencil_radius;

        /**
         * Build the per-cell stream stencil for a velocity component at a given level jump j.
         *
         * The transport is linear, so the reconstruction of the fine "slab" entering/leaving a
         * coarse cell over one time step composes into a single prediction map applied at the
         * cell's own level:
         *
         *     f_out(C) = f_in(C) + sum_{(o,w)} w * f_in(C + o).
         *
         * With coeff = 1/2^j and P = prediction(j, .) the MR reconstruction of a fine sub-cell:
         *   - c > 0: inflow = last sub-cell of C-1, outflow = last sub-cell of C  -> P = prediction(j, 2^j - 1)
         *   - c < 0: inflow = first sub-cell of C+1, outflow = first sub-cell of C -> P = prediction(j, 0)
         * At the finest level (j == 0) this degenerates to the plain nearest-neighbour shift.
         */
        static stream_stencil_t build_stream_stencil(std::size_t j, int cx)
        {
            stream_stencil_t stencil;
            if (cx == 0)
            {
                return stencil; // rest velocity: no transport
            }
            assert(std::abs(cx) == 1 && "|velocity| > 1 (e.g. D1Q5) is not supported yet");

            const double coeff = 1. / static_cast<double>(index_t{1} << j);

            if (cx > 0)
            {
                const auto& P = prediction<pred_order>(j, static_cast<index_t>((index_t{1} << j) - 1));
                for (const auto& [o, w] : P.coeff)
                {
                    stencil({o[0] - 1}) += w; // inflow from C-1
                    stencil({o[0]}) -= w;     // outflow from C
                }
            }
            else
            {
                const auto& P = prediction<pred_order>(j, static_cast<index_t>(0));
                for (const auto& [o, w] : P.coeff)
                {
                    stencil({o[0] + 1}) += w; // inflow from C+1
                    stencil({o[0]}) -= w;     // outflow from C
                }
            }
            stencil *= coeff;
            stencil.remove_small_entries();
            return stencil;
        }

        // stream: multi-level transport using precomputed inflow/outflow prediction maps (dim == 1).
        void stream(const field_t& f_in, field_t& f_out) const
        {
            static_assert(dim == 1, "multi-dimensional stream comes in a later step");
            using mesh_id_t = typename field_t::mesh_t::mesh_id_t;

            auto& mesh                  = f_in.mesh();
            const std::size_t max_level = (m_max_level == 0) ? mesh.max_level() : m_max_level;
            const int ghost_width       = static_cast<int>(mesh.cfg().ghost_width());

            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                const std::size_t j = max_level - level;

                // Precompute the stencil of every velocity component once per level.
                std::vector<std::pair<std::size_t, stream_stencil_t>> stencils;
                for_each_block(
                    [&](const auto& block, std::size_t offset)
                    {
                        constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                        for (std::size_t a = 0; a < q; ++a)
                        {
                            auto stencil = build_stream_stencil(j, block.velocities[a][0]);
                            for ([[maybe_unused]] const auto& [o, w] : stencil.coeff)
                            {
                                assert(std::abs(o[0]) <= ghost_width
                                       && "stream stencil reaches beyond the ghost band; increase max_stencil_size");
                            }
                            stencils.emplace_back(offset + a, std::move(stencil));
                        }
                    });

                for_each_interval(mesh[mesh_id_t::cells][level],
                                  [&](std::size_t lvl, const auto& i, const auto&)
                                  {
                                      for (const auto& [comp, stencil] : stencils)
                                      {
                                          auto res = xt::eval(f_in(comp, lvl, i));
                                          for (const auto& [o, w] : stencil.coeff)
                                          {
                                              res += w * f_in(comp, lvl, i + o[0]);
                                          }
                                          f_out(comp, lvl, i) = res;
                                      }
                                  });
            }
        }

        // collide: m = M.f ; relax (MRT) ; f = M^{-1} m  (local, per cell).
        template <class MField>
        void collide(field_t& f, MField& m) const
        {
            for_each_cell(f.mesh(),
                          [&](const auto& cell)
                          {
                              auto fc = f[cell];
                              auto mc = m[cell];
                              for_each_block(
                                  [&](const auto& block, std::size_t offset)
                                  {
                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                      std::array<double, q> fblock, meq;
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          fblock[k] = fc(offset + k);
                                      }
                                      auto mblock = matvec(block.M, fblock); // f2m
                                      block.equilibrium(meq, mblock);        // relax (MRT)
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          mblock[k] += block.s[k] * (meq[k] - mblock[k]);
                                      }
                                      const auto fnew = matvec(block.invM, mblock); // m2f
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          fc(offset + k) = fnew[k];
                                          mc(offset + k) = mblock[k];
                                      }
                                  });
                          });
        }

        std::string m_name;
        double m_lambda;
        std::size_t m_max_level = 0; // 0 = use the current mesh finest level (uniform case)
        std::tuple<Blocks...> m_blocks;
    };

    /**
     * Factory: build an @ref LBMScheme from a list of velocity blocks.
     * @c Field is the (vector) field type of the distributions / moments.
     */
    template <class Field, class... Blocks>
    LBMScheme<Field, Blocks...> make_lbm_scheme(const std::string& name, double lambda, Blocks... blocks)
    {
        return LBMScheme<Field, Blocks...>(name, lambda, std::move(blocks)...);
    }
}
