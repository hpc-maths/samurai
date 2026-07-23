// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <span>
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

                              std::array<double, n_comp> mall{};
                              for (std::size_t k = 0; k < n_comp; ++k)
                              {
                                  mall[k] = mc(k);
                              }

                              const auto feq = equilibrium_f(mall);
                              for (std::size_t k = 0; k < n_comp; ++k)
                              {
                                  fc(k) = feq[k];
                              }
                          });
        }

        /**
         * Equilibrium distribution f^eq from a full moment vector: the conserved moments (s_k == 0)
         * are kept, the non-conserved ones are set to their equilibrium value, then f^eq = M^{-1} m.
         * Public so that a wall boundary condition (e.g. anti-bounce-back, see @ref AntiBounceBack)
         * can build the equilibrium distribution to impose at the wall.
         */
        std::array<double, n_comp> equilibrium_f(const std::array<double, n_comp>& mall) const
        {
            std::array<double, n_comp> meq_all{};
            for_each_block(
                [&](const auto& block, std::size_t offset)
                {
                    constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                    std::array<double, q> meq;
                    block.equilibrium(meq, std::span<const double>(mall.data(), n_comp));
                    for (std::size_t k = 0; k < q; ++k)
                    {
                        meq_all[offset + k] = meq[k];
                    }
                });

            std::array<double, n_comp> feq{};
            for_each_block(
                [&](const auto& block, std::size_t offset)
                {
                    constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                    std::array<double, q> mblock;
                    for (std::size_t k = 0; k < q; ++k)
                    {
                        mblock[k] = (block.s[k] != 0.) ? meq_all[offset + k] : mall[offset + k];
                    }
                    const auto fblock = matvec(block.invM, mblock);
                    for (std::size_t k = 0; k < q; ++k)
                    {
                        feq[offset + k] = fblock[k];
                    }
                });
            return feq;
        }

        /**
         * One LBM time step. Updates both @a f (distributions) and @a m (moments). Wall boundary
         * conditions are the ones attached to @a f (see @ref BounceBack / @ref AntiBounceBack and
         * @c make_bc); they are applied by @c update_ghost_mr before the stream reads the ghosts.
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

        using velocity_t = std::array<int, dim>;
        using interval_t = typename field_t::interval_t;
        using value_t    = typename interval_t::value_t;

        /**
         * Streamed value of one velocity at a coarse interval, using the library's cached portion().
         *
         * The streamed coarse value is the projection (average) over the coarse cell's 2^{j.dim}
         * fine sub-cells of their reconstructed donor value (the sub-cell shifted by -c at the
         * finest level):
         *
         *     f_out(C) = (1/2^{j.dim}) sum_{local in [0,2^j)^dim} reconstruct(f, C, local - c).
         *
         * Each term is one portion() call reconstructing sub-cell (local - c) of the coarse cell,
         * vectorised over the whole coarse interval; portion() caches the underlying prediction map
         * (reconstruction.hpp). The caller applies the 1/2^{j.dim} projection weight. This handles
         * axial, diagonal and |c| > 1 velocities uniformly; at the finest level (j == 0) the column
         * is the single sub-cell {-c}, i.e. the plain shift by -c.
         *
         * @c transverse_seq is 0..dim-2 (the coarse-cell transverse indices), @c dim_seq is
         * 0..dim-1 (used to decode the sub-cell offsets, one per direction).
         */
        template <std::size_t... T, std::size_t... D>
        static auto portion_column(const field_t& f,
                                   std::size_t level,
                                   std::size_t j,
                                   const auto& i,
                                   const auto& index,
                                   const velocity_t& c,
                                   std::index_sequence<T...> /*transverse_seq*/,
                                   std::index_sequence<D...> /*dim_seq*/)
        {
            const auto width     = std::size_t{1} << j;         // 2^j sub-cells per direction
            const std::size_t nc = std::size_t{1} << (j * dim); // 2^{j.dim} sub-cells in the column
            const auto i_tuple   = std::make_tuple(i, index[T]...);

            // Sub-cell offsets (donor = local - c) of the n-th fine sub-cell, as a value_t tuple.
            auto sub = [&](std::size_t n)
            {
                return std::make_tuple((static_cast<value_t>((n >> (D * j)) & (width - 1)) - static_cast<value_t>(c[D]))...);
            };

            // portion() reconstructs the whole distribution vector at once (its per-component
            // overload is unusable for vector fields); the caller keeps the component owning c.
            auto res = portion(f, level, j, i_tuple, sub(0));
            for (std::size_t n = 1; n < nc; ++n)
            {
                res += portion(f, level, j, i_tuple, sub(n));
            }
            return res;
        }

        // Read/write f(comp, level, i + off[0], index[.] + off[.+1]) unpacking the transverse dims.
        template <class F, std::size_t... K>
        static auto access(F&& f,
                           std::size_t comp,
                           std::size_t level,
                           const auto& i,
                           const auto& index,
                           const std::array<int, dim>& off,
                           std::index_sequence<K...>)
        {
            return f(comp, level, i + off[0], (index[K] + off[K + 1])...);
        }

        // stream: multi-level transport via the library's cached portion().
        void stream(const field_t& f_in, field_t& f_out) const
        {
            using mesh_id_t     = typename field_t::mesh_t::mesh_id_t;
            constexpr auto tseq = std::make_index_sequence<dim - 1>{}; // transverse indices of the coarse cell
            constexpr auto dseq = std::make_index_sequence<dim>{};     // one sub-cell range per direction
            const std::array<int, dim> no_shift{};

            auto& mesh                  = f_in.mesh();
            const std::size_t max_level = (m_max_level == 0) ? mesh.max_level() : m_max_level;

            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                const std::size_t j = max_level - level;
                const double inv_nc = 1. / static_cast<double>(std::size_t{1} << (j * dim)); // 1/2^{j.dim} projection weight

                for_each_interval(mesh[mesh_id_t::cells][level],
                                  [&](std::size_t lvl, const auto& i, const auto& index)
                                  {
                                      for_each_block(
                                          [&](const auto& block, std::size_t offset)
                                          {
                                              constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                              for (std::size_t a = 0; a < q; ++a)
                                              {
                                                  const std::size_t comp = offset + a;
                                                  auto res = portion_column(f_in, lvl, j, i, index, block.velocities[a], tseq, dseq);
                                                  access(f_out, comp, lvl, i, index, no_shift, tseq) = inv_nc
                                                                                                     * xt::view(res, xt::all(), comp);
                                              }
                                          });
                                  });
            }
        }

        // collide: m = M.f (all blocks) ; equilibrium (sees all moments) ; relax (MRT) ; f = M^{-1} m.
        template <class MField>
        void collide(field_t& f, MField& m) const
        {
            for_each_cell(f.mesh(),
                          [&](const auto& cell)
                          {
                              auto fc = f[cell];
                              auto mc = m[cell];

                              std::array<double, n_comp> mall{};
                              for_each_block(
                                  [&](const auto& block, std::size_t offset)
                                  {
                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                      std::array<double, q> fblock;
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          fblock[k] = fc(offset + k);
                                      }
                                      const auto mblock = matvec(block.M, fblock); // f2m
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          mall[offset + k] = mblock[k];
                                      }
                                  });

                              std::array<double, n_comp> meq_all{};
                              for_each_block(
                                  [&](const auto& block, std::size_t offset)
                                  {
                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                      std::array<double, q> meq;
                                      block.equilibrium(meq, std::span<const double>(mall.data(), n_comp));
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          meq_all[offset + k] = meq[k];
                                      }
                                  });

                              for_each_block(
                                  [&](const auto& block, std::size_t offset)
                                  {
                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                      std::array<double, q> mblock;
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          const std::size_t g = offset + k;
                                          mall[g] += block.s[k] * (meq_all[g] - mall[g]); // relax (MRT)
                                          mblock[k] = mall[g];
                                      }
                                      const auto fnew = matvec(block.invM, mblock); // m2f
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          fc(offset + k) = fnew[k];
                                      }
                                  });

                              for (std::size_t k = 0; k < n_comp; ++k)
                              {
                                  mc(k) = mall[k];
                              }
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
