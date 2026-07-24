// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
         * Moments m = M.f from a full distribution vector (all blocks concatenated). Public so that a
         * velocity-consistent wall boundary condition can read the LOCAL flow state from the inner
         * cell distribution (e.g. anti-bounce-back imposing a height/pressure while letting the
         * momentum float, see @ref AntiBounceBack).
         */
        std::array<double, n_comp> moments(const std::array<double, n_comp>& fall) const
        {
            std::array<double, n_comp> mall{};
            for_each_block(
                [&](const auto& block, std::size_t offset)
                {
                    constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                    std::array<double, q> fblock;
                    for (std::size_t k = 0; k < q; ++k)
                    {
                        fblock[k] = fall[offset + k];
                    }
                    const auto mblock = matvec(block.M, fblock);
                    for (std::size_t k = 0; k < q; ++k)
                    {
                        mall[offset + k] = mblock[k];
                    }
                });
            return mall;
        }

        using source_t = std::function<void(std::span<double> m_all, double dt)>;

        /**
         * Register a body-force source term (e.g. gravity). It is applied once per time step,
         * after the MRT relaxation and before the moment-to-distribution transform, and receives
         * the full moment vector (writable, all blocks concatenated) and the time step @a dt.
         * Forward-Euler: the source usually adds @c dt * force to the conserved momenta / energy.
         * @a dt must then be passed to @c operator().
         */
        void set_source(source_t source)
        {
            m_source = std::move(source);
        }

        /**
         * One LBM time step. Updates both @a f (distributions) and @a m (moments). Wall boundary
         * conditions are the ones attached to @a f (see @ref BounceBack / @ref AntiBounceBack and
         * @c make_bc); they are applied by @c update_ghost_mr before the stream reads the ghosts.
         * @a dt is only used by a registered source term (see @ref set_source).
         */
        template <class MField>
        void operator()(field_t& f, MField& m, double dt = 0.) const
        {
            ScopedTimer timer_op(m_name + " operator");

            update_ghost_mr(f);

            // Reuse a worker field for the streamed distributions instead of reallocating it every
            // step; (re)bind it to f's mesh on the first call or after a mesh change, otherwise just
            // resize it to the current cell count. The stream overwrites every real cell, so its
            // previous content is irrelevant.
            if (!m_f_stream || &m_f_stream->mesh() != &f.mesh())
            {
                m_f_stream.emplace("lbm_f_stream", f.mesh());
            }
            else
            {
                m_f_stream->resize();
            }

            {
                ScopedTimer timer_stream("lbm stream");
                stream(f, *m_f_stream);
            }
            std::swap(f.array(), m_f_stream->array());
            {
                ScopedTimer timer_collide("lbm collide");
                collide(f, m, dt);
            }
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
                    // cppcheck-suppress variableScope // offset is accumulated across the fold expansion below
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

        // One tap of a stream stencil: read f_in at offset @c off and add it, weighted by @c w (the
        // 1/2^{j.dim} projection already folded in), into f_out.
        struct stencil_tap
        {
            std::array<int, dim> off;
            double w;
        };

        /**
         * Flattened stream stencil of one velocity at a level gap @a j: the combined slice prediction
         * map (@ref get_prediction), with the 1/2^{j.dim} projection weight @a inv_nc folded into
         * every coefficient. The streamed value of a coarse cell C is then simply
         *
         *     f_out(C) = sum_tap tap.w * f_in(C + tap.off).
         *
         * The stream is linear, so the projection (average) over the 2^{j.dim} fine sub-cells of their
         * reconstructed donor value is a single application of the summed prediction map of the donor
         * sub-cell slice [-c[d], 2^j - c[d]) (one interval per direction, the full column shifted by
         * -c). That map depends only on (j, c), not on the cell, so the stencil is built once per
         * (level, component) and reused over the whole level. It handles axial, diagonal and |c| > 1
         * velocities uniformly; at the finest level (j == 0) it is the plain shift by -c, and for the
         * rest velocity c == 0 it reduces to the identity (mean conservation).
         */
        template <std::size_t... D>
        static std::vector<stencil_tap>
        build_stencil(std::size_t level, std::size_t j, const velocity_t& c, double inv_nc, std::index_sequence<D...> /*dim_seq*/)
        {
            constexpr std::size_t radius = field_t::mesh_t::config_t::prediction_stencil_radius;
            const auto width             = static_cast<value_t>(std::size_t{1} << j); // 2^j sub-cells per direction
            const auto slice             = std::make_tuple(interval_t{-static_cast<value_t>(c[D]), width - static_cast<value_t>(c[D])}...);

            const auto& map = detail::get_prediction<radius, field_t>(level, j, slice);

            std::vector<stencil_tap> taps;
            taps.reserve(map.coeff.size());
            for (const auto& kv : map.coeff)
            {
                std::array<int, dim> off{};
                for (std::size_t d = 0; d < dim; ++d)
                {
                    off[d] = static_cast<int>(kv.first[d]);
                }
                taps.push_back({off, inv_nc * kv.second});
            }
            return taps;
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

        // stream: multi-level linear transport. For each level the per-component stencils are built
        // once (build_stencil; the underlying prediction maps are cached across steps), then applied
        // to every cell strip - reading f_in and writing f_out directly, with no per-strip prediction
        // lookup and no temporary.
        void stream(const field_t& f_in, field_t& f_out) const
        {
            using mesh_id_t     = typename field_t::mesh_t::mesh_id_t;
            constexpr auto tseq = std::make_index_sequence<dim - 1>{}; // transverse indices of the coarse cell
            constexpr auto dseq = std::make_index_sequence<dim>{};     // one sub-cell interval per direction
            const std::array<int, dim> no_shift{};

            auto& mesh                  = f_in.mesh();
            const std::size_t max_level = mesh.max_level(); // configured finest level of the hierarchy

            std::array<std::vector<stencil_tap>, n_comp> stencil;
            std::vector<double> res; // reused accumulation buffer, sized to the current strip

            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                const std::size_t j = max_level - level;
                const double inv_nc = 1. / static_cast<double>(std::size_t{1} << (j * dim)); // 1/2^{j.dim} projection weight

                for_each_block(
                    [&](const auto& block, std::size_t offset)
                    {
                        constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                        for (std::size_t a = 0; a < q; ++a)
                        {
                            stencil[offset + a] = build_stencil(level, j, block.velocities[a], inv_nc, dseq);
                        }
                    });

                for_each_interval(mesh[mesh_id_t::cells][level],
                                  [&](std::size_t lvl, const auto& i, const auto& index)
                                  {
                                      for (std::size_t comp = 0; comp < n_comp; ++comp)
                                      {
                                          const auto& taps = stencil[comp];
                                          auto out         = access(f_out, comp, lvl, i, index, no_shift, tseq);
                                          const std::size_t sz = static_cast<std::size_t>(out.size());

                                          // Accumulate the taps into a reused contiguous buffer with plain scalar
                                          // loops (no temporary allocation, no lazy-expression machinery), then write
                                          // the strided f_out strip once.
                                          res.resize(sz);
                                          {
                                              auto vin       = access(f_in, comp, lvl, i, index, taps[0].off, tseq);
                                              const double w = taps[0].w;
                                              for (std::size_t ic = 0; ic < sz; ++ic)
                                              {
                                                  res[ic] = w * vin(ic);
                                              }
                                          }
                                          for (std::size_t t = 1; t < taps.size(); ++t)
                                          {
                                              auto vin       = access(f_in, comp, lvl, i, index, taps[t].off, tseq);
                                              const double w = taps[t].w;
                                              for (std::size_t ic = 0; ic < sz; ++ic)
                                              {
                                                  res[ic] += w * vin(ic);
                                              }
                                          }
                                          for (std::size_t ic = 0; ic < sz; ++ic)
                                          {
                                              out(ic) = res[ic];
                                          }
                                      }
                                  });
            }
        }

        // collide: m = M.f (all blocks) ; equilibrium (sees all moments) ; relax (MRT) ;
        //          optional source (body force) ; f = M^{-1} m.
        template <class MField>
        void collide(field_t& f, MField& m, double dt) const
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
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          const std::size_t g = offset + k;
                                          mall[g] += block.s[k] * (meq_all[g] - mall[g]); // relax (MRT)
                                      }
                                  });

                              // Body-force source (e.g. gravity), applied after relaxation on the
                              // (conserved) moments, before rebuilding the distributions.
                              if (m_source)
                              {
                                  m_source(std::span<double>(mall.data(), n_comp), dt);
                              }

                              for_each_block(
                                  [&](const auto& block, std::size_t offset)
                                  {
                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                      std::array<double, q> mblock;
                                      for (std::size_t k = 0; k < q; ++k)
                                      {
                                          mblock[k] = mall[offset + k];
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
        std::tuple<Blocks...> m_blocks;
        source_t m_source;                         // optional body-force source term (see set_source)
        mutable std::optional<field_t> m_f_stream; // worker for the streamed distributions (reused across steps)
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
