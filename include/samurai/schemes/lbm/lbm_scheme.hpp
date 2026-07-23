// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../algorithm.hpp"
#include "../../algorithm/update_ghost_mr.hpp"
#include "../../boundary.hpp"
#include "../../reconstruction.hpp"
#include "../../stencil.hpp"
#include "boundary.hpp"
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
            compute_opposite_velocities();
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
         * Attach a boundary condition (@ref BounceBack / @ref AntiBounceBack). It is applied
         * after streaming, on every non-periodic boundary by default (or on the single boundary
         * selected with @c bc.on(normal)).
         */
        void add_bc(const BounceBack<dim>& bc)
        {
            m_bcs.push_back({lbm_bc_type::bounce_back, bc.all_boundaries, bc.normal, std::array<double, n_comp>{}});
        }

        void add_bc(const AntiBounceBack<dim>& bc)
        {
            assert(bc.wall_moments.size() == n_comp && "AntiBounceBack: wall_moments must have n_comp entries");
            std::array<double, n_comp> mwall{};
            for (std::size_t k = 0; k < n_comp; ++k)
            {
                mwall[k] = bc.wall_moments[k];
            }
            // Reflect around the equilibrium distribution at the wall: add_k = 2 f_k^eq(m_wall).
            const auto feq = equilibrium_f(mwall);
            std::array<double, n_comp> add{};
            for (std::size_t k = 0; k < n_comp; ++k)
            {
                add[k] = 2. * feq[k];
            }
            m_bcs.push_back({lbm_bc_type::anti_bounce_back, bc.all_boundaries, bc.normal, add});
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
            apply_bc(f, f_stream); // reflect the incoming populations on the physical walls
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

        /**
         * Equilibrium distribution f^eq from a full moment vector: the conserved moments (s_k == 0)
         * are kept, the non-conserved ones are set to their equilibrium value, then f^eq = M^{-1} m.
         * Shared by init_equilibrium (per cell) and the anti-bounce-back reflection value.
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

        // For each (global) velocity component, the component of the opposite velocity (c -> -c)
        // within the same block. A velocity with no opposite in its block (e.g. the rest velocity
        // c == 0) maps to itself; it is never an incoming velocity so the value is unused.
        void compute_opposite_velocities()
        {
            for_each_block(
                [&](const auto& block, std::size_t offset)
                {
                    constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                    for (std::size_t a = 0; a < q; ++a)
                    {
                        std::size_t opp = offset + a;
                        for (std::size_t b = 0; b < q; ++b)
                        {
                            bool is_opposite = true;
                            for (std::size_t d = 0; d < dim; ++d)
                            {
                                is_opposite &= (block.velocities[b][d] == -block.velocities[a][d]);
                            }
                            if (is_opposite)
                            {
                                opp = offset + b;
                                break;
                            }
                        }
                        m_opposite[offset + a] = opp;
                    }
                });
        }

        using index_t          = default_config::value_t;
        using velocity_t       = std::array<int, dim>;
        using stream_stencil_t = prediction_map<dim, index_t>;

        static constexpr std::size_t pred_order = Field::mesh_t::config_t::prediction_stencil_radius;

        // Call prediction<pred_order>(j, idx[0], ..., idx[dim-1]).
        template <std::size_t... K>
        static const stream_stencil_t& prediction_nd(std::size_t j, const std::array<index_t, dim>& idx, std::index_sequence<K...>)
        {
            return prediction<pred_order>(j, idx[K]...);
        }

        /**
         * Build the stream stencil of a velocity c at level jump j = max_level - level.
         *
         * The streamed coarse value is the projection (average) over the coarse cell's 2^{j.dim}
         * fine sub-cells of their reconstructed donor value (the sub-cell shifted by -c at the
         * finest level):
         *
         *     f_out(C) = (1/2^{j.dim}) sum_{local in [0,2^j)^dim} reconstruct(f, local - c)
         *              = sum_{(o,w)} w * f_in(C + o).
         *
         * reconstruct() is the MR prediction (reconstruction.hpp), so this is a single linear map.
         * At the finest level (j == 0) it is the plain shift by -c. This "column" form handles
         * axial, diagonal and |c| > 1 velocities uniformly; it is cached per (j, c).
         */
        stream_stencil_t build_stream_stencil(std::size_t j, const velocity_t& c) const
        {
            constexpr auto seq = std::make_index_sequence<dim>{};
            stream_stencil_t stencil;

            const index_t width  = index_t{1} << j;             // 2^j sub-cells per direction
            const std::size_t nc = std::size_t{1} << (j * dim); // 2^{j.dim} sub-cells in the column

            for (std::size_t n = 0; n < nc; ++n)
            {
                std::array<index_t, dim> donor;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const auto local_d = static_cast<index_t>((n >> (d * j)) & static_cast<std::size_t>(width - 1));
                    donor[d]           = local_d - c[d]; // fine index that flows into this sub-cell
                }
                stencil += prediction_nd(j, donor, seq);
            }

            stencil *= 1. / static_cast<double>(nc); // projection (average) over the column
            stencil.remove_small_entries();
            return stencil;
        }

        const stream_stencil_t& stream_stencil(std::size_t j, const velocity_t& c) const
        {
            auto key = std::make_pair(j, c);
            auto it  = m_stencil_cache.find(key);
            if (it == m_stencil_cache.end())
            {
                it = m_stencil_cache.emplace(std::move(key), build_stream_stencil(j, c)).first;
            }
            return it->second;
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

        // stream: multi-level transport via the cached column stencils.
        void stream(const field_t& f_in, field_t& f_out) const
        {
            using mesh_id_t                        = typename field_t::mesh_t::mesh_id_t;
            constexpr auto tseq                    = std::make_index_sequence<dim - 1>{};
            const std::array<int, dim> no_shift{};

            auto& mesh                             = f_in.mesh();
            const std::size_t max_level            = (m_max_level == 0) ? mesh.max_level() : m_max_level;
            [[maybe_unused]] const int ghost_width = static_cast<int>(mesh.cfg().ghost_width());

            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                const std::size_t j = max_level - level;

                std::vector<std::pair<std::size_t, const stream_stencil_t*>> stencils;
                for_each_block(
                    [&](const auto& block, std::size_t offset)
                    {
                        constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                        for (std::size_t a = 0; a < q; ++a)
                        {
                            const auto& st = stream_stencil(j, block.velocities[a]);
                            for ([[maybe_unused]] const auto& [o, w] : st.coeff)
                            {
                                for ([[maybe_unused]] std::size_t d = 0; d < dim; ++d)
                                {
                                    assert(std::abs(o[d]) <= ghost_width
                                           && "stream stencil reaches beyond the ghost band; increase max_stencil_size");
                                }
                            }
                            stencils.emplace_back(offset + a, &st);
                        }
                    });

                for_each_interval(mesh[mesh_id_t::cells][level],
                                  [&](std::size_t lvl, const auto& i, const auto& index)
                                  {
                                      for (const auto& [comp, st] : stencils)
                                      {
                                          auto it  = st->coeff.begin();
                                          auto res = xt::eval(it->second * access(f_in, comp, lvl, i, index, it->first, tseq));
                                          for (++it; it != st->coeff.end(); ++it)
                                          {
                                              res += it->second * access(f_in, comp, lvl, i, index, it->first, tseq);
                                          }
                                          access(f_out, comp, lvl, i, index, no_shift, tseq) = res;
                                      }
                                  });
            }
        }

        /**
         * Apply the wall boundary conditions after streaming (half-way bounce-back /
         * anti-bounce-back). For a boundary with outward normal n, every incoming velocity
         * alpha (c_alpha . n < 0) at the boundary cell C is overwritten from the opposite
         * (outgoing) population at the same cell in the pre-stream field:
         *
         *     bounce_back      : f_out(alpha, C) =  f_in(alphabar, C)
         *     anti_bounce_back : f_out(alpha, C) = -f_in(alphabar, C) + 2 f_alpha^eq(m_wall)
         *
         * f_in is the pre-stream field (@a f), f_out the streamed field (@a f_stream); they are
         * distinct arrays so there is no aliasing. This is level-local and mass-conserving.
         */
        void apply_bc(const field_t& f_in, field_t& f_out) const
        {
            if (m_bcs.empty())
            {
                return;
            }

            constexpr auto tseq = std::make_index_sequence<dim - 1>{};
            const std::array<int, dim> no_shift{};
            auto& mesh = f_in.mesh();

            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                for (const auto& bc : m_bcs)
                {
                    auto apply_on_normal = [&](const DirectionVector<dim>& n)
                    {
                        // Cartesian axis of this boundary; skip periodic directions.
                        std::size_t axis = 0;
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            if (n[d] != 0)
                            {
                                axis = d;
                            }
                        }
                        if (mesh.is_periodic(axis))
                        {
                            return;
                        }

                        auto bdry = domain_boundary(mesh, level, n);
                        for_each_interval(bdry,
                                          [&](std::size_t lvl, const auto& i, const auto& index)
                                          {
                                              for_each_block(
                                                  [&](const auto& block, std::size_t offset)
                                                  {
                                                      constexpr std::size_t q = std::decay_t<decltype(block)>::q;
                                                      for (std::size_t a = 0; a < q; ++a)
                                                      {
                                                          int dot = 0;
                                                          for (std::size_t d = 0; d < dim; ++d)
                                                          {
                                                              dot += block.velocities[a][d] * n[d];
                                                          }
                                                          if (dot >= 0) // keep only incoming velocities (c.n < 0)
                                                          {
                                                              continue;
                                                          }
                                                          const std::size_t comp = offset + a;
                                                          const std::size_t opp  = m_opposite[comp];
                                                          if (bc.type == lbm_bc_type::bounce_back)
                                                          {
                                                              access(f_out, comp, lvl, i, index, no_shift, tseq)
                                                                  = access(f_in, opp, lvl, i, index, no_shift, tseq);
                                                          }
                                                          else
                                                          {
                                                              access(f_out, comp, lvl, i, index, no_shift, tseq)
                                                                  = xt::eval(bc.add[comp] - access(f_in, opp, lvl, i, index, no_shift, tseq));
                                                          }
                                                      }
                                                  });
                                          });
                    };

                    if (bc.all_boundaries)
                    {
                        for_each_cartesian_direction<dim>([&](const DirectionVector<dim>& n) { apply_on_normal(n); });
                    }
                    else
                    {
                        apply_on_normal(bc.normal);
                    }
                }
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

        // Internal record of an attached boundary condition (see add_bc / apply_bc).
        struct bc_entry_t
        {
            lbm_bc_type type;
            bool all_boundaries;
            DirectionVector<dim> normal;
            std::array<double, n_comp> add; // 2 f^eq(m_wall) for anti-bounce-back, 0 for bounce-back
        };

        std::string m_name;
        double m_lambda;
        std::size_t m_max_level = 0; // 0 = use the current mesh finest level (uniform case)
        std::tuple<Blocks...> m_blocks;
        mutable std::map<std::pair<std::size_t, velocity_t>, stream_stencil_t> m_stencil_cache; // keyed by (level jump j, velocity)
        std::vector<bc_entry_t> m_bcs;
        std::array<std::size_t, n_comp> m_opposite{}; // global opposite-velocity component index
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
