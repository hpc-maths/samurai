// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>

namespace samurai
{
    /**
     * @class VelocityScheme
     *
     * Elementary velocity scheme (one "sub-scheme" in the pylbm sense): a group
     * of @a q lattice velocities that share the same moment matrices and
     * relaxation parameters. A full LBM scheme is a list of such blocks whose
     * moments are concatenated into a single field (see @ref LBMScheme).
     *
     * The number of velocities @a q and the dimension are compile-time; all data
     * are stored in @c std::array (element access is unrolled and cheap).
     *
     * Conventions:
     *  - velocities are integer lattice vectors c_alpha (unit shift at the finest level);
     *  - M maps distributions to moments (f -> m = M.f), invM the inverse;
     *  - s is the per-moment relaxation vector (s_k = 0 for a conserved moment);
     *  - equilibrium fills the q equilibrium moments meq of this block from the full
     *    moment vector m_all (all blocks concatenated), which allows inter-block
     *    coupling (e.g. the Euler equilibria).
     */
    template <std::size_t dim_, std::size_t q_>
    struct VelocityScheme
    {
        static constexpr std::size_t dim = dim_;
        static constexpr std::size_t q   = q_;

        using velocity_t    = std::array<int, dim>;
        using matrix_t      = std::array<std::array<double, q>, q>; // row-major
        using moments_t     = std::array<double, q>;
        using equilibrium_t = std::function<void(moments_t& meq, std::span<const double> m_all)>;

        std::array<velocity_t, q> velocities;
        matrix_t M;    // f -> m
        matrix_t invM; // m -> f
        moments_t s;   // relaxation
        equilibrium_t equilibrium;

        static constexpr std::size_t size()
        {
            return q;
        }
    };

    /**
     * Helper to build a @ref VelocityScheme. Template arguments (dim, q) are
     * usually deduced from the braced velocities/matrices at the call site,
     * e.g. @c velocity_scheme<1, 2>(...).
     */
    template <std::size_t dim, std::size_t q>
    VelocityScheme<dim, q> velocity_scheme(std::array<std::array<int, dim>, q> velocities,
                                           std::array<std::array<double, q>, q> M,
                                           std::array<std::array<double, q>, q> invM,
                                           std::array<double, q> s,
                                           typename VelocityScheme<dim, q>::equilibrium_t equilibrium)
    {
        return VelocityScheme<dim, q>{velocities, M, invM, s, std::move(equilibrium)};
    }
}
