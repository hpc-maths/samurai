// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "../../stencil.hpp" // DirectionVector

namespace samurai
{
    /**
     * Kind of lattice-Boltzmann wall boundary condition.
     *
     * Both are half-way schemes applied @e after streaming, as a local reflection at the
     * boundary cell C referencing the pre-stream distribution at that same cell (so they are
     * exact at any refinement level and never touch the MR ghost machinery):
     *
     *  - @c bounce_back      : f_alpha(C) <- f_alphabar(C)                (no-slip wall,
     *                          reverses the odd moments, e.g. zero normal velocity/momentum)
     *  - @c anti_bounce_back : f_alpha(C) <- -f_alphabar(C) + 2 f_alpha^eq(m_wall)
     *                          (Dirichlet on the even moment, e.g. imposed density / pressure /
     *                          water height at the wall)
     *
     * where alphabar is the velocity opposite to the (incoming) velocity alpha, i.e.
     * c_alphabar = -c_alpha, and "incoming" means c_alpha . n < 0 for the outward wall normal n.
     */
    enum class lbm_bc_type
    {
        bounce_back,
        anti_bounce_back
    };

    /**
     * No-slip (bounce-back) wall for an @ref LBMScheme.
     *
     * By default it applies to every non-periodic boundary of the domain. Restrict it to a
     * single boundary with @c on(normal), where @c normal is the outward Cartesian normal of
     * that boundary (e.g. @c {1} for the east/right wall, @c {-1} for the west/left wall,
     * @c {0,1} for the north wall, ...).
     */
    template <std::size_t dim>
    struct BounceBack
    {
        DirectionVector<dim> normal;
        bool all_boundaries = true;

        BounceBack()
        {
            normal.fill(0);
        }

        BounceBack& on(const DirectionVector<dim>& n)
        {
            normal         = n;
            all_boundaries = false;
            return *this;
        }
    };

    /**
     * Anti-bounce-back wall for an @ref LBMScheme, imposing a Dirichlet condition on the even
     * moment(s) at the wall.
     *
     * @c wall_moments is the target moment vector at the wall (size equal to the field n_comp).
     * The conserved entries fix the imposed physical value (density, pressure, water height, ...);
     * the non-conserved entries are recomputed to their equilibrium value by the scheme, so the
     * reflection is done around the equilibrium distribution f^eq(m_wall).
     *
     * As for @ref BounceBack, it applies to every non-periodic boundary by default, or to a single
     * boundary via @c on(normal).
     */
    template <std::size_t dim>
    struct AntiBounceBack
    {
        std::vector<double> wall_moments;
        DirectionVector<dim> normal;
        bool all_boundaries = true;

        explicit AntiBounceBack(std::vector<double> m)
            : wall_moments(std::move(m))
        {
            normal.fill(0);
        }

        AntiBounceBack& on(const DirectionVector<dim>& n)
        {
            normal         = n;
            all_boundaries = false;
            return *this;
        }
    };
}
