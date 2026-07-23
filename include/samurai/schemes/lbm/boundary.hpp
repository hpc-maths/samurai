// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>

#include "../../bc.hpp"

namespace samurai
{
    /**
     * Lattice-Boltzmann wall boundary conditions, attached to the distribution field @a f the
     * same way as the finite-volume boundary conditions (see @c make_bc), and applied by
     * @c update_ghost_mr before the stream reads the ghosts.
     *
     * Both are half-way schemes realised as a ghost fill: the outer ghost cell holds the inner
     * cell's distribution with every velocity reversed (c -> -c), so that after streaming the
     * incoming populations equal the reflected outgoing ones. The reflection is a fixed
     * permutation @c opposite[alpha] (the index of the velocity -c_alpha), independent of the
     * boundary direction:
     *
     *   BounceBack      : f_ghost(alpha) =  f_inner(opposite[alpha])          (no-slip wall)
     *   AntiBounceBack  : f_ghost(alpha) = -f_inner(opposite[alpha]) + 2 f_alpha^eq(m_wall)
     *                                                                          (imposed even moment)
     *
     * @c f_alpha^eq(m_wall) is the equilibrium distribution to impose at the wall, computed by the
     * caller with @c LBMScheme::equilibrium_f and passed to @c make_bc.
     *
     * Usage (velocities are the same list passed to @c velocity_scheme):
     *   samurai::make_bc<samurai::BounceBack>(f, velocities)->on(left, right);
     *   samurai::make_bc<samurai::AntiBounceBack>(f, velocities, f_wall)->on(right);
     *
     * @note The opposite velocity is searched over the whole velocity list, which is correct for a
     *       single-velocity-block scheme (D1Q3, D2Q9, ...). A multi-block scheme would need the
     *       search restricted to each block.
     */
    namespace detail
    {
        // opposite[a] = index b such that velocities[b] == -velocities[a] (b == a if none, e.g. c == 0).
        template <std::size_t n_comp, std::size_t dim, class Vel>
        std::array<std::size_t, n_comp> lbm_opposite_velocities(const Vel& velocities)
        {
            std::array<std::size_t, n_comp> opposite{};
            for (std::size_t a = 0; a < n_comp; ++a)
            {
                opposite[a] = a;
                for (std::size_t b = 0; b < n_comp; ++b)
                {
                    bool is_opposite = true;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        is_opposite = is_opposite && (velocities[b][d] == -velocities[a][d]);
                    }
                    if (is_opposite)
                    {
                        opposite[a] = b;
                        break;
                    }
                }
            }
            return opposite;
        }
    }

    template <class Field>
    struct BounceBackImpl : public Bc<Field>
    {
        INIT_BC(BounceBackImpl, 2) // stencil [inner, ghost]

        static constexpr std::size_t n_comp = Field::n_comp;

        std::array<std::size_t, n_comp> m_opposite{};

        template <class Vel>
        BounceBackImpl(const typename base_t::lca_t& domain, const BcValue<Field>& bcv, const Vel& velocities)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities))
        {
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            // cppcheck-suppress constParameterReference // f is written through f[cells[1]](a)
            return [opposite = m_opposite](Field& f, const stencil_cells_t& cells, const value_t&)
            {
                // [0] = inner cell, [1] = outer ghost
                for (std::size_t a = 0; a < n_comp; ++a)
                {
                    f[cells[1]](a) = f[cells[0]](opposite[a]);
                }
            };
        }
    };

    template <class Field>
    struct AntiBounceBackImpl : public Bc<Field>
    {
        INIT_BC(AntiBounceBackImpl, 2) // stencil [inner, ghost]

        static constexpr std::size_t n_comp = Field::n_comp;

        std::array<std::size_t, n_comp> m_opposite{};
        std::array<double, n_comp> m_add{}; // 2 f^eq(m_wall)

        template <class Vel, class Feq>
        AntiBounceBackImpl(const typename base_t::lca_t& domain, const BcValue<Field>& bcv, const Vel& velocities, const Feq& f_wall)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities))
        {
            for (std::size_t a = 0; a < n_comp; ++a)
            {
                m_add[a] = 2. * static_cast<double>(f_wall[a]);
            }
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            // cppcheck-suppress constParameterReference // f is written through f[cells[1]](a)
            return [opposite = m_opposite, add = m_add](Field& f, const stencil_cells_t& cells, const value_t&)
            {
                for (std::size_t a = 0; a < n_comp; ++a)
                {
                    f[cells[1]](a) = -f[cells[0]](opposite[a]) + add[a];
                }
            };
        }
    };

    // Tags selecting the implementation (mirrors samurai::Dirichlet / samurai::Neumann).
    struct BounceBack
    {
        using lbm_bc_tag = void; // marks the LBM make_bc overloads below

        template <class Field>
        using impl_t = BounceBackImpl<Field>;
    };

    struct AntiBounceBack
    {
        using lbm_bc_tag = void;

        template <class Field>
        using impl_t = AntiBounceBackImpl<Field>;
    };

    /**
     * make_bc for the LBM bounce-back wall: pass the lattice velocities (same list as the scheme).
     * Constrained to LBM boundary conditions (via @c lbm_bc_tag) so it never competes with the
     * generic finite-volume @c make_bc overloads.
     */
    template <class bc_type, class Field, class Vel>
        requires requires { typename bc_type::lbm_bc_tag; }
    auto make_bc(Field& field, const Vel& velocities)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;
        auto& mesh    = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(), velocities));
    }

    /**
     * make_bc for the LBM anti-bounce-back wall: the lattice velocities and the equilibrium
     * distribution to impose at the wall (e.g. @c scheme.equilibrium_f({h_wall, 0, ...})).
     */
    template <class bc_type, class Field, class Vel, class Feq>
        requires requires { typename bc_type::lbm_bc_tag; }
    auto make_bc(Field& field, const Vel& velocities, const Feq& f_wall)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;
        auto& mesh    = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(), velocities, f_wall));
    }
}
