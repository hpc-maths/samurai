// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <vector>

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
     *   BounceBack      : f_ghost(alpha) =  sign * f_inner(opposite[alpha])   (reflecting wall)
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
     * @par Multi-block reflecting (slip) wall
     *   For a multi-block scheme (D1Q222, D2Q4444, ... compressible Euler) the opposite velocity
     *   must be searched WITHIN each block, and a slip wall reverses the normal momentum: the block
     *   that carries the momentum component normal to the wall is reflected with @c sign = -1, all
     *   the others (density, energy, tangential momentum) with @c sign = +1. Pass the block sizes
     *   and, per block, the axis of the momentum it carries (or -1 for a scalar such as density or
     *   energy):
     *     samurai::make_bc<samurai::BounceBack>(f, velocities, block_sizes, block_odd_axis);
     *   With a single block and @c block_odd_axis = {-1} this is exactly the no-slip wall above
     *   (@c sign = +1 everywhere), so the single-argument overload is unchanged.
     */
    namespace detail
    {
        // opposite[a] = index b such that velocities[b] == -velocities[a], searched WITHIN the block
        // that contains a (blocks are the contiguous ranges given by block_sizes); b == a if none
        // (e.g. the rest velocity c == 0).
        template <std::size_t n_comp, std::size_t dim, class Vel>
        std::array<std::size_t, n_comp> lbm_opposite_velocities(const Vel& velocities, const std::vector<std::size_t>& block_sizes)
        {
            std::array<std::size_t, n_comp> opposite{};
            std::size_t offset = 0;
            for (const std::size_t q : block_sizes)
            {
                for (std::size_t a = offset; a < offset + q; ++a)
                {
                    opposite[a] = a;
                    for (std::size_t b = offset; b < offset + q; ++b)
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
                offset += q;
            }
            return opposite;
        }

        // Single-block search over the whole velocity list.
        template <std::size_t n_comp, std::size_t dim, class Vel>
        std::array<std::size_t, n_comp> lbm_opposite_velocities(const Vel& velocities)
        {
            return lbm_opposite_velocities<n_comp, dim>(velocities, std::vector<std::size_t>{n_comp});
        }

        // Expand a per-block reflection axis to a per-component one (-1 = even, no sign flip).
        template <std::size_t n_comp>
        std::array<int, n_comp> lbm_expand_odd_axis(const std::vector<std::size_t>& block_sizes, const std::vector<int>& block_odd_axis)
        {
            std::array<int, n_comp> odd_axis{};
            std::size_t offset = 0;
            for (std::size_t blk = 0; blk < block_sizes.size(); ++blk)
            {
                for (std::size_t k = 0; k < block_sizes[blk]; ++k)
                {
                    odd_axis[offset + k] = block_odd_axis[blk];
                }
                offset += block_sizes[blk];
            }
            return odd_axis;
        }
    }

    template <class Field>
    struct BounceBackImpl : public Bc<Field>
    {
        INIT_BC(BounceBackImpl, 2) // stencil [inner, ghost]

        static constexpr std::size_t n_comp = Field::n_comp;

        std::array<std::size_t, n_comp> m_opposite{};
        std::array<int, n_comp> m_odd_axis{}; // axis about which a component is odd (-1 = even)

        // Single-block no-slip wall: opposite over the whole velocity list, no sign flip.
        template <class Vel>
        BounceBackImpl(const typename base_t::lca_t& domain, const BcValue<Field>& bcv, const Vel& velocities)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities))
        {
            m_odd_axis.fill(-1);
        }

        // Multi-block reflecting (slip) wall: opposite within each block; the block carrying the
        // momentum normal to the wall is reflected with sign -1 (see @ref BounceBack).
        template <class Vel>
        BounceBackImpl(const typename base_t::lca_t& domain,
                       const BcValue<Field>& bcv,
                       const Vel& velocities,
                       const std::vector<std::size_t>& block_sizes,
                       const std::vector<int>& block_odd_axis)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities, block_sizes))
            , m_odd_axis(detail::lbm_expand_odd_axis<n_comp>(block_sizes, block_odd_axis))
        {
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t& direction) const override
        {
            // The reflection axis of the wall is the axis of the (axis-aligned) boundary direction.
            int wall_axis = -1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (direction(d) != 0)
                {
                    wall_axis = static_cast<int>(d);
                }
            }
            // cppcheck-suppress constParameterReference // f is written through f[cells[1]](a)
            return [opposite = m_opposite, odd = m_odd_axis, wall_axis](Field& f, const stencil_cells_t& cells, const value_t&)
            {
                // [0] = inner cell, [1] = outer ghost
                for (std::size_t a = 0; a < n_comp; ++a)
                {
                    const double sign = (odd[a] == wall_axis) ? -1. : 1.;
                    f[cells[1]](a)    = sign * f[cells[0]](opposite[a]);
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

    /**
     * Imposed-distribution inflow: the outer ghost holds a fixed distribution (typically the
     * free-stream equilibrium @c LBMScheme::equilibrium_f({rho, rho u, rho v, ...})), so streaming
     * pulls that distribution into the domain. This is the LBM counterpart of a Dirichlet inflow;
     * combine it with a homogeneous @c Neumann outflow on the opposite side.
     *
     *   samurai::make_bc<samurai::ImposedDistribution>(f, f_in)->on(left, top, bottom);
     */
    template <class Field>
    struct ImposedDistributionImpl : public Bc<Field>
    {
        INIT_BC(ImposedDistributionImpl, 2) // stencil [inner, ghost]

        static constexpr std::size_t n_comp = Field::n_comp;

        std::array<double, n_comp> m_value{}; // distribution imposed in the ghost

        template <class Dist>
        ImposedDistributionImpl(const typename base_t::lca_t& domain, const BcValue<Field>& bcv, const Dist& value)
            : base_t(domain, bcv)
        {
            for (std::size_t a = 0; a < n_comp; ++a)
            {
                m_value[a] = static_cast<double>(value[a]);
            }
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            // cppcheck-suppress constParameterReference // f is written through f[cells[1]](a)
            return [value = m_value](Field& f, const stencil_cells_t& cells, const value_t&)
            {
                for (std::size_t a = 0; a < n_comp; ++a)
                {
                    f[cells[1]](a) = value[a];
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

    struct ImposedDistribution
    {
        using lbm_bc_tag = void;

        template <class Field>
        using impl_t = ImposedDistributionImpl<Field>;
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
     * make_bc for the LBM multi-block reflecting (slip) wall: the lattice velocities, the block
     * sizes (q per block, summing to n_comp) and, per block, the axis of the momentum it carries
     * (or -1 for a scalar block such as density / energy). See @ref BounceBack.
     */
    template <class bc_type, class Field, class Vel>
        requires requires { typename bc_type::lbm_bc_tag; }
    auto make_bc(Field& field, const Vel& velocities, const std::vector<std::size_t>& block_sizes, const std::vector<int>& block_odd_axis)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;
        auto& mesh    = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(), velocities, block_sizes, block_odd_axis));
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
