// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

#include "../../bc.hpp"

namespace samurai
{
    /**
     * Lattice-Boltzmann wall boundary conditions, attached to the distribution field @a f the
     * same way as the finite-volume boundary conditions (see @c make_bc), and applied by
     * @c update_ghost_mr before the stream reads the ghosts.
     *
     * Both are half-way schemes realised as the SAME ghost fill: the outer ghost cell holds the
     * inner cell's distribution with every velocity reversed (c -> -c), so that after streaming the
     * incoming populations equal the reflected outgoing ones. The reflection is a fixed permutation
     * @c opposite[alpha] (the index of the velocity -c_alpha), independent of the boundary
     * direction. Bounce-back and anti-bounce-back differ ONLY by the reflection sign s in
     *
     *   f_ghost(alpha) = s * f_inner(opposite[alpha]) + rhs(alpha)
     *
     *   s = +1  (bounce-back)      closes the antisymmetric/odd part -> imposes the odd moments
     *                              (velocity / momentum), even moments left free: a no-slip wall.
     *                              rhs is the moving-wall momentum term
     *                              -2 w_alpha rho (c_alpha . u_wall) / c_s^2, and vanishes for a
     *                              wall at rest.
     *   s = -1  (anti-bounce-back) closes the symmetric/even part -> imposes the even moments
     *                              (density / pressure / height / temperature), odd moments left
     *                              free: a Dirichlet condition on that scalar.
     *
     * Given an equilibrium @c f^eq(m_wall) to reflect around, the rhs keeps exactly the parity that
     * @c s imposes:
     *
     *   rhs(alpha) = f^eq_alpha(m_wall) - s * f^eq_opposite(alpha)(m_wall)
     *
     * i.e. twice the EVEN part for anti-bounce-back (imposed density/height incl. its kinetic q^2/h
     * energy) and twice the ODD part for bounce-back (moving-wall momentum). It vanishes for a
     * homogeneous condition (wall at rest / zero scalar) and, when @c f^eq is symmetric (fluid at
     * rest), reduces to the familiar 2 f^eq_alpha. This matches Ginzburg & d'Humieres / Kruger,
     * where bounce-back and anti-bounce-back are the odd and even link-wise closures of the very
     * same reflection.
     *
     * @c m_wall is provided to @c make_bc either as a CONSTANT equilibrium distribution (fluid at
     * rest at the wall, e.g. @c LBMScheme::equilibrium_f({h_wall, 0, ...})) or as a CALLABLE
     * @c inner_f -> f^eq rebuilt every step from the LOCAL cell (@c LBMScheme::moments then
     * @c equilibrium_f with the imposed moment overridden). The velocity-consistent callable keeps
     * the imposed even part in step with the through-flow and is what makes an open "reservoir"
     * boundary stable under a sustained current.
     *
     * A single implementation @c LbmReflectionImpl realises the formula: the @c BounceBack /
     * @c AntiBounceBack tag fixes the base sign (+1 / -1), the optional per-block odd axes flip it
     * (slip wall, see below) and the optional @c m_wall sets the rhs. @c BounceBack therefore also
     * accepts an @c m_wall (moving wall) and @c AntiBounceBack a homogeneous form (zero scalar).
     *
     * Usage (velocities are the same list passed to @c velocity_scheme):
     *   samurai::make_bc<samurai::BounceBack>(f, velocities)->on(left, right);
     *   samurai::make_bc<samurai::AntiBounceBack>(f, velocities, f_wall)->on(right);         // constant
     *   samurai::make_bc<samurai::AntiBounceBack>(f, velocities, reservoir)->on(right);      // callable
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
     *
     *   Note on the sign vs the single-population picture above: here each conserved variable is the
     *   ZEROTH moment (the sum) of its own block, so negating it requires @c sign = -1. In the usual
     *   single-population fluid the velocity is instead a FIRST moment, which plain bounce-back
     *   (@c sign = +1, reversing every c) already negates; the two conventions therefore look
     *   opposite but impose the same physics (zero normal velocity at the wall).
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

    // The two reflection tags (defined below); forward-declared so LbmReflectionImpl can pick the
    // base sign from the tag type.
    struct BounceBack;
    struct AntiBounceBack;

    /**
     * Unified half-way reflection filling the outer ghost as
     *
     *   f_ghost(a) = sign(a) * f_inner(opposite[a]) + add(a)
     *
     * with sign(a) = base_sign * (odd-axis flip) and add(a) = 2 f^eq(m_wall) (0 when homogeneous).
     * The @c bc_type tag (@ref BounceBack / @ref AntiBounceBack) selects @c base_sign (+1 / -1);
     * bounce-back and anti-bounce-back are otherwise the same operation (see the file header).
     */
    template <class Field, class bc_type>
    struct LbmReflectionImpl : public Bc<Field>
    {
        INIT_BC(LbmReflectionImpl, 2) // stencil [inner, ghost]

        static constexpr std::size_t n_comp = Field::n_comp;
        static constexpr double base_sign   = std::is_same_v<bc_type, AntiBounceBack> ? -1. : 1.;

        using feq_fn = std::function<std::array<double, n_comp>(const std::array<double, n_comp>&)>;

        std::array<std::size_t, n_comp> m_opposite{};
        std::array<int, n_comp> m_odd_axis{}; // axis about which a component is odd (-1 = even, no flip)
        std::array<double, n_comp> m_add{};   // constant rhs (empty m_feq); all zero => homogeneous
        feq_fn m_feq{};                       // velocity-consistent rhs: inner distribution -> f^eq to reflect around

        // Set when the velocity set contains a diagonal velocity (more than one non-zero
        // component, e.g. D2Q9's {1,1} or D2Q4diag). Such a scheme streams across the domain
        // corners, so those ghosts must carry the reflection too: see fills_diagonal_directions().
        bool m_diagonal_velocities = false;

        bool fills_diagonal_directions() const override
        {
            return m_diagonal_velocities;
        }

      private:

        template <class Vel>
        static bool has_diagonal_velocity(const Vel& velocities)
        {
            for (const auto& c : velocities)
            {
                std::size_t nnz = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (c[d] != 0)
                    {
                        ++nnz;
                    }
                }
                if (nnz > 1)
                {
                    return true;
                }
            }
            return false;
        }

      public:

        // Single block, homogeneous (no-slip wall for BounceBack, zero even moment for AntiBounceBack).
        template <class Vel>
        LbmReflectionImpl(const typename base_t::lca_t& domain, const BcValue<Field>& bcv, const Vel& velocities)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities))
        {
            m_diagonal_velocities = has_diagonal_velocity(velocities);
            m_odd_axis.fill(-1);
            m_add.fill(0.);
        }

        // Single block with an imposed value: @a wall is either a constant equilibrium distribution
        // to reflect around (fluid at rest at the wall) or a callable inner_f -> f^eq computing it
        // from the LOCAL flow (velocity-consistent, e.g. impose a height while the momentum floats).
        template <class Vel, class Feq>
        LbmReflectionImpl(const typename base_t::lca_t& domain, const BcValue<Field>& bcv, const Vel& velocities, const Feq& wall)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities))
        {
            m_diagonal_velocities = has_diagonal_velocity(velocities);
            m_odd_axis.fill(-1);
            set_wall(wall);
        }

        // Multi-block reflecting (slip) wall: opposite within each block; the block carrying the
        // momentum normal to the wall is flipped (see the file header).
        template <class Vel>
        LbmReflectionImpl(const typename base_t::lca_t& domain,
                          const BcValue<Field>& bcv,
                          const Vel& velocities,
                          const std::vector<std::size_t>& block_sizes,
                          const std::vector<int>& block_odd_axis)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities, block_sizes))
            , m_odd_axis(detail::lbm_expand_odd_axis<n_comp>(block_sizes, block_odd_axis))
        {
            m_diagonal_velocities = has_diagonal_velocity(velocities);
            m_add.fill(0.);
        }

        // Multi-block with an imposed value (constant distribution or velocity-consistent callable,
        // see the single-block overload above).
        template <class Vel, class Feq>
        LbmReflectionImpl(const typename base_t::lca_t& domain,
                          const BcValue<Field>& bcv,
                          const Vel& velocities,
                          const Feq& wall,
                          const std::vector<std::size_t>& block_sizes,
                          const std::vector<int>& block_odd_axis)
            : base_t(domain, bcv)
            , m_opposite(detail::lbm_opposite_velocities<n_comp, dim>(velocities, block_sizes))
            , m_odd_axis(detail::lbm_expand_odd_axis<n_comp>(block_sizes, block_odd_axis))
        {
            m_diagonal_velocities = has_diagonal_velocity(velocities);
            set_wall(wall);
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
            return [opposite = m_opposite, odd = m_odd_axis, add = m_add, feq = m_feq, wall_axis](Field& f,
                                                                                                  const stencil_cells_t& cells,
                                                                                                  const value_t&)
            {
                // [0] = inner cell, [1] = outer ghost
                std::array<double, n_comp> rhs = add;
                if (feq)
                {
                    // Velocity-consistent rhs: reflect around f^eq(m_wall) built from the LOCAL flow.
                    std::array<double, n_comp> fin{};
                    for (std::size_t a = 0; a < n_comp; ++a)
                    {
                        fin[a] = f[cells[0]](a);
                    }
                    rhs = symmetrise(feq(fin), opposite);
                }
                for (std::size_t a = 0; a < n_comp; ++a)
                {
                    const double sign = base_sign * ((odd[a] == wall_axis) ? -1. : 1.);
                    f[cells[1]](a)    = sign * f[cells[0]](opposite[a]) + rhs[a];
                }
            };
        }

      private:

        // rhs from the equilibrium to reflect around: add(a) = f^eq_a - base_sign f^eq_opposite(a),
        // i.e. twice the EVEN part for anti-bounce-back (base_sign = -1) and twice the ODD part for
        // bounce-back (base_sign = +1). Reduces to 2 f^eq_a when f^eq is symmetric (fluid at rest).
        static std::array<double, n_comp> symmetrise(const std::array<double, n_comp>& feq, const std::array<std::size_t, n_comp>& opposite)
        {
            std::array<double, n_comp> add{};
            for (std::size_t a = 0; a < n_comp; ++a)
            {
                add[a] = feq[a] - base_sign * feq[opposite[a]];
            }
            return add;
        }

        // Constant wall (an equilibrium distribution) or a velocity-consistent callable inner_f -> f^eq.
        template <class Feq>
        void set_wall(const Feq& wall)
        {
            if constexpr (std::is_invocable_v<Feq, const std::array<double, n_comp>&>)
            {
                m_feq = wall;
            }
            else
            {
                std::array<double, n_comp> f_wall{};
                for (std::size_t a = 0; a < n_comp; ++a)
                {
                    f_wall[a] = static_cast<double>(wall[a]);
                }
                m_add = symmetrise(f_wall, m_opposite);
            }
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

    // Tags selecting the implementation (mirrors samurai::Dirichlet / samurai::Neumann). Both map
    // to the same LbmReflectionImpl, which reads the base reflection sign from the tag type (+1 for
    // BounceBack, -1 for AntiBounceBack; see the file header).
    struct BounceBack
    {
        using lbm_bc_tag = void; // marks the LBM make_bc overloads below

        template <class Field>
        using impl_t = LbmReflectionImpl<Field, BounceBack>;
    };

    struct AntiBounceBack
    {
        using lbm_bc_tag = void;

        template <class Field>
        using impl_t = LbmReflectionImpl<Field, AntiBounceBack>;
    };

    struct ImposedDistribution
    {
        using lbm_bc_tag = void;

        template <class Field>
        using impl_t = ImposedDistributionImpl<Field>;
    };

    /**
     * make_bc for a homogeneous LBM reflection (@ref BounceBack no-slip wall, @ref AntiBounceBack
     * zero even moment): pass the lattice velocities (same list as the scheme). Constrained to LBM
     * boundary conditions (via @c lbm_bc_tag) so it never competes with the generic finite-volume
     * @c make_bc overloads.
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
     * make_bc for an LBM reflection with an imposed value: the lattice velocities and @c wall, the
     * equilibrium to reflect around. @c wall is either a CONSTANT distribution (fluid at rest at the
     * wall, e.g. @c scheme.equilibrium_f({h_wall, 0, ...})) or a CALLABLE @c inner_f -> f^eq rebuilt
     * from the local flow (velocity-consistent, see the file header). @ref AntiBounceBack imposes the
     * even moment (density / pressure / height), @ref BounceBack the odd one (a moving wall).
     */
    template <class bc_type, class Field, class Vel, class Feq>
        requires requires { typename bc_type::lbm_bc_tag; }
    auto make_bc(Field& field, const Vel& velocities, const Feq& f_wall)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;
        auto& mesh    = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(), velocities, f_wall));
    }

    /**
     * make_bc for the multi-block LBM reflecting (slip) wall: the lattice velocities, the block
     * sizes (q per block, summing to n_comp) and, per block, the axis of the momentum it carries
     * (or -1 for a scalar block such as density / energy). See the file header.
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
     * make_bc for the multi-block LBM reflection with an imposed value: as the multi-block slip wall
     * above, plus the equilibrium distribution @c f_wall to impose.
     */
    template <class bc_type, class Field, class Vel, class Feq>
        requires requires { typename bc_type::lbm_bc_tag; }
    auto make_bc(Field& field,
                 const Vel& velocities,
                 const Feq& f_wall,
                 const std::vector<std::size_t>& block_sizes,
                 const std::vector<int>& block_odd_axis)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;
        auto& mesh    = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(), velocities, f_wall, block_sizes, block_odd_axis));
    }
}
