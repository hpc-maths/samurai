// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

#include "../../bc.hpp"

namespace samurai
{
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
     * Half-way reflection shared by the lattice Boltzmann wall boundary conditions @ref BounceBack
     * and @ref AntiBounceBack.
     *
     * It is attached to the distribution field @a f with @c make_bc, like the finite volume
     * boundary conditions, and @c update_ghost_mr applies it before the stream reads the ghosts.
     * The outer ghost cell holds the distribution of the inner cell with every velocity reversed
     * (c to -c), so that after the stream the incoming populations equal the reflected outgoing
     * ones. The reflection is a fixed permutation @c opposite[a], the index of the velocity -c_a
     * (or @c a itself when there is none, such as the rest velocity), independent of the boundary
     * direction. Bounce-back and anti-bounce-back differ only by the reflection sign s in
     *
     *     f_ghost(a) = s * f_inner(opposite[a]) + rhs(a)
     *
     * - s = +1 (@ref BounceBack) closes the odd part and imposes the odd moments, such as the
     *   momentum, leaving the even moments free: a no-slip wall.
     * - s = -1 (@ref AntiBounceBack) closes the even part and imposes the even moments, such as
     *   the density, the pressure, the water height or the temperature, leaving the odd moments
     *   free: a Dirichlet condition on that scalar.
     *
     * The @c bc_type tag sets the base sign, +1 or -1.
     *
     * Given a wall equilibrium distribution f^eq to reflect around, the right-hand side keeps the
     * parity that s imposes:
     *
     *     rhs(a) = f^eq(a) - s * f^eq(opposite[a])
     *
     * that is twice the even part of f^eq for anti-bounce-back (the imposed density or height,
     * with its kinetic energy) and twice its odd part for bounce-back (the momentum of a moving
     * wall, 2 w_a rho (c_a . u_wall) / c_s^2 for the usual equilibrium). Without a wall
     * equilibrium, rhs is zero: a wall at rest for bounce-back, a zero scalar for
     * anti-bounce-back. When f^eq is symmetric (fluid at rest), rhs is 2 f^eq(a) for
     * anti-bounce-back and 0 for bounce-back. Bounce-back and anti-bounce-back are the odd and
     * even link-wise closures of the same reflection, as in Ginzburg and d'Humieres, and in
     * Kruger et al.
     *
     * The wall equilibrium is either a constant distribution, such as
     * `LBMScheme::equilibrium_f({h_wall, 0, ...})` for a fluid at rest at the wall, or a callable
     * that takes the distribution of the inner cell and returns the equilibrium to reflect
     * around. The callable is evaluated at every step from the local flow, for example with
     * @c LBMScheme::moments, then @c LBMScheme::equilibrium_f with the imposed moment replaced. It
     * keeps the imposed even part consistent with the flow through the boundary, which keeps an
     * open "reservoir" boundary stable under a sustained current.
     *
     * Multi-block reflecting (slip) wall: for a scheme with several blocks (D1Q222, D2Q4444, and
     * the other compressible Euler schemes) the opposite velocity is searched within each block,
     * and a slip wall reverses the normal momentum. The block that carries the momentum component
     * normal to the wall is reflected with the sign flipped, the others (density, energy,
     * tangential momentum) with the base sign. Each block gives the axis of the momentum it
     * carries, or -1 for a scalar such as the density or the energy. With a single block and
     * @c block_odd_axis = {-1}, this is the no-slip wall above.
     *
     * In a multi-block scheme each conserved variable is the zeroth moment (the sum) of its own
     * block, so negating it flips the sign of the whole block. In a single-population fluid the
     * velocity is a first moment, which plain bounce-back already negates by reversing every
     * velocity. The two conventions look opposite but impose the same physics: zero normal
     * velocity at the wall.
     *
     * When the velocity set has a diagonal velocity, such as in D2Q9, the reflection also fills
     * the ghosts across the corners of the domain.
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
            return std::any_of(velocities.begin(),
                               velocities.end(),
                               [](const auto& c)
                               {
                                   return std::count_if(c.begin(),
                                                        c.end(),
                                                        [](int v)
                                                        {
                                                            return v != 0;
                                                        })
                                        > 1;
                               });
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
        // momentum normal to the wall is flipped (see the description of the class).
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
        // bounce-back (base_sign = +1). When f^eq is symmetric (fluid at rest), it is 2 f^eq_a for
        // anti-bounce-back and 0 for bounce-back.
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
     * Imposed-distribution inflow, selected by the @ref ImposedDistribution tag: the outer ghost
     * holds a fixed distribution, typically the free-stream equilibrium
     * `LBMScheme::equilibrium_f({rho, rho u, rho v, ...})`, so the stream pulls that distribution
     * into the domain. It is the lattice Boltzmann counterpart of a Dirichlet inflow; combine it
     * with a homogeneous @c Neumann outflow on the opposite side:
     *
     * @code
     * samurai::make_bc<samurai::ImposedDistribution>(f, f_in)->on(left, top, bottom);
     * @endcode
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

    // Tags selecting the implementation, like samurai::Dirichlet and samurai::Neumann. BounceBack
    // and AntiBounceBack map to the same LbmReflectionImpl, which reads the base reflection sign
    // from the tag type (+1 for BounceBack, -1 for AntiBounceBack).

    /**
     * Tag of the bounce-back wall boundary condition, passed to @c make_bc: imposes the odd
     * moments, such as the momentum (see @ref LbmReflectionImpl).
     */
    struct BounceBack
    {
        using lbm_bc_tag = void; // marks the LBM make_bc overloads below

        template <class Field>
        using impl_t = LbmReflectionImpl<Field, BounceBack>;
    };

    /**
     * Tag of the anti-bounce-back wall boundary condition, passed to @c make_bc: imposes the even
     * moments, such as the density, the pressure or the water height (see @ref LbmReflectionImpl).
     */
    struct AntiBounceBack
    {
        using lbm_bc_tag = void;

        template <class Field>
        using impl_t = LbmReflectionImpl<Field, AntiBounceBack>;
    };

    /**
     * Tag of the imposed-distribution inflow boundary condition, passed to @c make_bc with the
     * distribution to impose (see @ref ImposedDistributionImpl).
     */
    struct ImposedDistribution
    {
        using lbm_bc_tag = void;

        template <class Field>
        using impl_t = ImposedDistributionImpl<Field>;
    };

    /**
     * Attaches a lattice Boltzmann boundary condition without wall equilibrium to the
     * distribution field.
     *
     * With @ref BounceBack it is a no-slip wall, with @ref AntiBounceBack a zero even moment.
     * With @ref ImposedDistribution, the second argument is the distribution to impose instead of
     * the lattice velocities. The overload applies only to tags that declare @c lbm_bc_tag, so it
     * never competes with the @c make_bc overloads of the finite volume boundary conditions.
     *
     * @tparam bc_type @ref BounceBack, @ref AntiBounceBack or @ref ImposedDistribution
     * @param field the distribution field @a f
     * @param velocities the lattice velocities, the same list as in @c velocity_scheme
     * @return a pointer to the attached boundary condition, whose @c on(...) selects the boundaries
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
     * Attaches a lattice Boltzmann reflection with a wall equilibrium to the distribution field.
     *
     * @ref AntiBounceBack imposes the even moments of the wall equilibrium, such as the density,
     * the pressure or the water height; @ref BounceBack imposes its odd moments, for a moving
     * wall (see @ref LbmReflectionImpl).
     *
     * @tparam bc_type @ref BounceBack or @ref AntiBounceBack
     * @param field the distribution field @a f
     * @param velocities the lattice velocities, the same list as in @c velocity_scheme
     * @param f_wall the equilibrium to reflect around: either a constant distribution, such as
     *        `scheme.equilibrium_f({h_wall, 0, ...})` for a fluid at rest at the wall, or a
     *        callable that takes the distribution of the inner cell and returns the equilibrium,
     *        evaluated at every step from the local flow
     * @return a pointer to the attached boundary condition, whose @c on(...) selects the boundaries
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
     * Attaches a lattice Boltzmann reflection for a scheme with several blocks, such as a slip
     * wall, to the distribution field.
     *
     * The opposite velocity is searched within each block, and the sign of the block that
     * carries the momentum normal to the wall is flipped (see @ref LbmReflectionImpl).
     *
     * @tparam bc_type @ref BounceBack or @ref AntiBounceBack
     * @param field the distribution field @a f
     * @param velocities the lattice velocities of all blocks, the same lists as in
     *        @c velocity_scheme, concatenated
     * @param block_sizes the number of velocities q of each block; they sum to the number of
     *        components of @a field
     * @param block_odd_axis for each block, the axis of the momentum component it carries, or -1
     *        for a scalar such as the density or the energy
     * @return a pointer to the attached boundary condition, whose @c on(...) selects the boundaries
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
     * Attaches a lattice Boltzmann reflection with a wall equilibrium for a scheme with several
     * blocks to the distribution field.
     *
     * It combines the wall equilibrium of the overload with @a f_wall and the blocks of the
     * multi-block overload (see @ref LbmReflectionImpl).
     *
     * @tparam bc_type @ref BounceBack or @ref AntiBounceBack
     * @param field the distribution field @a f
     * @param velocities the lattice velocities of all blocks, concatenated
     * @param f_wall the equilibrium to reflect around, a constant distribution or a callable
     *        that takes the distribution of the inner cell
     * @param block_sizes the number of velocities q of each block
     * @param block_odd_axis for each block, the axis of the momentum component it carries, or -1
     *        for a scalar
     * @return a pointer to the attached boundary condition, whose @c on(...) selects the boundaries
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
