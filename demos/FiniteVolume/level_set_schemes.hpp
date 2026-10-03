// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>

#include <samurai/schemes/fv.hpp>

/**
 * Transport of the level set phi by the velocity field u, discretized by the
 * limited Lax-Wendroff flux of LeVeque (2002, equation 6.32) with the
 * monotonized central (MC) limiter of van Leer (1977).
 *
 * The flux depends on the time step: dt is captured by reference, so that the
 * scheme always reads its current value.
 */
template <class Field, class VelocityField>
    requires samurai::field_like<VelocityField>
auto make_level_set_convection(VelocityField& u, const double& dt)
{
    static_assert(Field::is_scalar, "The level set must be a scalar field");
    static_assert(VelocityField::dim == Field::dim && VelocityField::n_comp == Field::dim);

    static constexpr std::size_t dim = Field::dim;

    // Default stencil in the direction d: {-1, 0, 1, 2}, the face lies between the cells 1 and 2
    static constexpr std::size_t stencil_size = 4;
    using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field, VelocityField>;

    samurai::FluxDefinition<cfg> lax_wendroff;

    samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
        [&](auto integral_constant_d)
        {
            static constexpr std::size_t d = decltype(integral_constant_d)::value;

            lax_wendroff[d].cons_flux_function =
                [&u, &dt](samurai::FluxValue<cfg>& flux, const samurai::StencilData<cfg>& data, const samurai::StencilValues<cfg>& phi)
            {
                const auto& cells = data.cells;

                // Velocity at the face, interpolated from the four cells of the stencil
                const double v = (-u[cells[0]](d) + 9. * u[cells[1]](d) + 9. * u[cells[2]](d) - u[cells[3]](d)) / 16.;

                // theta: ratio of the jump at the upwind face to the jump at this face
                double jump = phi[2] - phi[1];
                if (std::abs(jump) < 1e-8)
                {
                    jump = 1e-8;
                }
                const double theta = ((v >= 0) ? phi[1] - phi[0] : phi[3] - phi[2]) / jump;

                const double mc      = std::max(0., std::min({2. * theta, .5 * (1. + theta), 2.}));
                const double courant = std::abs(v) * dt / data.cell_length;

                flux = std::max(v, 0.) * phi[1] + std::min(v, 0.) * phi[2]         // upwind flux
                     + .5 * mc * std::abs(v) * (1. - courant) * (phi[2] - phi[1]); // limited correction
            };
        });

    auto scheme = samurai::make_flux_based_scheme(lax_wendroff);
    scheme.set_name("level-set convection");
    scheme.set_parameter_field(u);
    return scheme;
}

/**
 * Godunov Hamiltonian of the reinitialization equation
 *
 *     d(phi)/d(tau) + sign(phi_0) (|grad(phi)| - 1) = 0,
 *
 * with the second-order one-sided differences of Min and Gibou (2007).
 *
 * The scheme returns (dx / dx_min) H(phi), where dx is the cell length and
 * dx_min the length of the finest cells: with a fictitious time step
 * proportional to dx_min, each level then advances with a time step
 * proportional to its own cell length (local time stepping).
 */
template <class Field>
auto make_level_set_reinitialization(const Field& phi_0)
{
    static_assert(Field::is_scalar, "The level set must be a scalar field");

    static constexpr std::size_t dim = Field::dim;

    // Star stencil of radius 2:
    //     left2, left, center, right, right2, bottom2, bottom, top, top2 (, front2, front, back, back2)
    static constexpr std::size_t radius = 2;
    using cfg                           = samurai::StarStencilSchemeConfig<samurai::SchemeType::NonLinear, radius, Field, Field>;

    auto scheme = samurai::make_cell_based_scheme<cfg>();
    scheme.set_name("level-set reinitialization");
    scheme.set_scheme_function(
        [&phi_0](samurai::SchemeValue<cfg>& value, const samurai::StencilCells<cfg>& cells, const Field& phi)
        {
            static constexpr std::size_t center = cfg::center_index;

            auto minmod = [](double a, double b)
            {
                return (a * b > 0.) ? ((std::abs(a) < std::abs(b)) ? a : b) : 0.;
            };

            const bool positive = phi_0[cells[center]] >= 0.;
            const double h      = cells[center].length;
            const double phi_c  = phi[cells[center]];

            double norm2 = 0;
            for (std::size_t d = 0; d < dim; ++d)
            {
                // Indices in the star stencil of the cells at -2, -1, +1 and +2 in the direction d
                const std::size_t m2 = (d == 0) ? 0 : 4 * d + 1;
                const std::size_t m1 = m2 + 1;
                const std::size_t p1 = (d == 0) ? 3 : m2 + 2;
                const std::size_t p2 = p1 + 1;

                const double phi_m2 = phi[cells[m2]];
                const double phi_m1 = phi[cells[m1]];
                const double phi_p1 = phi[cells[p1]];
                const double phi_p2 = phi[cells[p2]];

                // One-sided differences, corrected by the minmod of the second differences
                const double dd       = (phi_p1 - 2. * phi_c + phi_m1) / (h * h);
                const double dd_left  = (phi_c - 2. * phi_m1 + phi_m2) / (h * h);
                const double dd_right = (phi_p2 - 2. * phi_p1 + phi_c) / (h * h);

                const double d_minus = (phi_c - phi_m1) / h + .5 * h * minmod(dd, dd_left);
                const double d_plus  = (phi_p1 - phi_c) / h - .5 * h * minmod(dd, dd_right);

                // Godunov upwinding
                if (positive)
                {
                    norm2 += std::max(std::pow(std::max(d_minus, 0.), 2.), std::pow(std::min(d_plus, 0.), 2.));
                }
                else
                {
                    norm2 += std::max(std::pow(std::min(d_minus, 0.), 2.), std::pow(std::max(d_plus, 0.), 2.));
                }
            }

            const double scale = h / phi.mesh().min_cell_length();
            value              = (positive ? 1. : -1.) * scale * (std::sqrt(norm2) - 1.);
        });
    return scheme;
}
