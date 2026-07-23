// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// Tests for the schemes/lbm formalism: the multi-level column-form stream (axial, diagonal
// and |c| > 1 velocities), multi-block schemes, the D2Q9 MRT Navier-Stokes physics, and the
// bounce-back / anti-bounce-back wall boundary conditions. These mirror the demos/LBM
// validation cases at small sizes so they run quickly inside the CI test binary.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <span>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/lbm.hpp>

namespace samurai
{
    namespace
    {
        constexpr double pi = 3.14159265358979323846;

        // Wrap a coordinate back into [left, left + L) for periodic exact solutions.
        double wrap(double z, double left, double L)
        {
            return left + std::fmod(std::fmod(z - left, L) + L, L);
        }

        // ------------------------------------------------------------------ D1Q2 advection
        struct advection_result
        {
            double rel_l2_error;
            double mass_drift;
            double umin;
            double umax;
        };

        advection_result run_d1q2_advection(std::size_t max_level, double Tf, double a, bool burgers = false)
        {
            static constexpr std::size_t dim = 1;
            const double lambda              = burgers ? 2. : 1.;
            const double s1                  = 1.5;
            const double left = -1., right = 1., L = right - left;

            Box<double, dim> box({left}, {right});
            auto cfg  = mesh_config<dim>().min_level(max_level).max_level(max_level).periodic(true).max_stencil_size(4);
            auto mesh = mra::make_mesh(box, cfg);

            auto u0 = [&](double x)
            {
                return std::sin(pi * x);
            };

            auto m = make_vector_field<double, 2>("m", mesh);
            auto f = make_vector_field<double, 2>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              m[cell](0) = u0(cell.center(0));
                          });

            std::array<std::array<double, 2>, 2> M{
                {{1., 1.}, {lambda, -lambda}}
            };
            std::array<std::array<double, 2>, 2> invM{
                {{0.5, 0.5 / lambda}, {0.5, -0.5 / lambda}}
            };
            auto eq = [a, burgers](std::array<double, 2>& meq, std::span<const double> mm)
            {
                meq[0] = mm[0];
                meq[1] = burgers ? 0.5 * mm[0] * mm[0] : a * mm[0];
            };

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D1Q2",
                                                   lambda,
                                                   velocity_scheme<dim, 2>(
                                                       {
                                                           {{1}, {-1}}
            },
                                                       M,
                                                       invM,
                                                       {0., s1},
                                                       eq));
            scheme.init_equilibrium(f, m);

            const double dx = L / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));
            const double Te = static_cast<double>(nt) * dt;

            auto mass = [&]()
            {
                double s = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  s += (f[cell](0) + f[cell](1)) * cell.length;
                              });
                return s;
            };
            const double mass0 = mass();

            for (std::size_t n = 0; n < nt; ++n)
            {
                scheme(f, m);
            }

            advection_result r{0., std::abs(mass() - mass0), std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
            double err = 0., nrm = 0.;
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double x  = cell.center(0);
                              const double ue = burgers ? 0. : u0(wrap(x - a * Te, left, L));
                              const double d  = m[cell](0) - ue;
                              err += d * d * cell.length;
                              nrm += ue * ue * cell.length;
                              r.umin = std::min(r.umin, m[cell](0));
                              r.umax = std::max(r.umax, m[cell](0));
                          });
            r.rel_l2_error = burgers ? 0. : std::sqrt(err / nrm);
            return r;
        }

        // ------------------------------------------------------------------ D2Q4 advection
        // diagonal == false: axial velocities {(1,0),(0,1),(-1,0),(0,-1)}
        // diagonal == true : rotated velocities {(1,1),(-1,1),(-1,-1),(1,-1)}
        advection_result run_d2q4_advection(std::size_t max_level, double Tf, double ax, double ay, bool diagonal, bool adapt = false)
        {
            static constexpr std::size_t dim = 2;
            const double lambda = 1., l2 = 1., s1 = 1.5, s2 = 1.;
            const double left = -1., right = 1., L = right - left;

            Box<double, dim> box({left, left}, {right, right});
            const std::size_t ml = adapt ? 2 : max_level;
            auto cfg             = mesh_config<dim>().min_level(ml).max_level(max_level).periodic(true).max_stencil_size(4);
            auto mesh            = mra::make_mesh(box, cfg);

            auto u0 = [&](double x, double y)
            {
                return std::sin(pi * x) * std::sin(pi * y);
            };

            auto m = make_vector_field<double, 4>("m", mesh);
            auto f = make_vector_field<double, 4>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              m[cell](0) = u0(cell.center(0), cell.center(1));
                          });

            std::array<std::array<int, 2>, 4> vel = diagonal ? std::array<std::array<int, 2>, 4>{{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}}
                                                             : std::array<std::array<int, 2>, 4>{{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
            std::array<std::array<double, 4>, 4> M, invM;
            if (diagonal)
            {
                // Velocities {(1,1),(-1,1),(-1,-1),(1,-1)}; moments (u, x-flux, y-flux, cross).
                M = {
                    {{1., 1., 1., 1.}, {1., -1., -1., 1.}, {1., 1., -1., -1.}, {1., -1., 1., -1.}}
                };
                invM = {
                    {{0.25, 0.25, 0.25, 0.25}, {0.25, -0.25, 0.25, -0.25}, {0.25, -0.25, -0.25, 0.25}, {0.25, 0.25, -0.25, -0.25}}
                };
            }
            else
            {
                M = {
                    {{1., 1., 1., 1.}, {lambda, 0., -lambda, 0.}, {0., lambda, 0., -lambda}, {l2, -l2, l2, -l2}}
                };
                invM = {
                    {{0.25, 0.5, 0., 0.25}, {0.25, 0., 0.5, -0.25}, {0.25, -0.5, 0., 0.25}, {0.25, 0., -0.5, -0.25}}
                };
            }
            auto eq = [ax, ay](std::array<double, 4>& meq, std::span<const double> mm)
            {
                meq[0] = mm[0];
                meq[1] = ax * mm[0];
                meq[2] = ay * mm[0];
                meq[3] = 0.;
            };

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D2Q4", lambda, velocity_scheme<dim, 4>(vel, M, invM, {0., s1, s1, s2}, eq));
            scheme.init_equilibrium(f, m);

            const double dx = L / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));
            const double Te = static_cast<double>(nt) * dt;

            auto MRadaptation = make_MRAdapt(f);
            auto mra_config   = samurai::mra_config().epsilon(1e-4);
            if (adapt)
            {
                MRadaptation(mra_config);
                m.resize();
            }

            auto mass = [&]()
            {
                double s = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  const double area = cell.length * cell.length;
                                  s += (f[cell](0) + f[cell](1) + f[cell](2) + f[cell](3)) * area;
                              });
                return s;
            };
            const double mass0 = mass();

            for (std::size_t n = 0; n < nt; ++n)
            {
                if (adapt)
                {
                    MRadaptation(mra_config);
                    m.resize();
                }
                scheme(f, m);
            }

            advection_result r{0., std::abs(mass() - mass0), std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
            double err = 0., nrm = 0.;
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double area = cell.length * cell.length;
                              const double ue   = u0(wrap(cell.center(0) - ax * Te, left, L), wrap(cell.center(1) - ay * Te, left, L));
                              const double d    = m[cell](0) - ue;
                              err += d * d * area;
                              nrm += ue * ue * area;
                              r.umin = std::min(r.umin, m[cell](0));
                              r.umax = std::max(r.umax, m[cell](0));
                          });
            r.rel_l2_error = std::sqrt(err / nrm);
            return r;
        }
    }

    // ==================================================================== D1Q2 (single block)
    TEST(lbm_d1q2, advection_mass_bounded_and_convergent)
    {
        const double a = 0.75;
        auto r6        = run_d1q2_advection(6, 0.4, a);
        auto r7        = run_d1q2_advection(7, 0.4, a);

        // Mass conserved to round-off on a periodic domain.
        EXPECT_LT(r6.mass_drift, 1e-11);
        EXPECT_LT(r7.mass_drift, 1e-11);
        // No overshoot beyond the advected sine.
        EXPECT_LE(r7.umax, 1.02);
        EXPECT_GE(r7.umin, -1.02);
        // Consistent (error decreases under refinement) and reasonably small.
        EXPECT_LT(r7.rel_l2_error, r6.rel_l2_error);
        EXPECT_LT(r7.rel_l2_error, 0.02);
    }

    TEST(lbm_d1q2, burgers_stays_bounded_through_the_shock)
    {
        auto r = run_d1q2_advection(7, 0.4, 0., /*burgers*/ true);
        EXPECT_LT(r.mass_drift, 1e-11);
        // Burgers of a sine initial datum stays within the initial bounds.
        EXPECT_LE(r.umax, 1.01);
        EXPECT_GE(r.umin, -1.01);
    }

    // ==================================================================== D2Q4 stream (N-D)
    TEST(lbm_d2q4, axial_advection_mass_bounded_and_convergent)
    {
        auto r5 = run_d2q4_advection(5, 0.4, 0.5, 0.5, /*diagonal*/ false);
        auto r6 = run_d2q4_advection(6, 0.4, 0.5, 0.5, /*diagonal*/ false);
        EXPECT_LT(r5.mass_drift, 1e-11);
        EXPECT_LT(r6.mass_drift, 1e-11);
        EXPECT_LT(r6.rel_l2_error, r5.rel_l2_error);
        EXPECT_LT(r6.rel_l2_error, 0.05);
    }

    TEST(lbm_d2q4, diagonal_advection_mass_bounded_and_convergent)
    {
        auto r5 = run_d2q4_advection(5, 0.4, 0.5, 0.5, /*diagonal*/ true);
        auto r6 = run_d2q4_advection(6, 0.4, 0.5, 0.5, /*diagonal*/ true);
        EXPECT_LT(r5.mass_drift, 1e-11);
        EXPECT_LT(r6.mass_drift, 1e-11);
        EXPECT_LT(r6.rel_l2_error, r5.rel_l2_error);
        EXPECT_LT(r6.rel_l2_error, 0.1);
    }

    TEST(lbm_d2q4, diagonal_advection_adaptive_conserves_mass)
    {
        // Adaptive run exercises |c| > 1 donor offsets at coarse level jumps.
        auto r = run_d2q4_advection(7, 0.4, 0.5, 0.5, /*diagonal*/ true, /*adapt*/ true);
        EXPECT_LT(r.mass_drift, 1e-11);
        EXPECT_LT(r.rel_l2_error, 0.1);
    }

    // ==================================================================== D1Q5 |c| > 1 stream
    namespace
    {
        struct range_result
        {
            double mass_drift;
            double vmin;
            double vmax;
            double pmin; // minimum pressure (Euler cases), otherwise unused
        };

        // D1Q5 shallow-water dam break (velocities {0, +1, -1, +2, -2}) on an open domain.
        range_result run_d1q5_dam(std::size_t max_level, double Tf, bool adapt, double eps = 1e-4)
        {
            static constexpr std::size_t dim = 1;
            const double lambda = 2., g = 1., s = 1., hL = 2., hR = 1.;
            const double l = lambda, l2 = l * l, l3 = l2 * l, l4 = l2 * l2;
            const double left = -1., right = 1., L = right - left;

            Box<double, dim> box({left}, {right});
            const std::size_t ml = adapt ? 3 : max_level;
            auto cfg             = mesh_config<dim>().min_level(ml).max_level(max_level).periodic(false).max_stencil_size(4);
            auto mesh            = mra::make_mesh(box, cfg);

            auto m = make_vector_field<double, 5>("m", mesh);
            auto f = make_vector_field<double, 5>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              m[cell](0) = (cell.center(0) < 0.) ? hL : hR;
                              m[cell](1) = 0.;
                          });

            std::array<std::array<double, 5>, 5> M{
                {{1., 1., 1., 1., 1.},
                 {0., l, -l, 2. * l, -2. * l},
                 {0., l2, l2, 4. * l2, 4. * l2},
                 {0., l3, -l3, 8. * l3, -8. * l3},
                 {0., l4, l4, 16. * l4, 16. * l4}}
            };
            std::array<std::array<double, 5>, 5> invM{
                {{1., 0., -5. / (4. * l2), 0., 1. / (4. * l4)},
                 {0., 2. / (3. * l), 2. / (3. * l2), -1. / (6. * l3), -1. / (6. * l4)},
                 {0., -2. / (3. * l), 2. / (3. * l2), 1. / (6. * l3), -1. / (6. * l4)},
                 {0., -1. / (12. * l), -1. / (24. * l2), 1. / (12. * l3), 1. / (24. * l4)},
                 {0., 1. / (12. * l), -1. / (24. * l2), -1. / (12. * l3), 1. / (24. * l4)}}
            };
            auto eq = [g, l2](std::array<double, 5>& meq, std::span<const double> mm)
            {
                const double h = mm[0], q = mm[1], flux = q * q / h + 0.5 * g * h * h;
                meq[0] = h;
                meq[1] = q;
                meq[2] = flux;
                meq[3] = q * l2;
                meq[4] = flux * l2;
            };

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D1Q5",
                                                   lambda,
                                                   velocity_scheme<dim, 5>(
                                                       {
                                                           {{0}, {1}, {-1}, {2}, {-2}}
            },
                                                       M,
                                                       invM,
                                                       {0., 0., s, s, s},
                                                       eq));
            make_bc<Neumann<1>>(f);
            scheme.init_equilibrium(f, m);

            const double dx = L / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));

            auto MRadaptation = make_MRAdapt(f);
            auto mra_config   = samurai::mra_config().epsilon(eps);
            if (adapt)
            {
                MRadaptation(mra_config);
                m.resize();
            }

            auto mass = [&]()
            {
                double s_ = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  s_ += m[cell](0) * cell.length;
                              });
                return s_;
            };
            const double mass0 = mass();

            for (std::size_t n = 0; n < nt; ++n)
            {
                if (adapt)
                {
                    MRadaptation(mra_config);
                    m.resize();
                }
                scheme(f, m);
            }

            range_result r{0., std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.};
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              r.vmin = std::min(r.vmin, m[cell](0));
                              r.vmax = std::max(r.vmax, m[cell](0));
                          });
            r.mass_drift = std::abs(mass() - mass0);
            return r;
        }
    }

    TEST(lbm_d1q5, dam_break_bounded_and_convergent)
    {
        // Waves have not reached the open ends yet, so the water mass is still conserved.
        auto r6 = run_d1q5_dam(6, 0.2, /*adapt*/ false);
        auto r7 = run_d1q5_dam(7, 0.2, /*adapt*/ false);
        EXPECT_LT(r6.mass_drift, 1e-9);
        EXPECT_LT(r7.mass_drift, 1e-9);
        // Height stays within the initial [hR, hL] = [1, 2] envelope.
        EXPECT_GE(r7.vmin, 1. - 1e-9);
        EXPECT_LE(r7.vmax, 2. + 1e-9);
    }

    TEST(lbm_d1q5, dam_break_adaptive_matches_uniform)
    {
        auto ru = run_d1q5_dam(8, 0.2, /*adapt*/ false);
        auto ra = run_d1q5_dam(8, 0.2, /*adapt*/ true, 1e-5);
        // The adaptive height envelope converges to the uniform one.
        EXPECT_NEAR(ra.vmin, ru.vmin, 5e-3);
        EXPECT_NEAR(ra.vmax, ru.vmax, 5e-3);
    }

    // ==================================================================== D1Q222 Euler (multi-block)
    namespace
    {
        // D1Q222 Euler Sod shock tube (three D1Q2 blocks) on an open domain.
        range_result run_d1q222_sod(std::size_t max_level, double Tf, bool adapt, double eps = 1e-4)
        {
            static constexpr std::size_t dim = 1;
            const double lambda = 3., gamma = 1.4, s_rel = 1.5;
            const double rhoL = 1., rhoR = 0.125, pL = 1., pR = 0.1;
            const double left = -1., right = 1., L = right - left;

            Box<double, dim> box({left}, {right});
            const std::size_t ml = adapt ? 3 : max_level;
            auto cfg             = mesh_config<dim>().min_level(ml).max_level(max_level).periodic(false).max_stencil_size(4);
            auto mesh            = mra::make_mesh(box, cfg);

            auto m = make_vector_field<double, 6>("m", mesh);
            auto f = make_vector_field<double, 6>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const bool lft = cell.center(0) < 0.;
                              m[cell](0)     = lft ? rhoL : rhoR;
                              m[cell](2)     = 0.;
                              m[cell](4)     = (lft ? pL : pR) / (gamma - 1.);
                          });

            const double l = lambda;
            std::array<std::array<double, 2>, 2> M{
                {{1., 1.}, {l, -l}}
            };
            std::array<std::array<double, 2>, 2> invM{
                {{0.5, 0.5 / l}, {0.5, -0.5 / l}}
            };
            const std::array<std::array<int, dim>, 2> vel{
                {{1}, {-1}}
            };
            auto eq_rho = [](std::array<double, 2>& meq, std::span<const double> mm)
            {
                meq[0] = mm[0];
                meq[1] = mm[2];
            };
            auto eq_q = [gamma](std::array<double, 2>& meq, std::span<const double> mm)
            {
                const double rho = mm[0], q = mm[2], E = mm[4];
                meq[0] = q;
                meq[1] = (3. - gamma) / 2. * q * q / rho + (gamma - 1.) * E;
            };
            auto eq_E = [gamma](std::array<double, 2>& meq, std::span<const double> mm)
            {
                const double rho = mm[0], q = mm[2], E = mm[4];
                meq[0] = E;
                meq[1] = gamma * q * E / rho + (1. - gamma) / 2. * q * q * q / (rho * rho);
            };

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D1Q222",
                                                   lambda,
                                                   velocity_scheme<dim, 2>(vel, M, invM, {0., s_rel}, eq_rho),
                                                   velocity_scheme<dim, 2>(vel, M, invM, {0., s_rel}, eq_q),
                                                   velocity_scheme<dim, 2>(vel, M, invM, {0., s_rel}, eq_E));
            make_bc<Neumann<1>>(f);
            scheme.init_equilibrium(f, m);

            const double dx = L / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));

            auto MRadaptation = make_MRAdapt(f);
            auto mra_config   = samurai::mra_config().epsilon(eps);
            if (adapt)
            {
                MRadaptation(mra_config);
                m.resize();
            }

            auto mass = [&]()
            {
                double s_ = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  s_ += m[cell](0) * cell.length;
                              });
                return s_;
            };
            const double mass0 = mass();

            for (std::size_t n = 0; n < nt; ++n)
            {
                if (adapt)
                {
                    MRadaptation(mra_config);
                    m.resize();
                }
                scheme(f, m);
            }

            range_result r{0., std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double rho = m[cell](0), q = m[cell](2), E = m[cell](4);
                              const double p = (gamma - 1.) * (E - 0.5 * q * q / rho);
                              r.vmin         = std::min(r.vmin, rho);
                              r.vmax         = std::max(r.vmax, rho);
                              r.pmin         = std::min(r.pmin, p);
                          });
            r.mass_drift = std::abs(mass() - mass0);
            return r;
        }
    }

    TEST(lbm_d1q222, sod_conserves_mass_and_stays_physical)
    {
        // Before any wave reaches the open ends, the mass is conserved to round-off.
        auto r = run_d1q222_sod(9, 0.2, /*adapt*/ false);
        EXPECT_LT(r.mass_drift, 1e-11);
        // Density and pressure stay within the initial [right, left] envelope, and positive.
        EXPECT_GE(r.vmin, 0.125 - 1e-9);
        EXPECT_LE(r.vmax, 1. + 1e-9);
        EXPECT_GT(r.pmin, 0.);
    }

    TEST(lbm_d1q222, sod_adaptive_matches_uniform)
    {
        auto ru = run_d1q222_sod(10, 0.2, /*adapt*/ false);
        auto ra = run_d1q222_sod(10, 0.2, /*adapt*/ true, 1e-5);
        EXPECT_NEAR(ra.vmin, ru.vmin, 5e-3);
        EXPECT_NEAR(ra.vmax, ru.vmax, 5e-3);
        EXPECT_GT(ra.pmin, 0.);
    }

    // ==================================================================== D2Q4444 Euler (multi-block, 2D)
    namespace
    {
        // D2Q4444 Euler Lax-Liu config 12 (four D2Q4 blocks) on an open [0,1]^2 domain.
        range_result run_d2q4444_lax_liu(std::size_t max_level, double Tf, bool adapt, double eps = 1e-4)
        {
            static constexpr std::size_t dim = 2;
            const double lambda = 5., gamma = 1.4, gm1 = gamma - 1.;
            const double l = lambda, l2 = l * l;

            // config 12
            const std::array<double, 4> rho0{0.8, 1., 1., 0.5313};
            const std::array<double, 4> ux0{0., 0.7276, 0., 0.};
            const std::array<double, 4> uy0{0., 0., 0.7276, 0.};
            const std::array<double, 4> p0{1., 1., 1., 0.4};

            Box<double, dim> box({0., 0.}, {1., 1.});
            const std::size_t ml = adapt ? 2 : max_level;
            auto cfg             = mesh_config<dim>().min_level(ml).max_level(max_level).periodic(false).max_stencil_size(4);
            auto mesh            = mra::make_mesh(box, cfg);

            auto m = make_vector_field<double, 16>("m", mesh);
            auto f = make_vector_field<double, 16>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double x = cell.center(0), y = cell.center(1);
                              const std::size_t q = (x < 0.5) ? (y < 0.5 ? 0 : 1) : (y < 0.5 ? 2 : 3);
                              const double rho = rho0[q], qx = rho * ux0[q], qy = rho * uy0[q];
                              m[cell](0)  = rho;
                              m[cell](4)  = qx;
                              m[cell](8)  = qy;
                              m[cell](12) = p0[q] / gm1 + 0.5 * (qx * qx + qy * qy) / rho;
                          });

            std::array<std::array<double, 4>, 4> M{
                {{1., 1., 1., 1.}, {l, 0., -l, 0.}, {0., l, 0., -l}, {l2, -l2, l2, -l2}}
            };
            std::array<std::array<double, 4>, 4> invM{
                {{0.25, 0.5 / l, 0., 0.25 / l2},
                 {0.25, 0., 0.5 / l, -0.25 / l2},
                 {0.25, -0.5 / l, 0., 0.25 / l2},
                 {0.25, 0., -0.5 / l, -0.25 / l2}}
            };
            const std::array<std::array<int, dim>, 4> vel{
                {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
            };
            auto eq_rho = [](std::array<double, 4>& meq, std::span<const double> mm)
            {
                meq[0] = mm[0];
                meq[1] = mm[4];
                meq[2] = mm[8];
                meq[3] = 0.;
            };
            auto eq_qx = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
                meq[0] = qx;
                meq[1] = (1.5 - 0.5 * gamma) * qx * qx / r + (0.5 - 0.5 * gamma) * qy * qy / r + gm1 * E;
                meq[2] = qx * qy / r;
                meq[3] = 0.;
            };
            auto eq_qy = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
                meq[0] = qy;
                meq[1] = qx * qy / r;
                meq[2] = (1.5 - 0.5 * gamma) * qy * qy / r + (0.5 - 0.5 * gamma) * qx * qx / r + gm1 * E;
                meq[3] = 0.;
            };
            auto eq_E = [gamma](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
                const double h = 0.5 * (gamma - 1.);
                meq[0]         = E;
                meq[1]         = gamma * qx * E / r - h * qx * qx * qx / (r * r) - h * qx * qy * qy / (r * r);
                meq[2]         = gamma * qy * E / r - h * qy * qy * qy / (r * r) - h * qy * qx * qx / (r * r);
                meq[3]         = 0.;
            };
            const std::array<double, 4> s_rho{0., 1.9, 1.9, 1.0};
            const std::array<double, 4> s_var{0., 1.75, 1.75, 1.0};

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D2Q4444",
                                                   lambda,
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_rho, eq_rho),
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_var, eq_qx),
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_var, eq_qy),
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_var, eq_E));
            make_bc<Neumann<1>>(f);
            scheme.init_equilibrium(f, m);

            const double dx = 1. / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));

            auto MRadaptation = make_MRAdapt(f);
            auto mra_config   = samurai::mra_config().epsilon(eps);
            if (adapt)
            {
                MRadaptation(mra_config);
                m.resize();
            }

            for (std::size_t n = 0; n < nt; ++n)
            {
                if (adapt)
                {
                    MRadaptation(mra_config);
                    m.resize();
                }
                scheme(f, m);
            }

            range_result r{0., std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double rho = m[cell](0), qx = m[cell](4), qy = m[cell](8), E = m[cell](12);
                              const double p = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / rho);
                              r.vmin         = std::min(r.vmin, rho);
                              r.vmax         = std::max(r.vmax, rho);
                              r.pmin         = std::min(r.pmin, p);
                          });
            return r;
        }
    }

    TEST(lbm_d2q4444, lax_liu_stays_physical_and_bounded)
    {
        auto r5 = run_d2q4444_lax_liu(5, 0.1, /*adapt*/ false);
        auto r6 = run_d2q4444_lax_liu(6, 0.1, /*adapt*/ false);
        // Positive density and pressure, no spurious overshoot.
        EXPECT_GT(r5.vmin, 0.);
        EXPECT_GT(r5.pmin, 0.);
        EXPECT_GT(r6.vmin, 0.);
        EXPECT_GT(r6.pmin, 0.);
        // The interaction builds up a bounded density peak (sharper under refinement).
        EXPECT_GT(r5.vmax, 1.);
        EXPECT_LT(r5.vmax, 2.5);
        EXPECT_GT(r6.vmax, 1.);
        EXPECT_LT(r6.vmax, 2.5);
    }

    TEST(lbm_d2q4444, lax_liu_adaptive_matches_uniform)
    {
        auto ru = run_d2q4444_lax_liu(6, 0.1, /*adapt*/ false);
        auto ra = run_d2q4444_lax_liu(6, 0.1, /*adapt*/ true, 1e-4);
        EXPECT_GT(ra.vmin, 0.);
        EXPECT_GT(ra.pmin, 0.);
        EXPECT_NEAR(ra.vmax, ru.vmax, 0.1);
    }

    // ============================================= D2Q4444 Euler reflecting (slip) wall
    namespace
    {
        struct implosion_result
        {
            double mass_drift;
            double rhomin;
            double pmin;
            double momentum_asymmetry; // |integral(qx) - integral(qy)|, zero by diagonal symmetry
        };

        // D2Q4444 Euler implosion in a closed box with multi-block reflecting slip walls.
        implosion_result run_d2q4444_implosion(std::size_t max_level, double Tf)
        {
            static constexpr std::size_t dim = 2;
            const double lambda = 10., gamma = 1.4, gm1 = gamma - 1., s_x = 1.9;
            const double l = lambda, l2 = l * l;

            Box<double, dim> box({0., 0.}, {1., 1.});
            auto cfg  = mesh_config<dim>().min_level(max_level).max_level(max_level).periodic(false).max_stencil_size(4);
            auto mesh = mra::make_mesh(box, cfg);

            auto m = make_vector_field<double, 16>("m", mesh);
            auto f = make_vector_field<double, 16>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const bool inside = (cell.center(0) + cell.center(1)) <= 0.5;
                              m[cell](0)        = inside ? 0.125 : 1.;
                              m[cell](12)       = (inside ? 0.14 : 1.) / gm1;
                          });

            std::array<std::array<double, 4>, 4> M{
                {{1., 1., 1., 1.}, {l, 0., -l, 0.}, {0., l, 0., -l}, {l2, -l2, l2, -l2}}
            };
            std::array<std::array<double, 4>, 4> invM{
                {{0.25, 0.5 / l, 0., 0.25 / l2},
                 {0.25, 0., 0.5 / l, -0.25 / l2},
                 {0.25, -0.5 / l, 0., 0.25 / l2},
                 {0.25, 0., -0.5 / l, -0.25 / l2}}
            };
            const std::array<std::array<int, dim>, 4> vel{
                {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
            };
            auto eq_rho = [](std::array<double, 4>& meq, std::span<const double> mm)
            {
                meq[0] = mm[0];
                meq[1] = mm[4];
                meq[2] = mm[8];
                meq[3] = 0.;
            };
            auto eq_qx = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
                meq[0] = qx;
                meq[1] = (1.5 - 0.5 * gamma) * qx * qx / r + (0.5 - 0.5 * gamma) * qy * qy / r + gm1 * E;
                meq[2] = qx * qy / r;
                meq[3] = 0.;
            };
            auto eq_qy = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
                meq[0] = qy;
                meq[1] = qx * qy / r;
                meq[2] = (1.5 - 0.5 * gamma) * qy * qy / r + (0.5 - 0.5 * gamma) * qx * qx / r + gm1 * E;
                meq[3] = 0.;
            };
            auto eq_E = [gamma](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
                const double h = 0.5 * (gamma - 1.);
                meq[0]         = E;
                meq[1]         = gamma * qx * E / r - h * qx * qx * qx / (r * r) - h * qx * qy * qy / (r * r);
                meq[2]         = gamma * qy * E / r - h * qy * qy * qy / (r * r) - h * qy * qx * qx / (r * r);
                meq[3]         = 0.;
            };
            const std::array<double, 4> s_blk{0., s_x, s_x, 1.0};

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D2Q4444_implosion",
                                                   lambda,
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_rho),
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_qx),
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_qy),
                                                   velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_E));

            std::array<std::array<int, dim>, 16> velocities{};
            for (std::size_t blk = 0; blk < 4; ++blk)
            {
                for (std::size_t k = 0; k < 4; ++k)
                {
                    velocities[4 * blk + k] = vel[k];
                }
            }
            make_bc<BounceBack>(f, velocities, std::vector<std::size_t>{4, 4, 4, 4}, std::vector<int>{-1, 0, 1, -1});
            scheme.init_equilibrium(f, m);

            const double dx = 1. / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));

            auto mass = [&]()
            {
                double s = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  const double area = cell.length * cell.length;
                                  s += (f[cell](0) + f[cell](1) + f[cell](2) + f[cell](3)) * area;
                              });
                return s;
            };
            const double mass0 = mass();

            for (std::size_t n = 0; n < nt; ++n)
            {
                scheme(f, m);
            }

            implosion_result r{0., std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0.};
            double sum_qx = 0., sum_qy = 0.;
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double area = cell.length * cell.length;
                              const double rho = m[cell](0), qx = m[cell](4), qy = m[cell](8), E = m[cell](12);
                              const double p = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / rho);
                              r.rhomin       = std::min(r.rhomin, rho);
                              r.pmin         = std::min(r.pmin, p);
                              sum_qx += qx * area;
                              sum_qy += qy * area;
                          });
            r.mass_drift         = std::abs(mass() - mass0);
            r.momentum_asymmetry = std::abs(sum_qx - sum_qy);
            return r;
        }
    }

    TEST(lbm_d2q4444, implosion_reflecting_wall_conserves_mass_and_symmetry)
    {
        auto r = run_d2q4444_implosion(6, 0.5);
        // Closed box with reflecting walls: the mass is conserved to round-off.
        EXPECT_LT(r.mass_drift, 1e-11);
        // Physical state throughout.
        EXPECT_GT(r.rhomin, 0.);
        EXPECT_GT(r.pmin, 0.);
        // The initial datum is symmetric under (x,y) -> (y,x); the reflecting walls preserve it,
        // so the total x- and y-momenta stay equal.
        EXPECT_LT(r.momentum_asymmetry, 1e-10);
    }

    // ============================================= D2Q5444 Euler with gravity (source term)
    namespace
    {
        struct rt_result
        {
            double mass_drift;
            double rhomin;
            double pmin;
            double umax;
        };

        // D2Q5444 compressible Euler with a gravity source in a closed box. uniform_density == true
        // gives a single hydrostatic layer (equilibrium the scheme should keep at rest up to a
        // consistent discretisation error); false gives the Rayleigh-Taylor two-layer datum.
        rt_result run_d2q5444_rt(std::size_t max_level, double Tf, bool uniform_density)
        {
            static constexpr std::size_t dim = 2;
            const double lambda = 5., gamma = 1.4, gm1 = gamma - 1., g = 2.;
            const double rho_down = 1., rho_up = uniform_density ? 1. : 2.;
            const double l = lambda, l2 = l * l;
            constexpr double pi = 3.14159265358979323846;

            Box<double, dim> box({0., 0.}, {1., 1.});
            auto cfg  = mesh_config<dim>().min_level(max_level).max_level(max_level).periodic(false).max_stencil_size(4);
            auto mesh = mra::make_mesh(box, cfg);

            auto m = make_vector_field<double, 17>("m", mesh);
            auto f = make_vector_field<double, 17>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double x = cell.center(0), y = cell.center(1);
                              const double y_i = 0.5 + 0.01 * std::cos(4. * pi * x);
                              double rho, press;
                              if (y < y_i)
                              {
                                  rho   = rho_down;
                                  press = 1. + (1. - y_i) * g * rho_up + (y_i - y) * g * rho_down;
                              }
                              else
                              {
                                  rho   = rho_up;
                                  press = 1. + (1. - y) * g * rho_up;
                              }
                              m[cell](0)  = rho;
                              m[cell](13) = press / gm1;
                          });

            std::array<std::array<double, 5>, 5> M5{
                {{1., 1., 1., 1., 1.},
                 {0., l, 0., -l, 0.},
                 {0., 0., l, 0., -l},
                 {-4. * l2 / 5., 21. * l2 / 5., 21. * l2 / 5., 21. * l2 / 5., 21. * l2 / 5.},
                 {0., l2, -l2, l2, -l2}}
            };
            std::array<std::array<double, 5>, 5> invM5{
                {{21. / 25., 0., 0., -1. / (5. * l2), 0.},
                 {1. / 25., 0.5 / l, 0., 1. / (20. * l2), 0.25 / l2},
                 {1. / 25., 0., 0.5 / l, 1. / (20. * l2), -0.25 / l2},
                 {1. / 25., -0.5 / l, 0., 1. / (20. * l2), 0.25 / l2},
                 {1. / 25., 0., -0.5 / l, 1. / (20. * l2), -0.25 / l2}}
            };
            const std::array<std::array<int, dim>, 5> vel5{
                {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}}
            };
            std::array<std::array<double, 4>, 4> M4{
                {{1., 1., 1., 1.}, {l, 0., -l, 0.}, {0., l, 0., -l}, {l2, -l2, l2, -l2}}
            };
            std::array<std::array<double, 4>, 4> invM4{
                {{0.25, 0.5 / l, 0., 0.25 / l2},
                 {0.25, 0., 0.5 / l, -0.25 / l2},
                 {0.25, -0.5 / l, 0., 0.25 / l2},
                 {0.25, 0., -0.5 / l, -0.25 / l2}}
            };
            const std::array<std::array<int, dim>, 4> vel4{
                {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
            };
            auto eq_rho = [](std::array<double, 5>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[5], qy = mm[9];
                meq[0] = r;
                meq[1] = qx;
                meq[2] = qy;
                meq[3] = (qx * qx + qy * qy) / r;
                meq[4] = 0.;
            };
            auto eq_qx = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[5], qy = mm[9], E = mm[13];
                meq[0] = qx;
                meq[1] = (1.5 - 0.5 * gamma) * qx * qx / r + (0.5 - 0.5 * gamma) * qy * qy / r + gm1 * E;
                meq[2] = qx * qy / r;
                meq[3] = 0.;
            };
            auto eq_qy = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[5], qy = mm[9], E = mm[13];
                meq[0] = qy;
                meq[1] = qx * qy / r;
                meq[2] = (1.5 - 0.5 * gamma) * qy * qy / r + (0.5 - 0.5 * gamma) * qx * qx / r + gm1 * E;
                meq[3] = 0.;
            };
            auto eq_E = [gamma](std::array<double, 4>& meq, std::span<const double> mm)
            {
                const double r = mm[0], qx = mm[5], qy = mm[9], E = mm[13];
                const double h = 0.5 * (gamma - 1.);
                meq[0]         = E;
                meq[1]         = gamma * qx * E / r - h * qx * qx * qx / (r * r) - h * qx * qy * qy / (r * r);
                meq[2]         = gamma * qy * E / r - h * qy * qy * qy / (r * r) - h * qy * qx * qx / (r * r);
                meq[3]         = 0.;
            };
            const std::array<double, 5> s_rho{0., 1.75, 1.75, 1.0, 1.0};
            const std::array<double, 4> s_var{0., 1.5, 1.5, 1.0};

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D2Q5444_rt",
                                                   lambda,
                                                   velocity_scheme<dim, 5>(vel5, M5, invM5, s_rho, eq_rho),
                                                   velocity_scheme<dim, 4>(vel4, M4, invM4, s_var, eq_qx),
                                                   velocity_scheme<dim, 4>(vel4, M4, invM4, s_var, eq_qy),
                                                   velocity_scheme<dim, 4>(vel4, M4, invM4, s_var, eq_E));
            scheme.set_source(
                [g](std::span<double> mm, double dt)
                {
                    mm[9] += -mm[0] * g * dt;
                    mm[13] += -mm[9] * g * dt;
                });

            std::array<std::array<int, dim>, 17> velocities{};
            for (std::size_t k = 0; k < 5; ++k)
            {
                velocities[k] = vel5[k];
            }
            for (std::size_t blk = 0; blk < 3; ++blk)
            {
                for (std::size_t k = 0; k < 4; ++k)
                {
                    velocities[5 + 4 * blk + k] = vel4[k];
                }
            }
            make_bc<BounceBack>(f, velocities, std::vector<std::size_t>{5, 4, 4, 4}, std::vector<int>{-1, 0, 1, -1});
            scheme.init_equilibrium(f, m);

            const double dx = 1. / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));

            auto mass = [&]()
            {
                double s = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  const double area = cell.length * cell.length;
                                  s += (f[cell](0) + f[cell](1) + f[cell](2) + f[cell](3) + f[cell](4)) * area;
                              });
                return s;
            };
            const double mass0 = mass();

            for (std::size_t n = 0; n < nt; ++n)
            {
                scheme(f, m, dt);
            }

            rt_result r{0., std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0.};
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double rho = m[cell](0), qx = m[cell](5), qy = m[cell](9), E = m[cell](13);
                              const double p = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / rho);
                              r.rhomin       = std::min(r.rhomin, rho);
                              r.pmin         = std::min(r.pmin, p);
                              r.umax         = std::max(r.umax, std::sqrt(qx * qx + qy * qy) / rho);
                          });
            r.mass_drift = std::abs(mass() - mass0);
            return r;
        }
    }

    TEST(lbm_d2q5444, rayleigh_taylor_source_conserves_mass_and_grows)
    {
        auto r = run_d2q5444_rt(6, 1.0, /*uniform_density*/ false);
        // Closed box, density carries no source: mass conserved to round-off.
        EXPECT_LT(r.mass_drift, 1e-11);
        EXPECT_GT(r.rhomin, 0.);
        EXPECT_GT(r.pmin, 0.);
        // The heavy-over-light interface is unstable: the perturbation has grown into motion.
        EXPECT_GT(r.umax, 0.02);
    }

    TEST(lbm_d2q5444, gravity_source_hydrostatic_error_is_consistent)
    {
        // A single hydrostatic layer is a rest equilibrium; the (non-well-balanced) scheme keeps
        // it at rest up to a discretisation error that must shrink under refinement.
        auto r6 = run_d2q5444_rt(6, 0.5, /*uniform_density*/ true);
        auto r7 = run_d2q5444_rt(7, 0.5, /*uniform_density*/ true);
        EXPECT_LT(r6.mass_drift, 1e-11);
        EXPECT_LT(r7.mass_drift, 1e-11);
        EXPECT_GT(r6.pmin, 0.);
        EXPECT_GT(r7.pmin, 0.);
        // Spurious current decreases with resolution (consistent gravity source).
        EXPECT_LT(r7.umax, r6.umax);
    }

    // ==================================================================== D1Q3 + wall BC
    namespace
    {
        struct sw_result
        {
            double mass_drift;
            double hmin;
            double hmax;
            double max_dev_from_rest; // max |h - h_rest| for the flat-rest test
        };

        // D1Q3 shallow-water dam break on a non-periodic domain with a chosen wall BC.
        sw_result run_d1q3_dam(std::size_t max_level, double Tf, double hL, double hR, const std::string& bc)
        {
            static constexpr std::size_t dim = 1;
            const double lambda = 2., g = 1., s2 = 1.5;
            const double left = -1., right = 1., L = right - left;

            Box<double, dim> box({left}, {right});
            auto cfg  = mesh_config<dim>().min_level(max_level).max_level(max_level).periodic(false).max_stencil_size(4);
            auto mesh = mra::make_mesh(box, cfg);

            auto m = make_vector_field<double, 3>("m", mesh);
            auto f = make_vector_field<double, 3>("f", mesh);
            m.fill(0.);
            f.fill(0.);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double h = (cell.center(0) < 0.) ? hL : hR;
                              m[cell](0)     = h;
                              m[cell](1)     = 0.;
                              m[cell](2)     = 0.5 * g * h * h;
                          });

            const double l = lambda, l2 = lambda * lambda;
            std::array<std::array<double, 3>, 3> M{
                {{1., 1., 1.}, {0., l, -l}, {0., l2, l2}}
            };
            std::array<std::array<double, 3>, 3> invM{
                {{1., 0., -1. / l2}, {0., 0.5 / l, 0.5 / l2}, {0., -0.5 / l, 0.5 / l2}}
            };
            auto eq = [g](std::array<double, 3>& meq, std::span<const double> mm)
            {
                const double h = mm[0], q = mm[1];
                meq[0] = h;
                meq[1] = q;
                meq[2] = q * q / h + 0.5 * g * h * h;
            };

            using field_t = decltype(f);
            auto scheme   = make_lbm_scheme<field_t>("D1Q3",
                                                   lambda,
                                                   velocity_scheme<dim, 3>(
                                                       {
                                                           {{0}, {1}, {-1}}
            },
                                                       M,
                                                       invM,
                                                       {0., 0., s2},
                                                       eq));

            const std::array<std::array<int, dim>, 3> velocities{
                {{0}, {1}, {-1}}
            };
            const xt::xtensor_fixed<int, xt::xshape<dim>> west{-1};
            const xt::xtensor_fixed<int, xt::xshape<dim>> east{1};
            if (bc == "antibounceback")
            {
                make_bc<AntiBounceBack>(f, velocities, scheme.equilibrium_f({hL, 0., 0.}))->on(west);
                make_bc<AntiBounceBack>(f, velocities, scheme.equilibrium_f({hR, 0., 0.}))->on(east);
            }
            else
            {
                make_bc<BounceBack>(f, velocities);
            }
            scheme.init_equilibrium(f, m);

            const double dx = L / static_cast<double>(std::size_t{1} << max_level);
            const double dt = dx / lambda;
            const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));

            auto mass = [&]()
            {
                double s = 0.;
                for_each_cell(mesh,
                              [&](const auto& cell)
                              {
                                  s += (f[cell](0) + f[cell](1) + f[cell](2)) * cell.length;
                              });
                return s;
            };
            const double mass0  = mass();
            const double h_rest = hL; // used only when hL == hR

            for (std::size_t n = 0; n < nt; ++n)
            {
                scheme(f, m);
            }

            sw_result r{std::abs(mass() - mass0), std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.};
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const double h      = m[cell](0);
                              r.hmin              = std::min(r.hmin, h);
                              r.hmax              = std::max(r.hmax, h);
                              r.max_dev_from_rest = std::max(r.max_dev_from_rest, std::abs(h - h_rest));
                          });
            return r;
        }
    }

    TEST(lbm_d1q3_bc, bounce_back_conserves_mass_and_stays_positive)
    {
        auto r = run_d1q3_dam(6, 1.0, /*hL*/ 2., /*hR*/ 1., "bounceback");
        // A closed (bounce-back) basin conserves the water mass to round-off.
        EXPECT_LT(r.mass_drift, 1e-11);
        // Water height stays strictly positive and bounded by the initial extremes.
        EXPECT_GT(r.hmin, 0.);
        EXPECT_LE(r.hmax, 2.05);
    }

    TEST(lbm_d1q3_bc, anti_bounce_back_preserves_the_rest_state_exactly)
    {
        // A flat rest state matching the imposed wall height must stay perfectly flat.
        auto r = run_d1q3_dam(6, 2.0, /*hL*/ 1.5, /*hR*/ 1.5, "antibounceback");
        EXPECT_LT(r.max_dev_from_rest, 1e-10);
    }

    // ==================================================================== D2Q9 MRT Navier-Stokes
    TEST(lbm_d2q9, taylor_green_viscous_decay_and_mass)
    {
        static constexpr std::size_t dim = 2;
        const double lambda = 1., U0 = 0.05, rho0 = 1., nu = 0.02, Tf = 2.;
        const std::size_t max_level = 5;
        const double left = 0., right = 2. * pi;

        Box<double, dim> box({left, left}, {right, right});
        auto cfg  = mesh_config<dim>().min_level(max_level).max_level(max_level).periodic(true).max_stencil_size(4);
        auto mesh = mra::make_mesh(box, cfg);

        const double dx   = (right - left) / static_cast<double>(std::size_t{1} << max_level);
        const double dt   = dx / lambda;
        const double cs2  = lambda * lambda / 3.;
        const double s_nu = 1. / (0.5 + nu / (cs2 * dt));

        auto u_exact = [&](double x, double y, double t)
        {
            return -U0 * std::cos(x) * std::sin(y) * std::exp(-2. * nu * t);
        };
        auto v_exact = [&](double x, double y, double t)
        {
            return U0 * std::sin(x) * std::cos(y) * std::exp(-2. * nu * t);
        };

        auto m = make_vector_field<double, 9>("m", mesh);
        auto f = make_vector_field<double, 9>("f", mesh);
        m.fill(0.);
        f.fill(0.);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          const double x = cell.center(0), y = cell.center(1);
                          m[cell](0) = rho0;
                          m[cell](1) = rho0 * u_exact(x, y, 0.);
                          m[cell](2) = rho0 * v_exact(x, y, 0.);
                      });

        // Explicit Lallemand-Luo M (f -> m) and its exact inverse invM (m -> f); no numerical inverse.
        const double l  = lambda;
        const double l2 = lambda * lambda, l3 = l2 * lambda, l4 = l2 * l2;
        std::array<std::array<double, 9>, 9> M{
            {
             {1., 1., 1., 1., 1., 1., 1., 1., 1.},
             {0., l, 0., -l, 0., l, -l, -l, l},
             {0., 0., l, 0., -l, l, l, -l, -l},
             {-4. * l2, -l2, -l2, -l2, -l2, 2. * l2, 2. * l2, 2. * l2, 2. * l2},
             {0., -2. * l3, 0., 2. * l3, 0., l3, -l3, -l3, l3},
             {0., 0., -2. * l3, 0., 2. * l3, l3, l3, -l3, -l3},
             {4. * l4, -2. * l4, -2. * l4, -2. * l4, -2. * l4, l4, l4, l4, l4},
             {0., l2, -l2, l2, -l2, 0., 0., 0., 0.},
             {0., 0., 0., 0., 0., l2, -l2, l2, -l2},
             }
        };
        const double r1 = 1. / lambda, r2 = r1 * r1, r3 = r2 * r1, r4 = r3 * r1;
        std::array<std::array<double, 9>, 9> invM{
            {
             {1. / 9, 0., 0., -r2 / 9, 0., 0., r4 / 9, 0., 0.},
             {1. / 9, r1 / 6, 0., -r2 / 36, -r3 / 6, 0., -r4 / 18, r2 / 4, 0.},
             {1. / 9, 0., r1 / 6, -r2 / 36, 0., -r3 / 6, -r4 / 18, -r2 / 4, 0.},
             {1. / 9, -r1 / 6, 0., -r2 / 36, r3 / 6, 0., -r4 / 18, r2 / 4, 0.},
             {1. / 9, 0., -r1 / 6, -r2 / 36, 0., r3 / 6, -r4 / 18, -r2 / 4, 0.},
             {1. / 9, r1 / 6, r1 / 6, r2 / 18, r3 / 12, r3 / 12, r4 / 36, 0., r2 / 4},
             {1. / 9, -r1 / 6, r1 / 6, r2 / 18, -r3 / 12, r3 / 12, r4 / 36, 0., -r2 / 4},
             {1. / 9, -r1 / 6, -r1 / 6, r2 / 18, -r3 / 12, -r3 / 12, r4 / 36, 0., r2 / 4},
             {1. / 9, r1 / 6, -r1 / 6, r2 / 18, r3 / 12, -r3 / 12, r4 / 36, 0., -r2 / 4},
             }
        };
        // Sanity: M is the exact inverse of invM.
        for (std::size_t a = 0; a < 9; ++a)
        {
            for (std::size_t b = 0; b < 9; ++b)
            {
                double acc = 0.;
                for (std::size_t k = 0; k < 9; ++k)
                {
                    acc += M[a][k] * invM[k][b];
                }
                EXPECT_NEAR(acc, (a == b) ? 1. : 0., 1e-12);
            }
        }
        auto eq = [l2, l4](std::array<double, 9>& meq, std::span<const double> mm)
        {
            const double rho = mm[0], qx = mm[1], qy = mm[2];
            const double q2 = (qx * qx + qy * qy) / rho;
            meq[0]          = rho;
            meq[1]          = qx;
            meq[2]          = qy;
            meq[3]          = -2. * l2 * rho + 3. * q2;
            meq[4]          = -l2 * qx;
            meq[5]          = -l2 * qy;
            meq[6]          = l4 * rho - 3. * l2 * q2;
            meq[7]          = (qx * qx - qy * qy) / rho;
            meq[8]          = qx * qy / rho;
        };
        std::array<double, 9> s{0., 0., 0., 1.64, 1.54, 1.54, 1.64, s_nu, s_nu};

        using field_t = decltype(f);
        auto scheme   = make_lbm_scheme<field_t>("D2Q9",
                                               lambda,
                                               velocity_scheme<dim, 9>(
                                                   {
                                                       {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
        },
                                                   M,
                                                   invM,
                                                   s,
                                                   eq));
        scheme.init_equilibrium(f, m);

        const auto nt   = static_cast<std::size_t>(std::round(Tf / dt));
        const double Te = static_cast<double>(nt) * dt;

        auto mass = [&]()
        {
            double acc = 0.;
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              acc += m[cell](0) * cell.length * cell.length;
                          });
            return acc;
        };
        const double mass0 = mass();

        for (std::size_t n = 0; n < nt; ++n)
        {
            scheme(f, m);
        }

        double err = 0., nrm = 0., umax = 0.;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          const double rho = m[cell](0);
                          const double u = m[cell](1) / rho, v = m[cell](2) / rho;
                          const double ue = u_exact(cell.center(0), cell.center(1), Te);
                          const double ve = v_exact(cell.center(0), cell.center(1), Te);
                          err += ((u - ue) * (u - ue) + (v - ve) * (v - ve)) * cell.length * cell.length;
                          nrm += (ue * ue + ve * ve) * cell.length * cell.length;
                          umax = std::max(umax, std::sqrt(u * u + v * v));
                      });
        const double rel_l2 = std::sqrt(err / nrm);

        // Mass conserved to round-off.
        EXPECT_LT(std::abs(mass() - mass0), 1e-10);
        // Peak velocity follows the analytic viscous decay exp(-2 nu t).
        const double umax_exact = U0 * std::exp(-2. * nu * Te);
        EXPECT_NEAR(umax, umax_exact, 0.05 * U0);
        // Velocity field matches the exact Taylor-Green vortex to a few percent at this resolution.
        EXPECT_LT(rel_l2, 0.05);
    }
}
