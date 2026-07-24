// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D2Q5444 compressible Euler with gravity (Rayleigh-Taylor instability), validating the LBM
// body-force SOURCE TERM of the schemes/lbm formalism. The scheme is a D2Q5 block for the
// density and three D2Q4 blocks for (qx, qy, E), coupled through the Euler equilibria; gravity
// enters as a source added after the collision on the conserved momentum and energy:
//
//   qy <- qy - rho * g * dt        (weight)
//   E  <- E  - qy  * g * dt        (rate of work of gravity, with the updated qy)
//
// registered with scheme.set_source(...) and applied by the scheme once per step (dt passed to
// the scheme call). The domain is a closed box with reflecting slip walls (multi-block
// bounce-back); a heavy layer sits on top of a light one in hydrostatic balance, and a small
// interface perturbation triggers the instability. The total mass stays conserved to round-off.

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/lbm.hpp>

#include <filesystem>
namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("D2Q5444 Euler Rayleigh-Taylor with gravity (schemes/lbm, source term)", argc, argv);

    static constexpr std::size_t dim = 2;
    using Box                        = samurai::Box<double, dim>;

    // Simulation parameters
    double lambda   = 5.;
    double gamma    = 1.4;
    double g        = 2.; // gravity (downward, -y)
    double rho_down = 1.; // light fluid, below
    double rho_up   = 2.; // heavy fluid, above (Rayleigh-Taylor unstable)
    double Tf       = 2.;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "new_D2Q5444_euler_rayleigh_taylor";
    std::size_t nfiles   = 1;

    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--gamma", gamma, "Ratio of specific heats")->capture_default_str()->group("Simulation parameters");
    app.add_option("--gravity", g, "Gravity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--rho-down", rho_down, "Density of the lower (light) layer")->capture_default_str()->group("Simulation parameters");
    app.add_option("--rho-up", rho_up, "Density of the upper (heavy) layer")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    // Closed box [0,1]^2.
    const Box box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(7).periodic(false).max_stencil_size(4).graduation_width(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

    // Fields: n_comp = 17 (D2Q5 + 3 x D2Q4)
    auto m = samurai::make_vector_field<double, 17>("m", mesh);
    auto f = samurai::make_vector_field<double, 17>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial state: two layers in hydrostatic balance (dp/dy = -rho g), heavy on top, at rest.
    // Interface y_i(x) = 0.5 + 0.01 cos(4 pi x); pressure = 1 at the top wall.
    // Layout: rho @ 0, qx @ 5, qy @ 9, E @ 13.
    constexpr double pi = 3.14159265358979323846;
    const double gm1    = gamma - 1.;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double x   = cell.center(0);
                               const double y   = cell.center(1);
                               const double y_i = 0.5 + 0.01 * std::cos(4. * pi * x);
                               double rho, press;
                               if (y < y_i) // lower light layer
                               {
                                   rho   = rho_down;
                                   press = 1. + (1. - y_i) * g * rho_up + (y_i - y) * g * rho_down;
                               }
                               else // upper heavy layer
                               {
                                   rho   = rho_up;
                                   press = 1. + (1. - y) * g * rho_up;
                               }
                               m[cell](0)  = rho;
                               m[cell](5)  = 0.;          // qx
                               m[cell](9)  = 0.;          // qy
                               m[cell](13) = press / gm1; // E (u = 0)
                           });

    // Block matrices.
    using field_t   = decltype(f);
    const double l  = lambda;
    const double l2 = lambda * lambda;

    // D2Q5 (density) block, velocities {(0,0),(1,0),(0,1),(-1,0),(0,-1)}.
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

    // D2Q4 blocks, velocities {(1,0),(0,1),(-1,0),(0,-1)}.
    std::array<std::array<double, 4>, 4> M4{
        {{1., 1., 1., 1.}, {l, 0., -l, 0.}, {0., l, 0., -l}, {l2, -l2, l2, -l2}}
    };
    std::array<std::array<double, 4>, 4> invM4{
        {{0.25, 0.5 / l, 0., 0.25 / l2}, {0.25, 0., 0.5 / l, -0.25 / l2}, {0.25, -0.5 / l, 0., 0.25 / l2}, {0.25, 0., -0.5 / l, -0.25 / l2}}
    };
    const std::array<std::array<int, dim>, 4> vel4{
        {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
    };

    // Conserved moments read from the full vector: rho @ 0, qx @ 5, qy @ 9, E @ 13.
    auto eq_rho = [](std::array<double, 5>& meq, std::span<const double> mm)
    {
        const double r = mm[0], qx = mm[5], qy = mm[9];
        meq[0] = r;                       // rho (conserved)
        meq[1] = qx;                      // x momentum
        meq[2] = qy;                      // y momentum
        meq[3] = (qx * qx + qy * qy) / r; // energy-like second moment
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

    auto scheme = samurai::make_lbm_scheme<field_t>("D2Q5444_rayleigh_taylor",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 5>(vel5, M5, invM5, s_rho, eq_rho),
                                                    samurai::velocity_scheme<dim, 4>(vel4, M4, invM4, s_var, eq_qx),
                                                    samurai::velocity_scheme<dim, 4>(vel4, M4, invM4, s_var, eq_qy),
                                                    samurai::velocity_scheme<dim, 4>(vel4, M4, invM4, s_var, eq_E));

    // Gravity source: weight on qy, and its rate of work on E (using the just-updated qy).
    scheme.set_source(
        [g](std::span<double> mm, double dt)
        {
            mm[9] += -mm[0] * g * dt;  // qy -= rho g dt
            mm[13] += -mm[9] * g * dt; // E  -= qy  g dt
        });

    // Reflecting slip walls on all four sides. Full 17-velocity list, block sizes {5,4,4,4},
    // momentum axis of each block (rho: -1, qx: 0, qy: 1, E: -1).
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
    const std::vector<std::size_t> block_sizes{5, 4, 4, 4};
    const std::vector<int> block_odd_axis{-1, 0, 1, -1};
    samurai::make_bc<samurai::BounceBack>(f, velocities, block_sizes, block_odd_axis);

    scheme.init_equilibrium(f, m);

    // Time stepping: lambda = dx_fine / dt  =>  one-cell stream per step at the finest level
    const double dt = mesh.min_cell_length() / lambda;

    // Save the diagnostic fields (density, velocity, pressure) and the mesh level at a given
    // output index.
    auto save_solution = [&](const std::string& suffix)
    {
        auto level    = samurai::make_scalar_field<std::size_t>("level", mesh);
        auto rho      = samurai::make_scalar_field<double>("rho", mesh);
        auto velocity = samurai::make_vector_field<double, 2>("velocity", mesh);
        auto p        = samurai::make_scalar_field<double>("p", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   level[cell]       = cell.level;
                                   const double r    = m[cell](0);
                                   const double qx   = m[cell](5);
                                   const double qy   = m[cell](9);
                                   const double E    = m[cell](13);
                                   rho[cell]         = r;
                                   velocity[cell](0) = qx / r;
                                   velocity[cell](1) = qy / r;
                                   p[cell]           = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / r);
                               });
        samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, m, rho, velocity, p, level);
    };

    auto MRadaptation = samurai::make_MRAdapt(f);
    auto mra_config   = samurai::mra_config();
    mra_config.parse_args();
    MRadaptation(mra_config, m);
    save_solution("_init");

    auto mass = [&]()
    {
        double s = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   const double area = cell.length * cell.length;
                                   s += (f[cell](0) + f[cell](1) + f[cell](2) + f[cell](3) + f[cell](4)) * area;
                               });
        return s;
    };
    const double mass0 = mass();

    const double dt_save = Tf / static_cast<double>(nfiles);
    std::size_t nsave    = 1;
    std::size_t nt       = 0;
    double t             = 0.;
    while (t != Tf)
    {
        MRadaptation(mra_config, m);

        t += dt;
        if (t > Tf)
        {
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {:.4f}, dt = {:.4e}", nt++, t, dt) << std::endl;

        scheme(f, m, dt);

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save_solution(suffix);
        }
    }

    double rhomin = std::numeric_limits<double>::max();
    double rhomax = std::numeric_limits<double>::lowest();
    double pmin   = std::numeric_limits<double>::max();
    double umax   = 0.;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double r  = m[cell](0);
                               const double qx = m[cell](5);
                               const double qy = m[cell](9);
                               const double E  = m[cell](13);
                               const double p  = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / r);
                               rhomin          = std::min(rhomin, r);
                               rhomax          = std::max(rhomax, r);
                               pmin            = std::min(pmin, p);
                               umax            = std::max(umax, std::sqrt(qx * qx + qy * qy) / r);
                           });

    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", rho in [" << rhomin << ", " << rhomax << "], p_min = " << pmin
              << ", |u|max = " << umax << std::endl;

    samurai::finalize();
    return 0;
}
