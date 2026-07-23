// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D2Q4444 Euler implosion in a closed box, validating the MULTI-BLOCK reflecting (slip) wall
// boundary condition of the schemes/lbm formalism. The scheme is the same four-block D2Q4 Euler
// discretisation as new_D2Q4444_euler_lax_liu; here the domain is a closed box with reflecting
// walls on all four sides, so the total mass is conserved to round-off and the diagonally
// symmetric initial datum stays symmetric.
//
// Reflecting slip wall: within each velocity block the incoming population is filled from the
// opposite one; the block carrying the momentum normal to the wall is reflected with sign -1
// (so the normal velocity vanishes at the wall), the others (density, energy, tangential
// momentum) with sign +1. This is expressed with the multi-block make_bc<BounceBack> overload,
// passing the block sizes {4,4,4,4} and the momentum axis of each block {-1, 0, 1, -1}.

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
    auto& app = samurai::initialize("D2Q4444 Euler implosion in a closed box (schemes/lbm, reflecting walls)", argc, argv);

    static constexpr std::size_t dim = 2;
    using Box                        = samurai::Box<double, dim>;

    // Simulation parameters
    double lambda = 10.;
    double gamma  = 1.4;
    double s_x    = 1.9; // relaxation of the flux moments (all blocks)
    double Tf     = 1.5;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "new_D2Q4444_euler_implosion";
    std::size_t nfiles   = 1;

    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--gamma", gamma, "Ratio of specific heats")->capture_default_str()->group("Simulation parameters");
    app.add_option("--s", s_x, "Relaxation parameter of the flux moments")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    // Closed box [0,1]^2.
    const Box box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(7).periodic(false).max_stencil_size(4).graduation_width(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

    // Fields: n_comp = 16 (D2Q4444)
    auto m = samurai::make_vector_field<double, 16>("m", mesh);
    auto f = samurai::make_vector_field<double, 16>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial state: low-density/low-pressure fluid inside the SW diamond x + y <= 0.5, ambient
    // fluid outside; at rest. The layout is rho @ 0, qx @ 4, qy @ 8, E @ 12.
    const double gm1 = gamma - 1.;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double x     = cell.center(0);
                               const double y     = cell.center(1);
                               const bool inside  = (x + y) <= 0.5;
                               const double rho   = inside ? 0.125 : 1.;
                               const double press = inside ? 0.14 : 1.;
                               m[cell](0)         = rho;
                               m[cell](4)         = 0.;          // qx
                               m[cell](8)         = 0.;          // qy
                               m[cell](12)        = press / gm1; // E (u = 0)
                           });

    // D2Q4 block matrices (shared by the four blocks).
    using field_t   = decltype(f);
    const double l  = lambda;
    const double l2 = lambda * lambda;
    std::array<std::array<double, 4>, 4> M{
        {{1., 1., 1., 1.}, {l, 0., -l, 0.}, {0., l, 0., -l}, {l2, -l2, l2, -l2}}
    };
    std::array<std::array<double, 4>, 4> invM{
        {{0.25, 0.5 / l, 0., 0.25 / l2}, {0.25, 0., 0.5 / l, -0.25 / l2}, {0.25, -0.5 / l, 0., 0.25 / l2}, {0.25, 0., -0.5 / l, -0.25 / l2}}
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

    auto scheme = samurai::make_lbm_scheme<field_t>("D2Q4444_euler_implosion",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_rho),
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_qx),
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_qy),
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_blk, eq_E));

    // Reflecting slip walls on all four sides: the full 16-velocity list (the 4 block velocities
    // repeated), the block sizes {4,4,4,4} and the momentum axis of each block (rho: -1, qx: 0,
    // qy: 1, E: -1).
    std::array<std::array<int, dim>, 16> velocities{};
    for (std::size_t blk = 0; blk < 4; ++blk)
    {
        for (std::size_t k = 0; k < 4; ++k)
        {
            velocities[4 * blk + k] = vel[k];
        }
    }
    const std::vector<std::size_t> block_sizes{4, 4, 4, 4};
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
                                   rho[cell]         = m[cell](0);
                                   velocity[cell](0) = m[cell](4) / m[cell](0);
                                   velocity[cell](1) = m[cell](8) / m[cell](0);
                                   p[cell]           = (gamma - 1.)
                                           * (m[cell](12) - 0.5 * (m[cell](4) * m[cell](4) + m[cell](8) * m[cell](8)) / m[cell](0));
                               });
        samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, m, rho, velocity, p, level);
    };

    auto MRadaptation = samurai::make_MRAdapt(f);
    auto mra_config   = samurai::mra_config();
    mra_config.parse_args();
    MRadaptation(mra_config, m);
    save_solution("_init");

    // Total mass (integral of rho = f0 + f1 + f2 + f3), computed right before the time loop to
    // track its conservation through the reflecting walls.
    auto mass = [&]()
    {
        double s = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   const double area = cell.length * cell.length;
                                   s += (f[cell](0) + f[cell](1) + f[cell](2) + f[cell](3)) * area;
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

        scheme(f, m);

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save_solution(suffix);
        }
    }

    double rhomin = std::numeric_limits<double>::max();
    double rhomax = std::numeric_limits<double>::lowest();
    double pmin   = std::numeric_limits<double>::max();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double r  = m[cell](0);
                               const double qx = m[cell](4);
                               const double qy = m[cell](8);
                               const double E  = m[cell](12);
                               const double p  = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / r);
                               rhomin          = std::min(rhomin, r);
                               rhomax          = std::max(rhomax, r);
                               pmin            = std::min(pmin, p);
                           });
    std::cout << "cells = " << mesh.nb_cells() << ", mass drift = " << std::abs(mass() - mass0) << ", rho in [" << rhomin << ", " << rhomax
              << "], p_min = " << pmin << std::endl;

    samurai::finalize();
    return 0;
}
