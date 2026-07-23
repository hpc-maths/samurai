// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D1Q222 Euler (Sod shock tube), validating the MULTI-BLOCK coupling of the schemes/lbm
// formalism: the 1D compressible Euler system is solved with three D1Q2 blocks (six velocities),
// one per conserved variable, coupled only through their equilibria.
//
//   block 0 (f0,f1)  ->  rho = f0 + f1                     (density)
//   block 1 (f2,f3)  ->  q   = f2 + f3                     (momentum rho*u)
//   block 2 (f4,f5)  ->  E   = f4 + f5                     (total energy)
// with, per block, the two D1Q2 moments  m0 = f+ + f-,  m1 = lambda (f+ - f-),  velocities {+1, -1}.
//
// The conserved moments are the three m0 (rho, q, E). The three first-order moments m1 relax
// towards the Euler fluxes (p = (gamma - 1)(E - 1/2 q^2/rho)):
//   block 0 : m1^eq = q                                     (mass flux    = rho*u)
//   block 1 : m1^eq = (3 - gamma)/2 q^2/rho + (gamma - 1) E  (momentum flux = rho*u^2 + p)
//   block 2 : m1^eq = gamma q E/rho + (1 - gamma)/2 q^3/rho^2 (energy flux  = u (E + p))
//
// Non-periodic domain with a zero-gradient (constant extension) boundary on both sides
// (homogeneous Neumann on the distributions f).

#include <cmath>
#include <limits>
#include <span>
#include <string>

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
    auto& app = samurai::initialize("D1Q222 Euler Sod shock tube (schemes/lbm, multi-block)", argc, argv);

    static constexpr std::size_t dim = 1;
    using Box                        = samurai::Box<double, dim>;

    // Simulation parameters
    double left_box  = -1.;
    double right_box = 1.;
    double lambda    = 3.;
    double gamma     = 1.4;
    double s_rel     = 1.5; // relaxation of the flux moments (all three blocks)
    double rhoL      = 1.;
    double rhoR      = 0.125;
    double pL        = 1.;
    double pR        = 0.1;
    double Tf        = 0.4;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "new_D1Q222_euler_sod";
    std::size_t nfiles   = 1;

    app.add_option("--left", left_box, "Left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "Right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--gamma", gamma, "Ratio of specific heats")->capture_default_str()->group("Simulation parameters");
    app.add_option("--s", s_rel, "Relaxation parameter of the flux moments")->capture_default_str()->group("Simulation parameters");
    app.add_option("--rhoL", rhoL, "Left density")->capture_default_str()->group("Simulation parameters");
    app.add_option("--rhoR", rhoR, "Right density")->capture_default_str()->group("Simulation parameters");
    app.add_option("--pL", pL, "Left pressure")->capture_default_str()->group("Simulation parameters");
    app.add_option("--pR", pR, "Right pressure")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    // NON-periodic mesh. Default levels (overridden by --min-level / --max-level); min < max enables
    // multiresolution, min == max gives a uniform mesh.
    const Box box({left_box}, {right_box});
    auto config = samurai::mesh_config<dim>().min_level(4).max_level(10).periodic(false).max_stencil_size(4).graduation_width(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

    // Fields: n_comp = 6 (D1Q222)
    auto m = samurai::make_vector_field<double, 6>("m", mesh);
    auto f = samurai::make_vector_field<double, 6>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial moments: Sod (left/right states at rest, u = 0). Only the conserved moments
    // (rho, q, E) are set; the flux moments are filled at equilibrium by init_equilibrium.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const bool left  = cell.center(0) < 0.;
                               const double rho = left ? rhoL : rhoR;
                               const double p   = left ? pL : pR;
                               m[cell](0)       = rho;              // rho
                               m[cell](2)       = 0.;               // q = rho*u
                               m[cell](4)       = p / (gamma - 1.); // E = p/(gamma-1) + 1/2 q^2/rho (u = 0)
                           });

    // D1Q222 scheme definition: three identical D1Q2 blocks, coupled through the equilibria.
    using field_t  = decltype(f);
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

    // Euler equilibria (each block sees the full moment vector mm = [rho, ., q, ., E, .]).
    auto eq_rho = [](std::array<double, 2>& meq, std::span<const double> mm)
    {
        meq[0] = mm[0]; // rho (conserved)
        meq[1] = mm[2]; // mass flux = q
    };
    auto eq_q = [gamma](std::array<double, 2>& meq, std::span<const double> mm)
    {
        const double rho = mm[0];
        const double q   = mm[2];
        const double E   = mm[4];
        meq[0]           = q;                                                  // q (conserved)
        meq[1]           = (3. - gamma) / 2. * q * q / rho + (gamma - 1.) * E; // momentum flux
    };
    auto eq_E = [gamma](std::array<double, 2>& meq, std::span<const double> mm)
    {
        const double rho = mm[0];
        const double q   = mm[2];
        const double E   = mm[4];
        meq[0]           = E;                                                                 // E (conserved)
        meq[1]           = gamma * q * E / rho + (1. - gamma) / 2. * q * q * q / (rho * rho); // energy flux
    };

    auto scheme = samurai::make_lbm_scheme<field_t>("D1Q222_euler",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 2>(vel, M, invM, {0., s_rel}, eq_rho),
                                                    samurai::velocity_scheme<dim, 2>(vel, M, invM, {0., s_rel}, eq_q),
                                                    samurai::velocity_scheme<dim, 2>(vel, M, invM, {0., s_rel}, eq_E));

    // Open ends: zero-gradient (constant extension) on the distributions.
    samurai::make_bc<samurai::Neumann<1>>(f);

    scheme.init_equilibrium(f, m);

    // Time stepping: lambda = dx_fine / dt  =>  one-cell stream per step at the finest level
    const double dt = mesh.min_cell_length() / lambda;

    // Save the density, velocity, pressure (and the mesh level) at a given output index.
    auto save_solution = [&](const std::string& suffix)
    {
        auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
        auto rho   = samurai::make_scalar_field<double>("rho", mesh);
        auto u     = samurai::make_scalar_field<double>("u", mesh);
        auto p     = samurai::make_scalar_field<double>("p", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   const double r = m[cell](0);
                                   const double q = m[cell](2);
                                   const double E = m[cell](4);
                                   level[cell]    = cell.level;
                                   rho[cell]      = r;
                                   u[cell]        = q / r;
                                   p[cell]        = (gamma - 1.) * (E - 0.5 * q * q / r);
                               });
        samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, m, rho, u, p, level);
    };

    auto MRadaptation = samurai::make_MRAdapt(f);
    auto mra_config   = samurai::mra_config();
    mra_config.parse_args();
    MRadaptation(mra_config, m);
    save_solution("_init");

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
                               const double rho = m[cell](0);
                               const double q   = m[cell](2);
                               const double E   = m[cell](4);
                               const double p   = (gamma - 1.) * (E - 0.5 * q * q / rho);
                               rhomin           = std::min(rhomin, rho);
                               rhomax           = std::max(rhomax, rho);
                               pmin             = std::min(pmin, p);
                           });
    std::cout << "cells = " << mesh.nb_cells() << ", rho in [" << rhomin << ", " << rhomax << "], p_min = " << pmin << std::endl;

    samurai::finalize();
    return 0;
}
