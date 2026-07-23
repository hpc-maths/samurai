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

    // Parameters
    double left_box       = -1.;
    double right_box      = 1.;
    std::size_t max_level = 9;
    std::size_t min_level = 2;
    double eps            = 1e-3;
    bool adapt            = false;
    double lambda         = 3.;
    double gamma          = 1.4;
    double s_rel          = 1.5; // relaxation of the flux moments (all three blocks)
    double rhoL           = 1.;
    double rhoR           = 0.125;
    double pL             = 1.;
    double pR             = 0.1;
    double Tf             = 0.4;
    fs::path path         = fs::current_path();
    std::string filename  = "new_D1Q222_euler_sod";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--gamma", gamma, "Ratio of specific heats")->capture_default_str();
    app.add_option("--s", s_rel, "Relaxation parameter of the flux moments")->capture_default_str();
    app.add_option("--rhoL", rhoL, "Left density")->capture_default_str();
    app.add_option("--rhoR", rhoR, "Right density")->capture_default_str();
    app.add_option("--pL", pL, "Left pressure")->capture_default_str();
    app.add_option("--pR", pR, "Right pressure")->capture_default_str();
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    app.add_option("--filename", filename, "File name prefix")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    // NON-periodic mesh (uniform if !adapt, else min_level..max_level).
    const Box box({left_box}, {right_box});
    const std::size_t ml = adapt ? min_level : max_level;
    auto config          = samurai::mesh_config<dim>().min_level(ml).max_level(max_level).periodic(false).max_stencil_size(4);
    auto mesh            = samurai::mra::make_mesh(box, config);

    const double L = right_box - left_box;

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
    const double dx_fine = L / static_cast<double>(std::size_t{1} << max_level);
    const double dt      = dx_fine / lambda;
    const auto nt        = static_cast<std::size_t>(std::round(Tf / dt));
    const double Tf_eff  = static_cast<double>(nt) * dt;

    auto MRadaptation = samurai::make_MRAdapt(f);
    auto mra_config   = samurai::mra_config().epsilon(eps);
    if (adapt)
    {
        MRadaptation(mra_config);
        m.resize();
    }

    // Total mass = integral of rho = f0 + f1
    auto mass = [&]()
    {
        double s = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   s += (f[cell](0) + f[cell](1)) * cell.length;
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

    std::cout << "case = D1Q222 Euler Sod, " << (adapt ? "adaptive" : "uniform") << ", max_level = " << max_level
              << (adapt ? (", min_level = " + std::to_string(min_level)) : "") << ", cells = " << mesh.nb_cells() << ", dt = " << dt
              << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;
    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", rho in [" << rhomin << ", " << rhomax << "], p_min = " << pmin
              << std::endl;

    // Diagnostic fields for the output
    auto rho = samurai::make_scalar_field<double>("rho", mesh);
    auto u   = samurai::make_scalar_field<double>("u", mesh);
    auto p   = samurai::make_scalar_field<double>("p", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double r = m[cell](0);
                               const double q = m[cell](2);
                               const double E = m[cell](4);
                               rho[cell]      = r;
                               u[cell]        = q / r;
                               p[cell]        = (gamma - 1.) * (E - 0.5 * q * q / r);
                           });
    samurai::save(path, filename, mesh, m, rho, u, p);

    samurai::finalize();
    return 0;
}
