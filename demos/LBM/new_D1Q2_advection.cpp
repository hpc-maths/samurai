// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D1Q2 linear advection / Burgers, validating the new schemes/lbm formalism.
// Uniform mesh (--level) or multiresolution (--adapt): the multi-level stream
// uses precomputed inflow/outflow prediction maps; the collision is local.
//
// Moments : m0 = f0 + f1        (conserved, u)
//           m1 = lambda (f0 - f1)
// Equilibrium (advection at speed a) : m1^eq = a * m0
// M    = [[1, 1], [lambda, -lambda]]
// M^-1 = [[1/2, 1/(2 lambda)], [1/2, -1/(2 lambda)]]

#include <cmath>
#include <limits>
#include <span>

#include <samurai/algorithm.hpp>
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
    auto& app = samurai::initialize("D1Q2 linear advection (new schemes/lbm formalism, uniform mesh)", argc, argv);

    static constexpr std::size_t dim = 1;
    using Box                        = samurai::Box<double, dim>;

    // Parameters
    double left_box       = -1.;
    double right_box      = 1.;
    std::size_t max_level = 8;
    std::size_t min_level = 2;
    double eps            = 1e-4;
    bool adapt            = false;
    double lambda         = 1.;
    double a              = 0.75; // advection velocity (|a| < lambda)
    double s1             = 1.5;  // relaxation parameter
    double Tf             = 0.4;
    bool burgers          = false;
    fs::path path         = fs::current_path();
    std::string filename  = "new_D1Q2_advection";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--velocity", a, "Advection velocity")->capture_default_str();
    app.add_option("--s", s1, "Relaxation parameter")->capture_default_str();
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_flag("--burgers", burgers, "Solve Burgers (equilibrium 1/2 u^2) instead of linear advection")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    app.add_option("--filename", filename, "File name prefix")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    if (burgers)
    {
        lambda = std::max(lambda, 2.); // lattice velocity must dominate |u|
    }

    // Periodic mesh (uniform if !adapt, else min_level..max_level).
    const Box box({left_box}, {right_box});
    const std::size_t ml = adapt ? min_level : max_level;
    auto config          = samurai::mesh_config<dim>().min_level(ml).max_level(max_level).periodic(true).max_stencil_size(4);
    auto mesh            = samurai::mra::make_mesh(box, config);

    const double L      = right_box - left_box;
    constexpr double pi = 3.14159265358979323846;
    auto u0             = [&](double x)
    {
        return std::sin(pi * x); // periodic on [-1, 1]
    };

    // Fields: n_comp = 2 (D1Q2)
    auto m = samurai::make_vector_field<double, 2>("m", mesh);
    auto f = samurai::make_vector_field<double, 2>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial moments (only the conserved moment u = m0 is meaningful here)
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               m[cell](0) = u0(cell.center(0));
                               m[cell](1) = 0.;
                           });

    // D1Q2 scheme definition
    using field_t = decltype(f);
    std::array<std::array<double, 2>, 2> M{
        {{1., 1.}, {lambda, -lambda}}
    };
    std::array<std::array<double, 2>, 2> invM{
        {{0.5, 0.5 / lambda}, {0.5, -0.5 / lambda}}
    };
    auto eq = [a, burgers](std::array<double, 2>& meq, std::span<const double> mm)
    {
        meq[0] = mm[0];                        // conserved
        meq[1] = burgers ? 0.5 * mm[0] * mm[0] // Burgers flux
                         : a * mm[0];          // linear advection flux
    };

    auto scheme = samurai::make_lbm_scheme<field_t>("D1Q2_advection",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 2>(
                                                        {
                                                            {{1}, {-1}}
    },
                                                        M,
                                                        invM,
                                                        {0., s1},
                                                        eq));

    scheme.init_equilibrium(f, m);

    // Time stepping: lambda = dx_fine / dt  =>  one-cell stream per step at the finest level
    const double dx_fine = L / static_cast<double>(std::size_t{1} << max_level);
    const double dt      = dx_fine / lambda;
    const auto nt        = static_cast<std::size_t>(std::round(Tf / dt));
    const double Tf_eff  = static_cast<double>(nt) * dt;

    // Adaptation acts on the distributions f (the full LBM state); m is a derived diagnostic.
    auto MRadaptation = samurai::make_MRAdapt(f);
    auto mra_config   = samurai::mra_config().epsilon(eps);
    if (adapt)
    {
        MRadaptation(mra_config);
        m.resize();
    }

    // Mass = integral of u = f0 + f1 (conserved for a periodic domain)
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

    std::cout << "case = " << (burgers ? "Burgers" : "advection") << ", " << (adapt ? "adaptive" : "uniform")
              << ", max_level = " << max_level << (adapt ? (", min_level = " + std::to_string(min_level)) : "")
              << ", cells = " << mesh.nb_cells() << ", dt = " << dt << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;

    // Mass conservation + boundedness
    double umin = std::numeric_limits<double>::max();
    double umax = std::numeric_limits<double>::lowest();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               umin = std::min(umin, m[cell](0));
                               umax = std::max(umax, m[cell](0));
                           });
    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", u in [" << umin << ", " << umax << "]" << std::endl;

    if (!burgers)
    {
        // Error vs the exact solution u(x, t) = u0(x - a t)  (periodic wrap), cell.length-weighted
        double err_l2 = 0.;
        double norm   = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   double x     = cell.center(0);
                                   double xs    = x - a * Tf_eff;
                                   xs           = left_box + std::fmod(std::fmod(xs - left_box, L) + L, L); // wrap into [left, right)
                                   double exact = u0(xs);
                                   double diff  = m[cell](0) - exact;
                                   err_l2 += diff * diff * cell.length;
                                   norm += exact * exact * cell.length;
                               });
        err_l2 = std::sqrt(err_l2 / norm);
        std::cout << "relative L2 error = " << err_l2 << std::endl;
    }

    samurai::save(path, filename, mesh, m);

    samurai::finalize();
    return 0;
}
