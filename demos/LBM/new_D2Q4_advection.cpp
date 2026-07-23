// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D2Q4 scalar advection, validating the N-D stream of the schemes/lbm module.
// One block of 4 velocities {(1,0),(0,1),(-1,0),(0,-1)}.
//
// Moments : m0 = f0+f1+f2+f3            (conserved, u)
//           m1 = lambda (f0 - f2)       (x-flux)
//           m2 = lambda (f1 - f3)       (y-flux)
//           m3 = lambda^2 (f0-f1+f2-f3) (diagonal)
// Equilibrium (advection at velocity (ax, ay)) : m1^eq = ax u, m2^eq = ay u, m3^eq = 0

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
    auto& app = samurai::initialize("D2Q4 scalar advection (schemes/lbm, N-D stream)", argc, argv);

    static constexpr std::size_t dim = 2;
    using Box                        = samurai::Box<double, dim>;

    // Parameters
    double left_box       = -1.;
    double right_box      = 1.;
    std::size_t max_level = 7;
    std::size_t min_level = 2;
    double eps            = 1e-4;
    bool adapt            = false;
    double lambda         = 1.;
    double ax             = 0.5;
    double ay             = 0.5;
    double s1             = 1.5; // relaxation of the flux moments
    double s2             = 1.0; // relaxation of the diagonal moment
    double Tf             = 0.4;
    fs::path path         = fs::current_path();
    std::string filename  = "new_D2Q4_advection";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--ax", ax, "Advection velocity (x)")->capture_default_str();
    app.add_option("--ay", ay, "Advection velocity (y)")->capture_default_str();
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    app.add_option("--filename", filename, "File name prefix")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    // Periodic mesh (uniform if !adapt, else min_level..max_level).
    const Box box({left_box, left_box}, {right_box, right_box});
    const std::size_t ml = adapt ? min_level : max_level;
    auto config          = samurai::mesh_config<dim>().min_level(ml).max_level(max_level).periodic(true).max_stencil_size(4);
    auto mesh            = samurai::mra::make_mesh(box, config);

    const double L      = right_box - left_box;
    constexpr double pi = 3.14159265358979323846;
    auto u0             = [&](double x, double y)
    {
        return std::sin(pi * x) * std::sin(pi * y); // periodic on [-1, 1]^2
    };

    // Fields: n_comp = 4 (D2Q4)
    auto m = samurai::make_vector_field<double, 4>("m", mesh);
    auto f = samurai::make_vector_field<double, 4>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               m[cell](0) = u0(cell.center(0), cell.center(1));
                           });

    // D2Q4 scheme
    using field_t = decltype(f);
    const double l  = lambda;
    const double l2 = lambda * lambda;
    std::array<std::array<double, 4>, 4> M{{{1., 1., 1., 1.}, {l, 0., -l, 0.}, {0., l, 0., -l}, {l2, -l2, l2, -l2}}};
    std::array<std::array<double, 4>, 4> invM{{{0.25, 0.5 / l, 0., 0.25 / l2},
                                               {0.25, 0., 0.5 / l, -0.25 / l2},
                                               {0.25, -0.5 / l, 0., 0.25 / l2},
                                               {0.25, 0., -0.5 / l, -0.25 / l2}}};
    auto eq = [ax, ay](std::array<double, 4>& meq, std::span<const double> mm)
    {
        meq[0] = mm[0];      // conserved
        meq[1] = ax * mm[0]; // x-flux
        meq[2] = ay * mm[0]; // y-flux
        meq[3] = 0.;         // diagonal
    };

    auto scheme = samurai::make_lbm_scheme<field_t>(
        "D2Q4_advection",
        lambda,
        samurai::velocity_scheme<dim, 4>({{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}}, M, invM, {0., s1, s1, s2}, eq));
    scheme.set_max_level(max_level);
    scheme.init_equilibrium(f, m);

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

    for (std::size_t n = 0; n < nt; ++n)
    {
        if (adapt)
        {
            MRadaptation(mra_config);
            m.resize();
        }
        scheme(f, m);
    }

    std::cout << "case = D2Q4 advection, " << (adapt ? "adaptive" : "uniform") << ", max_level = " << max_level
              << (adapt ? (", min_level = " + std::to_string(min_level)) : "") << ", cells = " << mesh.nb_cells()
              << ", dt = " << dt << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;

    // Error vs exact u(x,y,t) = u0(x - ax t, y - ay t) (periodic wrap), area-weighted
    double err_l2 = 0.;
    double norm   = 0.;
    double umin   = std::numeric_limits<double>::max();
    double umax   = std::numeric_limits<double>::lowest();
    auto wrap     = [&](double z)
    {
        return left_box + std::fmod(std::fmod(z - left_box, L) + L, L);
    };
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double area  = cell.length * cell.length;
                               const double exact = u0(wrap(cell.center(0) - ax * Tf_eff), wrap(cell.center(1) - ay * Tf_eff));
                               const double diff  = m[cell](0) - exact;
                               err_l2 += diff * diff * area;
                               norm += exact * exact * area;
                               umin = std::min(umin, m[cell](0));
                               umax = std::max(umax, m[cell](0));
                           });
    err_l2 = std::sqrt(err_l2 / norm);

    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", u in [" << umin << ", " << umax << "]" << std::endl;
    std::cout << "relative L2 error = " << err_l2 << std::endl;

    samurai::save(path, filename, mesh, m);

    samurai::finalize();
    return 0;
}
