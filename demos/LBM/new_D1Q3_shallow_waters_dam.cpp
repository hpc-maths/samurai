// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D1Q3 shallow-water dam break, validating the LBM wall boundary conditions
// (BounceBack / AntiBounceBack) of the schemes/lbm formalism on a NON-periodic domain.
//
// One block of 3 velocities {0, +1, -1}.
//   m0 = h        = f0 + f1 + f2                 (conserved, water height)
//   m1 = q        = lambda (f1 - f2)             (conserved, momentum)
//   m2 = kinetic  = lambda^2 (f1 + f2)           (relaxed towards q^2/h + 1/2 g h^2)
// M    = [[1, 1, 1], [0, lambda, -lambda], [0, lambda^2, lambda^2]]
// M^-1 = [[1, 0, -1/lambda^2], [0, 1/(2 lambda), 1/(2 lambda^2)], [0, -1/(2 lambda), 1/(2 lambda^2)]]
//
// Boundaries (default): reflecting solid walls on both sides (bounce-back), so the
// dam-break waves bounce back and forth and the total water mass is conserved to
// machine precision. With --bc antibounceback a fixed water height is imposed at each wall
// (an open "reservoir" condition: it exchanges mass with the reservoir, so the mass is NOT
// conserved, and it preserves a matching rest state exactly). It is well posed for moderate
// flows; a strong Riemann surge crashing against a low imposed height is physically stiff.

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
    auto& app = samurai::initialize("D1Q3 shallow-water dam break (schemes/lbm, wall boundary conditions)", argc, argv);

    static constexpr std::size_t dim = 1;
    using Box                        = samurai::Box<double, dim>;

    // Parameters
    double left_box       = -1.;
    double right_box      = 1.;
    std::size_t max_level = 8;
    std::size_t min_level = 2;
    double eps            = 1e-4;
    bool adapt            = false;
    double lambda         = 2.;
    double g              = 1.;
    double s2             = 1.5; // relaxation of the kinetic moment
    double hL             = 2.;  // left  water height
    double hR             = 1.;  // right water height
    double Tf             = 1.;
    std::string bc        = "bounceback"; // "bounceback" or "antibounceback"
    fs::path path         = fs::current_path();
    std::string filename  = "new_D1Q3_shallow_waters_dam";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--gravity", g, "Gravity")->capture_default_str();
    app.add_option("--s", s2, "Relaxation parameter")->capture_default_str();
    app.add_option("--hL", hL, "Left water height")->capture_default_str();
    app.add_option("--hR", hR, "Right water height")->capture_default_str();
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_option("--bc", bc, "Wall boundary condition: bounceback | antibounceback")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    app.add_option("--filename", filename, "File name prefix")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    // NON-periodic mesh (uniform if !adapt, else min_level..max_level).
    const Box box({left_box}, {right_box});
    const std::size_t ml = adapt ? min_level : max_level;
    auto config          = samurai::mesh_config<dim>().min_level(ml).max_level(max_level).periodic(false).max_stencil_size(4);
    auto mesh            = samurai::mra::make_mesh(box, config);

    const double L = right_box - left_box;

    // Fields: n_comp = 3 (D1Q3)
    auto m = samurai::make_vector_field<double, 3>("m", mesh);
    auto f = samurai::make_vector_field<double, 3>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial moments: dam break (h = hL for x < 0, hR otherwise), fluid at rest (q = 0).
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double h = (cell.center(0) < 0.) ? hL : hR;
                               m[cell](0)     = h;               // height
                               m[cell](1)     = 0.;              // momentum
                               m[cell](2)     = 0.5 * g * h * h; // kinetic moment at equilibrium (q = 0)
                           });

    // D1Q3 scheme definition
    using field_t   = decltype(f);
    const double l  = lambda;
    const double l2 = lambda * lambda;
    std::array<std::array<double, 3>, 3> M{
        {{1., 1., 1.}, {0., l, -l}, {0., l2, l2}}
    };
    std::array<std::array<double, 3>, 3> invM{
        {{1., 0., -1. / l2}, {0., 0.5 / l, 0.5 / l2}, {0., -0.5 / l, 0.5 / l2}}
    };
    auto eq = [g](std::array<double, 3>& meq, std::span<const double> mm)
    {
        const double h = mm[0];
        const double q = mm[1];
        meq[0]         = h;                           // conserved
        meq[1]         = q;                           // conserved
        meq[2]         = q * q / h + 0.5 * g * h * h; // shallow-water flux
    };

    auto scheme = samurai::make_lbm_scheme<field_t>("D1Q3_shallow_waters",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 3>(
                                                        {
                                                            {{0}, {1}, {-1}}
    },
                                                        M,
                                                        invM,
                                                        {0., 0., s2},
                                                        eq));
    scheme.set_max_level(max_level);

    // Lattice velocities (same list as the scheme), used by the wall boundary conditions.
    const std::array<std::array<int, dim>, 3> velocities{
        {{0}, {1}, {-1}}
    };
    const xt::xtensor_fixed<int, xt::xshape<dim>> left{-1};
    const xt::xtensor_fixed<int, xt::xshape<dim>> right{1};

    if (bc == "antibounceback")
    {
        // Impose the initial wall height at each wall (fluid at rest -> q = 0) via anti-bounce-back:
        // the equilibrium distribution to reflect around is built by the scheme from the wall moments.
        samurai::make_bc<samurai::AntiBounceBack>(f, velocities, scheme.equilibrium_f({hL, 0., 0.}))->on(left);
        samurai::make_bc<samurai::AntiBounceBack>(f, velocities, scheme.equilibrium_f({hR, 0., 0.}))->on(right);
    }
    else
    {
        // Reflecting solid walls on both sides.
        samurai::make_bc<samurai::BounceBack>(f, velocities);
    }

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

    // Total water mass = integral of h = f0 + f1 + f2
    auto mass = [&]()
    {
        double s = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   s += (f[cell](0) + f[cell](1) + f[cell](2)) * cell.length;
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

    double hmin = std::numeric_limits<double>::max();
    double hmax = std::numeric_limits<double>::lowest();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               hmin = std::min(hmin, m[cell](0));
                               hmax = std::max(hmax, m[cell](0));
                           });

    std::cout << "case = D1Q3 shallow-water dam, bc = " << bc << ", " << (adapt ? "adaptive" : "uniform") << ", max_level = " << max_level
              << (adapt ? (", min_level = " + std::to_string(min_level)) : "") << ", cells = " << mesh.nb_cells() << ", dt = " << dt
              << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;
    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", h in [" << hmin << ", " << hmax << "]" << std::endl;

    // Diagnostic fields for the output
    auto h = samurai::make_scalar_field<double>("h", mesh);
    auto q = samurai::make_scalar_field<double>("q", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               h[cell] = m[cell](0);
                               q[cell] = m[cell](1);
                           });
    samurai::save(path, filename, mesh, m, h, q);

    samurai::finalize();
    return 0;
}
