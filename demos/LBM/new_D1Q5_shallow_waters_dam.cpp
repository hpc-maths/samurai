// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D1Q5 shallow-water dam break, validating the |c| > 1 streaming of the schemes/lbm formalism
// in 1D (one block of 5 velocities {0, +1, -1, +2, -2}).
//
//   m0 = h  = f0 + f1 + f2 + f3 + f4                         (conserved, water height)
//   m1 = q  = lambda   (f1 - f2 + 2 f3 - 2 f4)               (conserved, momentum)
//   m2 = k  = lambda^2 (f1 + f2 + 4 f3 + 4 f4)               (relaxed towards q^2/h + 1/2 g h^2)
//   m3 = v  = lambda^3 (f1 - f2 + 8 f3 - 8 f4)               (relaxed towards q lambda^2)
//   m4 = z  = lambda^4 (f1 + f2 + 16 f3 + 16 f4)             (relaxed towards (q^2/h + 1/2 g h^2) lambda^2)
//
// The domain is NON-periodic with a zero-gradient (constant extension) boundary on both sides,
// realised as a homogeneous Neumann condition on the distributions f; the dam-break waves leave
// the domain (open ends) so the water mass is not exactly conserved once a wave reaches a border,
// but it stays bounded and positive. This is the 1D counterpart of the 2D diagonal / |c| > 1
// streaming validation (see new_D2Q4diag_advection).

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
    auto& app = samurai::initialize("D1Q5 shallow-water dam break (schemes/lbm, |c| > 1 streaming)", argc, argv);

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
    double s2             = 1.; // relaxation of the non-conserved moments (k, v, z)
    double hL             = 2.; // left  water height
    double hR             = 1.; // right water height
    double Tf             = 0.35;
    fs::path path         = fs::current_path();
    std::string filename  = "new_D1Q5_shallow_waters_dam";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--gravity", g, "Gravity")->capture_default_str();
    app.add_option("--s", s2, "Relaxation parameter of the non-conserved moments")->capture_default_str();
    app.add_option("--hL", hL, "Left water height")->capture_default_str();
    app.add_option("--hR", hR, "Right water height")->capture_default_str();
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

    // Fields: n_comp = 5 (D1Q5)
    auto m = samurai::make_vector_field<double, 5>("m", mesh);
    auto f = samurai::make_vector_field<double, 5>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial moments: dam break (h = hL for x < 0, hR otherwise), fluid at rest (q = 0).
    // The non-conserved moments are filled at equilibrium by init_equilibrium below.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               m[cell](0) = (cell.center(0) < 0.) ? hL : hR; // height
                               m[cell](1) = 0.;                              // momentum
                           });

    // D1Q5 scheme definition
    using field_t   = decltype(f);
    const double l  = lambda;
    const double l2 = lambda * lambda;
    const double l3 = l2 * lambda;
    const double l4 = l2 * l2;
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
        const double h    = mm[0];
        const double q    = mm[1];
        const double flux = q * q / h + 0.5 * g * h * h; // shallow-water flux
        meq[0]            = h;                           // conserved
        meq[1]            = q;                           // conserved
        meq[2]            = flux;                        // k
        meq[3]            = q * l2;                      // v
        meq[4]            = flux * l2;                   // z
    };

    auto scheme = samurai::make_lbm_scheme<field_t>("D1Q5_shallow_waters",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 5>(
                                                        {
                                                            {{0}, {1}, {-1}, {2}, {-2}}
    },
                                                        M,
                                                        invM,
                                                        {0., 0., s2, s2, s2},
                                                        eq));

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

    std::cout << "case = D1Q5 shallow-water dam, " << (adapt ? "adaptive" : "uniform") << ", max_level = " << max_level
              << (adapt ? (", min_level = " + std::to_string(min_level)) : "") << ", cells = " << mesh.nb_cells() << ", dt = " << dt
              << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;
    std::cout << "h in [" << hmin << ", " << hmax << "]" << std::endl;

    // Diagnostic fields for the output
    auto h = samurai::make_scalar_field<double>("h", mesh);
    auto u = samurai::make_scalar_field<double>("u", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               h[cell] = m[cell](0);
                               u[cell] = m[cell](1) / m[cell](0);
                           });
    samurai::save(path, filename, mesh, m, h, u);

    samurai::finalize();
    return 0;
}
