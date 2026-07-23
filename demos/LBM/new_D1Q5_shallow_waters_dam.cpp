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
// The domain is NON-periodic with a zero-gradient (constant extension) boundary on both sides
// (homogeneous Neumann on the distributions f). Multiresolution adaptation is enabled by default
// (set --min-level equal to --max-level for a uniform mesh).

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

    // Simulation parameters
    double left_box  = -1.;
    double right_box = 1.;
    double lambda    = 2.;
    double g         = 1.;
    double s2        = 1.; // relaxation of the non-conserved moments (k, v, z)
    double hL        = 2.; // left  water height
    double hR        = 1.; // right water height
    double Tf        = 0.35;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "new_D1Q5_shallow_waters_dam";
    std::size_t nfiles   = 1;

    app.add_option("--left", left_box, "Left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "Right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--gravity", g, "Gravity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--s", s2, "Relaxation parameter of the non-conserved moments")->capture_default_str()->group("Simulation parameters");
    app.add_option("--hL", hL, "Left water height")->capture_default_str()->group("Simulation parameters");
    app.add_option("--hR", hR, "Right water height")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    // NON-periodic mesh. Default levels (overridden by --min-level / --max-level); min < max enables
    // multiresolution, min == max gives a uniform mesh.
    const Box box({left_box}, {right_box});
    auto config = samurai::mesh_config<dim>().min_level(4).max_level(9).periodic(false).max_stencil_size(4).graduation_width(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

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
    const double dt = mesh.min_cell_length() / lambda;

    // Save the height and velocity (and the mesh level) at a given output index.
    auto save_solution = [&](const std::string& suffix)
    {
        auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
        auto h     = samurai::make_scalar_field<double>("h", mesh);
        auto u     = samurai::make_scalar_field<double>("u", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   level[cell] = cell.level;
                                   h[cell]     = m[cell](0);
                                   u[cell]     = m[cell](1) / m[cell](0);
                               });
        samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, m, h, u, level);
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

    double hmin = std::numeric_limits<double>::max();
    double hmax = std::numeric_limits<double>::lowest();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               hmin = std::min(hmin, m[cell](0));
                               hmax = std::max(hmax, m[cell](0));
                           });
    std::cout << "cells = " << mesh.nb_cells() << ", h in [" << hmin << ", " << hmax << "]" << std::endl;

    samurai::finalize();
    return 0;
}
