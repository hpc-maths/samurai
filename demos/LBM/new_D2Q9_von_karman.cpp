// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D2Q9 MRT Navier-Stokes, flow past a circular cylinder (von Karman vortex street), validating
// the LBM inflow / outflow boundary conditions and an immersed obstacle of the schemes/lbm
// formalism:
//   - inflow (left, top, bottom): a fixed free-stream equilibrium distribution imposed in the
//     ghosts (make_bc<ImposedDistribution>), i.e. a uniform horizontal stream u0;
//   - outflow (right): homogeneous Neumann (zero-gradient) on the distributions;
//   - cylinder: volume penalisation towards the rest equilibrium, blended by the fraction of
//     each cell inside the disk (a smooth immersed boundary), applied after every collision.
// The drag and lift coefficients are recovered from the momentum absorbed by the penalisation.
//
// The D2Q9 Lallemand-Luo moment set is the same as new_D2Q9_taylor_green; the shear relaxation
// fixes the kinematic viscosity nu = u0 D / Re (D = cylinder diameter).

#include <array>
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
    auto& app = samurai::initialize("D2Q9 MRT Navier-Stokes, von Karman street past a cylinder (schemes/lbm)", argc, argv);

    static constexpr std::size_t dim = 2;
    using Box                        = samurai::Box<double, dim>;

    // Simulation parameters
    double lambda = 1.;
    double rho0   = 1.;
    double u0     = 0.1;  // free-stream velocity (<< cs = lambda/sqrt(3))
    double Re     = 100.; // Reynolds number on the cylinder diameter
    double radius = 1. / 16.;
    double cx     = 0.5; // cylinder centre
    double cy     = 0.5;
    double Tf     = 20.;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "new_D2Q9_von_karman";
    std::size_t nfiles   = 1;

    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--u0", u0, "Free-stream velocity")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Re", Re, "Reynolds number (on the diameter)")->capture_default_str()->group("Simulation parameters");
    app.add_option("--radius", radius, "Cylinder radius")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    // Channel [0,2] x [0,1], non-periodic.
    const Box box({0., 0.}, {2., 1.});
    auto config = samurai::mesh_config<dim>().min_level(4).max_level(7).periodic(false).max_stencil_size(4).graduation_width(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

    const double dt   = mesh.min_cell_length() / lambda;
    const double cs2  = lambda * lambda / 3.;
    const double diam = 2. * radius;
    const double nu   = u0 * diam / Re;               // kinematic viscosity from the Reynolds number
    const double s_nu = 1. / (0.5 + nu / (cs2 * dt)); // shear relaxation fixing nu

    // Fraction of a cell inside the cylinder, by n x n subsampling (0 outside, 1 fully inside).
    auto solid_fraction = [&](double xc, double yc, double h)
    {
        constexpr int n = 8;
        int inside      = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                const double xs = xc - 0.5 * h + (i + 0.5) * h / n;
                const double ys = yc - 0.5 * h + (j + 0.5) * h / n;
                if ((xs - cx) * (xs - cx) + (ys - cy) * (ys - cy) <= radius * radius)
                {
                    ++inside;
                }
            }
        }
        return static_cast<double>(inside) / static_cast<double>(n * n);
    };

    auto m = samurai::make_vector_field<double, 9>("m", mesh);
    auto f = samurai::make_vector_field<double, 9>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial moments: uniform stream u0 outside the cylinder (with a tiny transverse perturbation
    // to trigger shedding), fluid at rest inside.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double frac = solid_fraction(cell.center(0), cell.center(1), cell.length);
                               m[cell](0)        = rho0;
                               m[cell](1)        = (1. - frac) * rho0 * u0;
                               m[cell](2)        = (1. - frac) * 0.01 * rho0 * u0;
                           });

    // D2Q9 Lallemand-Luo moment matrices (see new_D2Q9_taylor_green for the derivation).
    using field_t   = decltype(f);
    const double l  = lambda;
    const double l2 = lambda * lambda;
    const double l3 = l2 * lambda;
    const double l4 = l2 * l2;
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
    const double r1 = 1. / lambda;
    const double r2 = 1. / (lambda * lambda);
    const double r3 = 1. / (lambda * lambda * lambda);
    const double r4 = 1. / (lambda * lambda * lambda * lambda);
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
    auto eq = [l2, l4](std::array<double, 9>& meq, std::span<const double> mm)
    {
        const double rho = mm[0];
        const double qx  = mm[1];
        const double qy  = mm[2];
        const double q2  = (qx * qx + qy * qy) / rho;
        meq[0]           = rho;
        meq[1]           = qx;
        meq[2]           = qy;
        meq[3]           = -2. * l2 * rho + 3. * q2;
        meq[4]           = -l2 * qx;
        meq[5]           = -l2 * qy;
        meq[6]           = l4 * rho - 3. * l2 * q2;
        meq[7]           = (qx * qx - qy * qy) / rho;
        meq[8]           = qx * qy / rho;
    };
    std::array<double, 9> s{0., 0., 0., 1.64, 1.54, 1.54, 1.64, s_nu, s_nu};

    auto scheme = samurai::make_lbm_scheme<field_t>("D2Q9_von_karman",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 9>(
                                                        {
                                                            {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
    },
                                                        M,
                                                        invM,
                                                        s,
                                                        eq));

    // Free-stream inflow (left, top, bottom) and rest equilibria for the obstacle.
    const std::array<double, 9> feq_inflow = scheme.equilibrium_f({rho0, rho0 * u0, 0., 0., 0., 0., 0., 0., 0.});
    const std::array<double, 9> feq_rest   = scheme.equilibrium_f({rho0, 0., 0., 0., 0., 0., 0., 0., 0.});

    const xt::xtensor_fixed<int, xt::xshape<dim>> left{-1, 0};
    const xt::xtensor_fixed<int, xt::xshape<dim>> right{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<dim>> top{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<dim>> bottom{0, -1};
    samurai::make_bc<samurai::ImposedDistribution>(f, feq_inflow)->on(left, top, bottom);
    samurai::make_bc<samurai::Neumann<1>>(f)->on(right);

    scheme.init_equilibrium(f, m);

    // Volume-penalisation of the cylinder: relax the distributions of the solid cells to the rest
    // equilibrium, blended by the solid fraction. Returns the drag/lift coefficients from the
    // momentum absorbed this step.
    auto penalise = [&]()
    {
        double Fx = 0., Fy = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   const double frac = solid_fraction(cell.center(0), cell.center(1), cell.length);
                                   if (frac == 0.)
                                   {
                                       return;
                                   }
                                   auto fc = f[cell];
                                   // Momentum removed by the penalisation = force exerted on the solid.
                                   double qx = 0., qy = 0.;
                                   for (std::size_t a = 0; a < 9; ++a)
                                   {
                                       qx += M[1][a] * fc(a);
                                       qy += M[2][a] * fc(a);
                                   }
                                   const double area = cell.length * cell.length;
                                   Fx += frac * qx / dt * area;
                                   Fy += frac * qy / dt * area;
                                   for (std::size_t a = 0; a < 9; ++a)
                                   {
                                       fc(a) = (1. - frac) * fc(a) + frac * feq_rest[a];
                                   }
                               });
        const double norm = 0.5 * rho0 * u0 * u0 * diam;
        return std::array<double, 2>{Fx / norm, Fy / norm};
    };

    // Save the level, the diagnostics (rho, velocity) and the solid fraction at a given output
    // index. The moments are refreshed from the distributions first, since the penalisation
    // modifies f after the last collision.
    auto save_solution = [&](const std::string& suffix)
    {
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   auto fc    = f[cell];
                                   double rho = 0., qx = 0., qy = 0.;
                                   for (std::size_t a = 0; a < 9; ++a)
                                   {
                                       rho += fc(a);
                                       qx += M[1][a] * fc(a);
                                       qy += M[2][a] * fc(a);
                                   }
                                   m[cell](0) = rho;
                                   m[cell](1) = qx;
                                   m[cell](2) = qy;
                               });

        auto level    = samurai::make_scalar_field<std::size_t>("level", mesh);
        auto rho      = samurai::make_scalar_field<double>("rho", mesh);
        auto velocity = samurai::make_vector_field<double, 2>("velocity", mesh);
        auto solid    = samurai::make_scalar_field<double>("solid", mesh);
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   level[cell]       = cell.level;
                                   rho[cell]         = m[cell](0);
                                   velocity[cell](0) = m[cell](1) / m[cell](0);
                                   velocity[cell](1) = m[cell](2) / m[cell](0);
                                   solid[cell]       = solid_fraction(cell.center(0), cell.center(1), cell.length);
                               });
        samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, m, rho, velocity, solid, level);
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
    std::array<double, 2> cdl{0., 0.};
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
        cdl = penalise();

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save_solution(suffix);
        }
    }

    // Recompute the moments from the penalised distributions for the final diagnostics.
    double umax = 0., u_solid_max = 0.;
    double rhomin = std::numeric_limits<double>::max(), rhomax = std::numeric_limits<double>::lowest();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               auto fc    = f[cell];
                               double rho = 0., qx = 0., qy = 0.;
                               for (std::size_t a = 0; a < 9; ++a)
                               {
                                   rho += fc(a);
                                   qx += M[1][a] * fc(a);
                                   qy += M[2][a] * fc(a);
                               }
                               m[cell](0)         = rho;
                               m[cell](1)         = qx;
                               m[cell](2)         = qy;
                               const double speed = std::sqrt(qx * qx + qy * qy) / rho;
                               rhomin             = std::min(rhomin, rho);
                               rhomax             = std::max(rhomax, rho);
                               umax               = std::max(umax, speed);
                               if (solid_fraction(cell.center(0), cell.center(1), cell.length) == 1.)
                               {
                                   u_solid_max = std::max(u_solid_max, speed); // penalised cells: should be ~0
                               }
                           });

    std::cout << "cells = " << mesh.nb_cells() << ", Re = " << Re << ", nu = " << nu << ", s_nu = " << s_nu << ", dt = " << dt << std::endl;
    std::cout << "rho in [" << rhomin << ", " << rhomax << "], |u|max = " << umax << ", |u|max in solid = " << u_solid_max
              << ", Cd = " << cdl[0] << ", Cl = " << cdl[1] << std::endl;

    samurai::finalize();
    return 0;
}
