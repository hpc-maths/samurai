// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D2Q4444 Euler (Lax-Liu 2D Riemann problem), validating the N-D multi-block coupling of the
// schemes/lbm formalism: the 2D compressible Euler system is solved with four D2Q4 blocks
// (sixteen velocities), one per conserved variable (rho, qx = rho*ux, qy = rho*uy, E), coupled
// only through their equilibria.
//
// Each block has the four velocities {(+1,0), (0,+1), (-1,0), (0,-1)} with
//   m0 = f0+f1+f2+f3           (conserved variable of the block)
//   m1 = lambda   (f0 - f2)    (x-flux)
//   m2 = lambda   (f1 - f3)    (y-flux)
//   m3 = lambda^2 (f0-f1+f2-f3)(diagonal, relaxed towards 0)
// The four m0 are the conserved Euler variables; the m1/m2 relax towards the Euler fluxes
// (p = (gamma-1)(E - 1/2 (qx^2+qy^2)/rho)):
//   mass     flux : (qx, qy)
//   momentum flux : (rho ux^2 + p, rho ux uy) and (rho ux uy, rho uy^2 + p)
//   energy   flux : ((E+p) ux, (E+p) uy)
//
// Outflow (zero-gradient) boundaries on all four sides (homogeneous Neumann on the distributions).

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
    auto& app = samurai::initialize("D2Q4444 Euler Lax-Liu Riemann problem (schemes/lbm, multi-block)", argc, argv);

    static constexpr std::size_t dim = 2;
    using Box                        = samurai::Box<double, dim>;

    // Parameters
    std::size_t max_level = 7;
    std::size_t min_level = 2;
    double eps            = 1e-4;
    bool adapt            = false;
    double lambda         = 5.;
    double gamma          = 1.4;
    int config            = 12;  // Lax-Liu configuration (3, 11, 12, 17)
    double Tf             = -1.; // <0: use the configuration's default final time
    fs::path path         = fs::current_path();
    std::string filename  = "new_D2Q4444_euler_lax_liu";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--gamma", gamma, "Ratio of specific heats")->capture_default_str();
    app.add_option("--riemann", config, "Lax-Liu configuration: 3, 11, 12, 17")->capture_default_str();
    app.add_option("--Tf", Tf, "Final time (<0 uses the configuration default)")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    app.add_option("--filename", filename, "File name prefix")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    // Four-quadrant initial states, ordered {SW, NW, SE, NE} (x<0.5/y<0.5, x<0.5/y>=0.5, ...).
    std::array<double, 4> rho0, ux0, uy0, p0;
    double Tf_cfg;
    switch (config)
    {
        case 3:
            rho0   = {0.138, 0.5323, 0.5323, 1.5};
            ux0    = {1.206, 1.206, 0., 0.};
            uy0    = {1.206, 0., 1.206, 0.};
            p0     = {0.029, 0.3, 0.3, 1.5};
            Tf_cfg = 0.3;
            break;
        case 11:
            rho0   = {0.8, 0.5313, 0.5313, 1.};
            ux0    = {0.1, 0.8276, 0.1, 0.1};
            uy0    = {0., 0., 0.7276, 0.};
            p0     = {0.4, 0.4, 0.4, 1.};
            Tf_cfg = 0.3;
            break;
        case 17:
            rho0   = {1.0625, 2., 0.5197, 1.};
            ux0    = {0., 0., 0., 0.};
            uy0    = {0.2145, -0.3, -1.1259, 0.};
            p0     = {0.4, 1., 0.4, 1.5};
            Tf_cfg = 0.3;
            break;
        case 12:
        default:
            rho0   = {0.8, 1., 1., 0.5313};
            ux0    = {0., 0.7276, 0., 0.};
            uy0    = {0., 0., 0.7276, 0.};
            p0     = {1., 1., 1., 0.4};
            Tf_cfg = 0.25;
            break;
    }
    if (Tf < 0.)
    {
        Tf = Tf_cfg;
    }

    // Periodic-free mesh (uniform if !adapt, else min_level..max_level), domain [0,1]^2.
    const Box box({0., 0.}, {1., 1.});
    const std::size_t ml = adapt ? min_level : max_level;
    auto mesh_config     = samurai::mesh_config<dim>().min_level(ml).max_level(max_level).periodic(false).max_stencil_size(4);
    auto mesh            = samurai::mra::make_mesh(box, mesh_config);

    // Fields: n_comp = 16 (D2Q4444)
    auto m = samurai::make_vector_field<double, 16>("m", mesh);
    auto f = samurai::make_vector_field<double, 16>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial conserved moments: quadrant state -> (rho, qx, qy, E). The block layout is
    // rho @ 0, qx @ 4, qy @ 8, E @ 12; the non-conserved moments are set at equilibrium below.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double x      = cell.center(0);
                               const double y      = cell.center(1);
                               const std::size_t q = (x < 0.5) ? (y < 0.5 ? 0 : 1) : (y < 0.5 ? 2 : 3);
                               const double rho    = rho0[q];
                               const double qx     = rho * ux0[q];
                               const double qy     = rho * uy0[q];
                               const double E      = p0[q] / (gamma - 1.) + 0.5 * (qx * qx + qy * qy) / rho;
                               m[cell](0)          = rho;
                               m[cell](4)          = qx;
                               m[cell](8)          = qy;
                               m[cell](12)         = E;
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

    // Conserved moments read from the full vector: rho @ 0, qx @ 4, qy @ 8, E @ 12.
    const double gm1 = gamma - 1.;
    auto eq_rho      = [](std::array<double, 4>& meq, std::span<const double> mm)
    {
        meq[0] = mm[0]; // rho (conserved)
        meq[1] = mm[4]; // qx
        meq[2] = mm[8]; // qy
        meq[3] = 0.;
    };
    auto eq_qx = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
    {
        const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
        meq[0] = qx;                                                                              // qx (conserved)
        meq[1] = (1.5 - 0.5 * gamma) * qx * qx / r + (0.5 - 0.5 * gamma) * qy * qy / r + gm1 * E; // rho ux^2 + p
        meq[2] = qx * qy / r;                                                                     // rho ux uy
        meq[3] = 0.;
    };
    auto eq_qy = [gamma, gm1](std::array<double, 4>& meq, std::span<const double> mm)
    {
        const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
        meq[0] = qy;                                                                              // qy (conserved)
        meq[1] = qx * qy / r;                                                                     // rho ux uy
        meq[2] = (1.5 - 0.5 * gamma) * qy * qy / r + (0.5 - 0.5 * gamma) * qx * qx / r + gm1 * E; // rho uy^2 + p
        meq[3] = 0.;
    };
    auto eq_E = [gamma](std::array<double, 4>& meq, std::span<const double> mm)
    {
        const double r = mm[0], qx = mm[4], qy = mm[8], E = mm[12];
        const double h = 0.5 * (gamma - 1.);                                                           // (gamma/2 - 1/2)
        meq[0]         = E;                                                                            // E (conserved)
        meq[1]         = gamma * qx * E / r - h * qx * qx * qx / (r * r) - h * qx * qy * qy / (r * r); // (E+p) ux
        meq[2]         = gamma * qy * E / r - h * qy * qy * qy / (r * r) - h * qy * qx * qx / (r * r); // (E+p) uy
        meq[3]         = 0.;
    };

    // Relaxation: block 0 (rho) flux modes at 1.9, blocks 1-3 at 1.75, diagonal mode at 1.0.
    const std::array<double, 4> s_rho{0., 1.9, 1.9, 1.0};
    const std::array<double, 4> s_var{0., 1.75, 1.75, 1.0};

    auto scheme = samurai::make_lbm_scheme<field_t>("D2Q4444_euler",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_rho, eq_rho),
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_var, eq_qx),
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_var, eq_qy),
                                                    samurai::velocity_scheme<dim, 4>(vel, M, invM, s_var, eq_E));

    // Outflow (zero-gradient) on all four sides.
    samurai::make_bc<samurai::Neumann<1>>(f);

    scheme.init_equilibrium(f, m);

    const double dx_fine = 1. / static_cast<double>(std::size_t{1} << max_level);
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

    std::cout << "case = D2Q4444 Euler Lax-Liu config " << config << ", " << (adapt ? "adaptive" : "uniform")
              << ", max_level = " << max_level << (adapt ? (", min_level = " + std::to_string(min_level)) : "")
              << ", cells = " << mesh.nb_cells() << ", dt = " << dt << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;
    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", rho in [" << rhomin << ", " << rhomax << "], p_min = " << pmin
              << std::endl;

    // Diagnostic fields for the output
    auto rho       = samurai::make_scalar_field<double>("rho", mesh);
    auto vel_field = samurai::make_vector_field<double, 2>("velocity", mesh);
    auto p         = samurai::make_scalar_field<double>("p", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double r     = m[cell](0);
                               const double qx    = m[cell](4);
                               const double qy    = m[cell](8);
                               const double E     = m[cell](12);
                               rho[cell]          = r;
                               vel_field[cell](0) = qx / r;
                               vel_field[cell](1) = qy / r;
                               p[cell]            = (gamma - 1.) * (E - 0.5 * (qx * qx + qy * qy) / r);
                           });
    samurai::save(path, filename, mesh, m, rho, vel_field, p);

    samurai::finalize();
    return 0;
}
