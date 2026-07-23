// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
//
// D2Q9 MRT Navier-Stokes on a Taylor-Green vortex, validating the D2Q9 physics of the
// schemes/lbm formalism: a periodic domain with an exact decaying solution, no boundary
// conditions and no obstacle.
//
// Lallemand-Luo MRT moment set (as in the legacy D2Q9 demo), velocity ordering
//   0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
// Conserved moments: rho (m0), qx (m1), qy (m2). The shear modes m7, m8 relax at s_nu, which
// fixes the kinematic viscosity nu = cs2 (1/s_nu - 1/2) dt, cs2 = lambda^2 / 3.
//
// Exact solution on [0, 2 pi]^2 (k = 1):
//   u = -U0 cos(x) sin(y) exp(-2 nu t),  v =  U0 sin(x) cos(y) exp(-2 nu t)
// so the velocity L2 norm decays as exp(-2 nu t) and the kinetic energy as exp(-4 nu t).

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string>

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/lbm.hpp>

#include <filesystem>
namespace fs = std::filesystem;

// Gauss-Jordan inverse of a 9x9 matrix (used once at startup to get M = invM^{-1}).
static std::array<std::array<double, 9>, 9> inverse9(std::array<std::array<double, 9>, 9> a)
{
    std::array<std::array<double, 9>, 9> inv{};
    for (std::size_t i = 0; i < 9; ++i)
    {
        inv[i][i] = 1.;
    }
    for (std::size_t col = 0; col < 9; ++col)
    {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < 9; ++r)
        {
            if (std::abs(a[r][col]) > std::abs(a[piv][col]))
            {
                piv = r;
            }
        }
        std::swap(a[piv], a[col]);
        std::swap(inv[piv], inv[col]);
        const double d = a[col][col];
        for (std::size_t k = 0; k < 9; ++k)
        {
            a[col][k] /= d;
            inv[col][k] /= d;
        }
        for (std::size_t r = 0; r < 9; ++r)
        {
            if (r == col)
            {
                continue;
            }
            const double factor = a[r][col];
            for (std::size_t k = 0; k < 9; ++k)
            {
                a[r][k] -= factor * a[col][k];
                inv[r][k] -= factor * inv[col][k];
            }
        }
    }
    return inv;
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("D2Q9 MRT Taylor-Green vortex (schemes/lbm)", argc, argv);

    static constexpr std::size_t dim = 2;
    using Box                        = samurai::Box<double, dim>;

    std::size_t max_level = 7;
    std::size_t min_level = 3;
    double eps            = 1e-4;
    bool adapt            = false;
    double lambda         = 1.;
    double U0             = 0.05; // vortex amplitude (must stay << cs = lambda/sqrt(3))
    double rho0           = 1.;
    double nu             = 0.01; // target kinematic viscosity (held fixed across levels)
    double Tf             = 5.;
    fs::path path         = fs::current_path();
    std::string filename  = "new_D2Q9_taylor_green";

    app.add_option("--level", max_level, "Finest level")->capture_default_str();
    app.add_option("--min-lvl", min_level, "Coarsest level (adaptive)")->capture_default_str();
    app.add_flag("--adapt", adapt, "Enable multiresolution adaptation")->capture_default_str();
    app.add_option("--eps", eps, "MR adaptation threshold")->capture_default_str();
    app.add_option("--lambda", lambda, "Lattice velocity")->capture_default_str();
    app.add_option("--U0", U0, "Vortex amplitude")->capture_default_str();
    app.add_option("--nu", nu, "Target kinematic viscosity (fixes the shear relaxation per level)")->capture_default_str();
    app.add_option("--Tf", Tf, "Final time")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    app.add_option("--filename", filename, "File name prefix")->capture_default_str();
    SAMURAI_PARSE(argc, argv);

    constexpr double pi = 3.14159265358979323846;
    const double left   = 0.;
    const double right  = 2. * pi;

    const Box box({left, left}, {right, right});
    const std::size_t ml = adapt ? min_level : max_level;
    auto config          = samurai::mesh_config<dim>().min_level(ml).max_level(max_level).periodic(true).max_stencil_size(4);
    auto mesh            = samurai::mra::make_mesh(box, config);

    // Fix the physical viscosity nu across levels by choosing the shear relaxation accordingly:
    // nu = cs2 (1/s_nu - 1/2) dt  with cs2 = lambda^2/3 and dt = dx_fine / lambda.
    const double dx_fine = (right - left) / static_cast<double>(std::size_t{1} << max_level);
    const double dt      = dx_fine / lambda;
    const double cs2     = lambda * lambda / 3.;
    const double s_nu    = 1. / (0.5 + nu / (cs2 * dt));

    // Exact Taylor-Green velocity (k = 1).
    auto u_exact = [&](double x, double y, double t)
    {
        return -U0 * std::cos(x) * std::sin(y) * std::exp(-2. * nu * t);
    };
    auto v_exact = [&](double x, double y, double t)
    {
        return U0 * std::sin(x) * std::cos(y) * std::exp(-2. * nu * t);
    };

    auto m = samurai::make_vector_field<double, 9>("m", mesh);
    auto f = samurai::make_vector_field<double, 9>("f", mesh);
    m.fill(0.);
    f.fill(0.);

    // Initial moments: rho = rho0, momentum = rho0 * (u, v).
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double x = cell.center(0);
                               const double y = cell.center(1);
                               m[cell](0)     = rho0;
                               m[cell](1)     = rho0 * u_exact(x, y, 0.);
                               m[cell](2)     = rho0 * v_exact(x, y, 0.);
                           });

    // Lallemand-Luo m -> f matrix (invM), then M = invM^{-1}.
    const double r1 = 1. / lambda;
    const double r2 = 1. / (lambda * lambda);
    const double r3 = 1. / (lambda * lambda * lambda);
    const double r4 = 1. / (lambda * lambda * lambda * lambda);
    // Columns: (rho, qx, qy, e, qx-flux, qy-flux, eps, pxx, pxy)
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
    const auto M = inverse9(invM);

    const double l2 = lambda * lambda;
    const double l4 = l2 * l2;
    auto eq         = [l2, l4](std::array<double, 9>& meq, std::span<const double> mm)
    {
        const double rho = mm[0];
        const double qx  = mm[1];
        const double qy  = mm[2];
        const double q2  = (qx * qx + qy * qy) / rho;
        meq[0]           = rho;                       // conserved
        meq[1]           = qx;                        // conserved
        meq[2]           = qy;                        // conserved
        meq[3]           = -2. * l2 * rho + 3. * q2;  // e
        meq[4]           = -l2 * qx;                  // energy flux x
        meq[5]           = -l2 * qy;                  // energy flux y
        meq[6]           = l4 * rho - 3. * l2 * q2;   // epsilon
        meq[7]           = (qx * qx - qy * qy) / rho; // pxx
        meq[8]           = qx * qy / rho;             // pxy
    };

    // Relaxation rates: conserved (rho, qx, qy) frozen; standard MRT values for the ghost/bulk
    // modes; the shear modes (pxx, pxy) at s_nu, which fixes the viscosity.
    std::array<double, 9> s{0., 0., 0., 1.64, 1.54, 1.54, 1.64, s_nu, s_nu};

    using field_t = decltype(f);
    auto scheme   = samurai::make_lbm_scheme<field_t>("D2Q9_taylor_green",
                                                    lambda,
                                                    samurai::velocity_scheme<dim, 9>(
                                                        {
                                                            {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
    },
                                                        M,
                                                        invM,
                                                        s,
                                                        eq));
    scheme.set_max_level(max_level);
    scheme.init_equilibrium(f, m);

    const auto nt       = static_cast<std::size_t>(std::round(Tf / dt));
    const double Tf_eff = static_cast<double>(nt) * dt;

    auto MRadaptation = samurai::make_MRAdapt(f);
    auto mra_config   = samurai::mra_config().epsilon(eps);
    if (adapt)
    {
        MRadaptation(mra_config);
        m.resize();
    }

    auto mass = [&]()
    {
        double s_ = 0.;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   s_ += m[cell](0) * cell.length * cell.length;
                               });
        return s_;
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

    // Velocity error vs the exact Taylor-Green solution, area-weighted.
    double err2 = 0.;
    double nrm2 = 0.;
    double umax = 0.;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double area = cell.length * cell.length;
                               const double rho  = m[cell](0);
                               const double u    = m[cell](1) / rho;
                               const double v    = m[cell](2) / rho;
                               const double ue   = u_exact(cell.center(0), cell.center(1), Tf_eff);
                               const double ve   = v_exact(cell.center(0), cell.center(1), Tf_eff);
                               err2 += ((u - ue) * (u - ue) + (v - ve) * (v - ve)) * area;
                               nrm2 += (ue * ue + ve * ve) * area;
                               umax = std::max(umax, std::sqrt(u * u + v * v));
                           });
    const double err_l2 = std::sqrt(err2 / nrm2);

    std::cout << "case = D2Q9 Taylor-Green, " << (adapt ? "adaptive" : "uniform") << ", max_level = " << max_level
              << (adapt ? (", min_level = " + std::to_string(min_level)) : "") << ", cells = " << mesh.nb_cells() << ", nu = " << nu
              << ", dt = " << dt << ", nt = " << nt << ", Tf_eff = " << Tf_eff << std::endl;
    std::cout << "mass drift = " << std::abs(mass() - mass0) << ", |u|max = " << umax
              << " (exact decay factor exp(-2 nu Tf) = " << std::exp(-2. * nu * Tf_eff) << ")" << std::endl;
    std::cout << "relative L2 velocity error = " << err_l2 << std::endl;

    // Diagnostic fields
    auto vel = samurai::make_vector_field<double, 2>("velocity", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               vel[cell](0) = m[cell](1) / m[cell](0);
                               vel[cell](1) = m[cell](2) / m[cell](0);
                           });
    samurai::save(path, filename, mesh, m, vel);

    samurai::finalize();
    return 0;
}
