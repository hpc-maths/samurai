// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "CLI/CLI.hpp"
#include <iostream>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>
#include <samurai/reconstruction.hpp>

static constexpr double pi = M_PI;

template <class Field>
bool check_nan_or_inf(const Field& f)
{
    std::size_t n      = f.mesh().nb_cells();
    bool is_nan_or_inf = false;
    for (std::size_t i = 0; i < n * Field::size; ++i)
    {
        double value = f.array().data()[i];
        if (std::isnan(value) || std::isinf(value) || (abs(value) < 1e-300 && abs(value) != 0))
        {
            is_nan_or_inf = true;
            std::cout << f << std::endl;
            break;
        }
    }
    return !is_nan_or_inf;
}

//
// Configuration of the PETSc solver for the Stokes problem
//
template <class Solver>
void configure_saddle_point_solver(Solver& block_solver)
{
    // The matrix has the saddle-point structure
    //           | A    B |
    //           | B^T  C |

    // The Schur complement eliminating the first variable (here, the velocity) is
    //            Schur = C - B^T * A^-1 * B
    // We define the preconditioner
    //            S = C - B^T * ksp(A) * B
    // where ksp(A) is a solver for A.

    KSP ksp = block_solver.Ksp();
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCFIELDSPLIT);                 // (equiv. '-pc_type fieldsplit')
    PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR); // Schur complement preconditioner (equiv. '-pc_fieldsplit_type schur')
    PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, PETSC_NULL); // (equiv. '-pc_fieldsplit_schur_precondition selfp')
    PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_FULL);        // (equiv. '-pc_fieldsplit_schur_fact_type full')

    // Configure the sub-solvers
    block_solver.setup(); // must be called before using PCFieldSplitSchurGetSubKSP(), because the matrices are needed.
    KSP* sub_ksp;
    PCFieldSplitSchurGetSubKSP(pc, nullptr, &sub_ksp);
    KSP A_ksp     = sub_ksp[0];
    KSP schur_ksp = sub_ksp[1];

    // Set LU by default for the A block (diffusion). Consider using 'hypre' for large problems,
    // using the option '-fieldsplit_velocity_[np1]_pc_type hypre'.
    PC A_pc;
    KSPGetPC(A_ksp, &A_pc);
    PCSetType(A_pc, PCLU);    // (equiv. '-fieldsplit_velocity_[np1]_pc_type lu')
    KSPSetFromOptions(A_ksp); // KSP and PC overwritten by user value if needed

    PC schur_pc;
    KSPGetPC(schur_ksp, &schur_pc);
    PCSetType(schur_pc, PCNONE);  // (equiv. '-fieldsplit_pressure_[np1]_pc_type none')
    KSPSetFromOptions(schur_ksp); // KSP and PC overwritten by user value if needed

    // If a tolerance is set by the user ('-ksp-rtol XXX'), then we set that
    // tolerance to all the sub-solvers
    PetscReal ksp_rtol;
    KSPGetTolerances(ksp, &ksp_rtol, PETSC_NULL, PETSC_NULL, PETSC_NULL);
    KSPSetTolerances(A_ksp, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);     // (equiv. '-fieldsplit_velocity_ksp_rtol XXX')
    KSPSetTolerances(schur_ksp, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); // (equiv. '-fieldsplit_pressure_ksp_rtol XXX')
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    // using Config = samurai::amr::Config<dim>;
    // using Mesh = samurai::amr::Mesh<Config>;
    using Config          = samurai::MRConfig<dim, 1>;
    using Mesh            = samurai::MRMesh<Config>;
    using mesh_id_t       = typename Mesh::mesh_id_t;
    constexpr bool is_soa = false;

    //----------------//
    //   Parameters   //
    //----------------//

    std::string test_case = "ns";

    std::size_t min_level = 2;
    std::size_t max_level = 2;
    double Tf             = 1.;
    double dt             = Tf / 100;

    double mr_epsilon    = 1e-1; // Threshold used by multiresolution
    double mr_regularity = 3;    // Regularity guess for multiresolution
    std::size_t nfiles   = 50;

    fs::path path        = fs::current_path();
    std::string filename = "velocity";

    CLI::App app{"Stokes problem"};
    app.add_option("--test-case", test_case, "Test case (s = stationary, ns = non-stationary, ldc = lid-driven cavity)")
        ->capture_default_str()
        ->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    PetscInitialize(&argc, &argv, 0, nullptr);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
    PetscOptionsSetValue(NULL, "-options_left", "off"); // disable warning for unused options

    auto box = samurai::Box<double, dim>({0, 0}, {1, 1});
    // auto mesh = Mesh(box, start_level, min_level, max_level); // amr::Mesh
    auto mesh = Mesh(box, static_cast<std::size_t>(min_level), static_cast<std::size_t>(max_level)); // MRMesh

    //--------------------//
    // Stationary problem //
    //--------------------//

    if (test_case == "s")
    {
        // 2 equations: -Lap(v) + Grad(p) = f
        //              -Div(v)           = 0
        // where v = velocity
        //       p = pressure

        // Unknowns
        auto velocity = samurai::make_field<double, dim, is_soa>("velocity", mesh);
        auto pressure = samurai::make_field<double, 1, is_soa>("pressure", mesh);

        // Boundary conditions
        samurai::make_bc<samurai::Dirichlet>(velocity,
                                             [](const auto& coord)
                                             {
                                                 const auto& x = coord[0];
                                                 const auto& y = coord[1];
                                                 double v_x    = 1 / (pi * pi) * sin(pi * (x + y));
                                                 double v_y    = -v_x;
                                                 return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                                             });

        samurai::make_bc<samurai::Neumann>(pressure,
                                           [](const auto& coord)
                                           {
                                               const auto& x = coord[0];
                                               const auto& y = coord[1];
                                               int normal    = (x == 0 || y == 0) ? -1 : 1;
                                               return normal * (1 / pi) * cos(pi * (x + y));
                                           });

        // clang-format off

        // Stokes operator
        //             |  Diff  Grad |
        //             | -Div     0  |
        auto diff_v = samurai::petsc::make_diffusion_FV(velocity);
        auto grad_p = samurai::petsc::make_gradient_FV(pressure);
        auto div_v  = samurai::petsc::make_divergence_FV(velocity);
        auto zero_p = samurai::petsc::make_zero_operator_FV<1>(pressure);

        auto stokes = samurai::petsc::make_block_operator<2, 2>(diff_v, grad_p,
                                                                -div_v, zero_p);

        // clang-format on

        // Right-hand side
        auto f    = samurai::make_field<double, dim, is_soa>("f",
                                                          mesh,
                                                          [](const auto& coord)
                                                          {
                                                              const auto& x = coord[0];
                                                              const auto& y = coord[1];
                                                              double f_x    = 2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
                                                              double f_y    = -2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
                                                              return xt::xtensor_fixed<double, xt::xshape<dim>>{f_x, f_y};
                                                          });
        auto zero = samurai::make_field<double, 1, is_soa>("zero", mesh);
        zero.fill(0);

        // Linear solver
        std::cout << "Solving Stokes system..." << std::endl;
        auto stokes_solver = samurai::petsc::make_solver(stokes);
        configure_saddle_point_solver(stokes_solver);
        stokes_solver.solve(f, zero);
        std::cout << stokes_solver.iterations() << " iterations" << std::endl << std::endl;

        // Error
        double error = L2_error(velocity,
                                [](auto& coord)
                                {
                                    const auto& x = coord[0];
                                    const auto& y = coord[1];
                                    auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                                    auto v_y      = -v_x;
                                    return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                                });
        std::cout.precision(2);
        std::cout << "L2-error on the velocity: " << std::scientific << error << std::endl;

        // Save solution
        bool save_solution = false;
        if (save_solution)
        {
            std::cout << "Saving solution..." << std::endl;

            samurai::save(path, filename, mesh, velocity);
            samurai::save(path, "pressure", mesh, pressure);

            auto exact_velocity = samurai::make_field<double, dim, is_soa>("exact_velocity",
                                                                           mesh,
                                                                           [](const auto& coord)
                                                                           {
                                                                               const auto& x = coord[0];
                                                                               const auto& y = coord[1];
                                                                               auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                                                                               auto v_y      = -v_x;
                                                                               return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                                                                           });
            samurai::save(path, "exact_velocity", mesh, exact_velocity);

            /*auto err = samurai::make_field<double, dim, is_soa>("error", mesh);
            for_each_cell(err.mesh(), [&](const auto& cell)
                {
                    err[cell] = exact_velocity[cell] - velocity[cell];
                });
            samurai::save(path, "error_velocity", mesh, err);*/

            auto exact_pressure = samurai::make_field<double, 1, is_soa>("exact_pressure",
                                                                         mesh,
                                                                         [](const auto& coord)
                                                                         {
                                                                             const auto& x = coord[0];
                                                                             const auto& y = coord[1];
                                                                             return 1 / (pi * pi) * sin(pi * (x + y));
                                                                         });
            samurai::save(path, "exact_pressure", mesh, exact_pressure);
        }
    }

    //------------------------//
    // Non stationary problem //
    //------------------------//

    else if (test_case == "ns")
    {
        // Equations:
        //              v_np1 + dt * (-diff_coeff*Lap(v_np1) + Grad(p_np1)) = dt*f_n + v_n
        //                                        Div(v_np1)                = 0
        // where v = velocity
        //       p = pressure

        double diff_coeff = 0.1;
        auto analytic_f   = [&](double t, const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            double f_x    = (cos(t) + 8 * diff_coeff * pi * pi * sin(t)) * sin(2 * pi * x) * cos(2 * pi * y)
                       - pi * sin(t) * sin(t) * sin(pi * x) * sin(pi * y);
            double f_y = -(cos(t) + 8 * diff_coeff * pi * pi * sin(t)) * cos(2 * pi * x) * sin(2 * pi * y)
                       + pi * sin(t) * sin(t) * cos(pi * x) * cos(pi * y);
            return xt::xtensor_fixed<double, xt::xshape<dim>>{f_x, f_y};
        };

        // Exact solution
        auto exact_velocity = [&](double t, const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            double v_x    = std::sin(t) * std::sin(2 * pi * x) * std::cos(2 * pi * y);
            double v_y    = -std::sin(t) * std::cos(2 * pi * x) * std::sin(2 * pi * y);
            return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
        };
        auto exact_normal_grad_pressure = [&](double t, const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            int normal    = (x == 0 || y == 0) ? -1 : 1;
            return normal * (-pi * std::sin(t) * std::sin(t) * std::sin(pi * x) * std::sin(pi * y));
        };

        // Unknowns
        auto velocity     = samurai::make_field<double, dim, is_soa>("velocity", mesh);
        auto velocity_np1 = samurai::make_field<double, dim, is_soa>("velocity_np1", mesh);
        auto pressure_np1 = samurai::make_field<double, 1, is_soa>("pressure_np1", mesh);
        // Right-hand side
        auto rhs  = samurai::make_field<double, dim, is_soa>("rhs", mesh);
        auto zero = samurai::make_field<double, 1, is_soa>("zero", mesh);

        // Boundary conditions
        samurai::make_bc<samurai::Dirichlet>(velocity_np1,
                                             [&](const auto& coord)
                                             {
                                                 return exact_velocity(0, coord);
                                             });
        samurai::make_bc<samurai::Neumann>(pressure_np1,
                                           [&](const auto& coord)
                                           {
                                               return exact_normal_grad_pressure(0, coord);
                                           });

        // clang-format off

        // Stokes operator
        //             |  Diff  Grad |
        //             | -Div     0  |
        auto diff_v = diff_coeff * samurai::petsc::make_diffusion_FV(velocity_np1);
        auto grad_p =              samurai::petsc::make_gradient_FV(pressure_np1);
        auto div_v  =              samurai::petsc::make_divergence_FV(velocity_np1);
        auto zero_p =              samurai::petsc::make_zero_operator_FV<1>(pressure_np1);
        auto id_v   =              samurai::petsc::make_identity_FV(velocity_np1);

        // Stokes with backward Euler
        //             | I + dt*Diff    dt*Grad |
        //             |       -Div        0    |
        auto stokes = samurai::petsc::make_block_operator<2, 2>(id_v + dt * diff_v, dt * grad_p,
                                                                            -div_v,      zero_p);
        // clang-format on

        // Linear solver
        auto stokes_solver = samurai::petsc::make_solver(stokes);
        configure_saddle_point_solver(stokes_solver);

        // Initial condition
        velocity.fill(0);

        velocity_np1.fill(0);
        pressure_np1.fill(0);

        // Time iteration
        auto MRadaptation = samurai::make_MRAdapt(velocity);

        samurai::save(path, fmt::format("{}{}", filename, "_init"), mesh, velocity);
        std::size_t nsave = 1, nt = 0;

        bool mesh_has_changed = false;
        bool dt_has_changed   = false;

        double t_n   = 0;
        double t_np1 = 0;
        while (t_np1 != Tf)
        {
            // Move to next timestep
            t_np1 += dt;
            if (t_np1 > Tf)
            {
                dt += Tf - t_np1;
                t_np1          = Tf;
                dt_has_changed = true;
            }
            std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t_np1, dt);

            if (min_level != max_level)
            {
                // Mesh adaptation
                MRadaptation(mr_epsilon, mr_regularity);
                mesh_has_changed = true;

                velocity_np1.resize();
                pressure_np1.resize();
                rhs.resize();
                zero.resize();
                std::cout << ", levels " << mesh[mesh_id_t::cells].min_level() << "-" << mesh[mesh_id_t::cells].max_level();
            }
            std::cout.flush();

            if (mesh_has_changed || dt_has_changed)
            {
                stokes_solver.reset();
                configure_saddle_point_solver(stokes_solver);
            }

            // Boundary conditions
            velocity_np1.get_bc().clear();
            samurai::make_bc<samurai::Dirichlet>(velocity_np1,
                                                 [&](const auto& coord)
                                                 {
                                                     return exact_velocity(t_np1, coord);
                                                 });
            pressure_np1.get_bc().clear();
            samurai::make_bc<samurai::Neumann>(pressure_np1,
                                               [&](const auto& coord)
                                               {
                                                   return exact_normal_grad_pressure(t_np1, coord);
                                               });

            // Solve the linear equation
            //                [I + dt*Diff] v_np1 + dt*p_np1 = v_n + dt*f
            //                         -Div v_np1            = 0
            auto f = samurai::make_field<double, dim, is_soa>("f",
                                                              mesh,
                                                              [&](const auto& coord)
                                                              {
                                                                  return analytic_f(t_n, coord);
                                                              });
            rhs.fill(0);
            rhs = velocity + dt * f;
            zero.fill(0);
            stokes_solver.solve(rhs, zero);

            // Prepare next step
            std::swap(velocity.array(), velocity_np1.array());
            t_n = t_np1;

            // Error
            double error = L2_error(velocity,
                                    [&](auto& coord)
                                    {
                                        return exact_velocity(t_n, coord);
                                    });
            std::cout.precision(2);
            std::cout << ", L2-error: " << std::scientific << error;

            // Save the result
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            samurai::save(path, fmt::format("{}{}", filename, suffix), velocity.mesh(), velocity);

            if (min_level != max_level)
            {
                // Reconstruction on the finest level
                samurai::update_ghost_mr(velocity);
                auto velocity_recons = samurai::reconstruction(velocity);
                // Error
                double error_recons = L2_error(velocity_recons,
                                               [&](auto& coord)
                                               {
                                                   return exact_velocity(t_n, coord);
                                               });
                std::cout.precision(2);
                std::cout << ", L2-error (recons): " << std::scientific << error_recons;
                // Save
                samurai::save(path, fmt::format("{}_recons{}", filename, suffix), velocity_recons.mesh(), velocity_recons);
            }
            std::cout << std::endl;
        }
    }

    //------------------------//
    //   Lid-driven cavity    //
    //------------------------//

    else if (test_case == "ldc")
    {
        // 2 equations: v_np1 + dt * (-diff_coeff*Lap(v_np1) + Grad(p_np1)) = v_n
        //                                        Div(v_np1)                = 0
        // where v = velocity
        //       p = pressure

        double diff_coeff = 1. / 100;

        // Unknowns
        auto velocity     = samurai::make_field<double, dim, is_soa>("velocity", mesh);
        auto velocity_np1 = samurai::make_field<double, dim, is_soa>("velocity_np1", mesh);
        auto pressure_np1 = samurai::make_field<double, 1, is_soa>("pressure_np1", mesh);
        auto zero         = samurai::make_field<double, 1, is_soa>("zero", mesh);
        zero.fill(0);

        // Boundary conditions (n)
        samurai::DirectionVector<dim> left   = {-1, 0};
        samurai::DirectionVector<dim> right  = {1, 0};
        samurai::DirectionVector<dim> bottom = {0, -1};
        samurai::DirectionVector<dim> top    = {0, 1};
        samurai::make_bc<samurai::Dirichlet>(velocity, 1., 0.)->on(top);
        samurai::make_bc<samurai::Dirichlet>(velocity, 0., 0.)->on(left, bottom, right);

        // Boundary conditions (n+1)
        samurai::make_bc<samurai::Dirichlet>(velocity_np1, 1., 0.)->on(top);
        samurai::make_bc<samurai::Dirichlet>(velocity_np1, 0., 0.)->on(left, bottom, right);

        samurai::make_bc<samurai::Neumann>(pressure_np1, 0.);

        // Initial condition
        velocity.fill(0);

        velocity_np1.fill(0);
        pressure_np1.fill(0);

        // clang-format off

        // Stokes operator
        //             |  Diff  Grad |
        //             | -Div     0  |
        auto diff_v = diff_coeff * samurai::petsc::make_diffusion_FV(velocity_np1);
        auto grad_p =              samurai::petsc::make_gradient_FV(pressure_np1);
        auto div_v  =              samurai::petsc::make_divergence_FV(velocity_np1);
        auto zero_p =              samurai::petsc::make_zero_operator_FV<1>(pressure_np1);
        auto id_v   =              samurai::petsc::make_identity_FV(velocity_np1);

        // Stokes with backward Euler
        //             | I + dt*Diff    dt*Grad |
        //             |       -Div        0    |
        auto stokes = samurai::petsc::make_block_operator<2, 2>(id_v + dt * diff_v, dt * grad_p,
                                                                            -div_v,      zero_p);
        // clang-format on

        auto MRadaptation = samurai::make_MRAdapt(velocity);

        // Linear solver
        auto stokes_solver = samurai::petsc::make_solver(stokes);
        configure_saddle_point_solver(stokes_solver);

        // Time iteration
        samurai::save(path, fmt::format("{}{}", filename, "_init"), mesh, velocity);
        double dt_save    = dt; // Tf/static_cast<double>(nfiles);
        std::size_t nsave = 1, nt = 0;

        bool mesh_has_changed = false;
        bool dt_has_changed   = false;
        double t              = 0;
        while (t != Tf)
        {
            // Move to next timestep
            t += dt;
            if (t > Tf)
            {
                dt += Tf - t;
                t              = Tf;
                dt_has_changed = true;
            }
            std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt);

            if (min_level != max_level)
            {
                // Mesh adaptation
                MRadaptation(mr_epsilon, mr_regularity);
                velocity_np1.resize();
                pressure_np1.resize();
                zero.resize();
                mesh_has_changed = true;
                std::cout << ", levels " << mesh[mesh_id_t::cells].min_level() << "-" << mesh[mesh_id_t::cells].max_level();
            }
            std::cout << std::endl;

            // Solve system
            if (mesh_has_changed || dt_has_changed)
            {
                stokes_solver.reset();
                configure_saddle_point_solver(stokes_solver);
            }

            zero.fill(0);
            stokes_solver.solve(velocity, zero);

            // Prepare next step
            std::swap(velocity.array(), velocity_np1.array());

            // Save the result
            if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
            {
                samurai::update_ghost_mr(velocity);
                auto velocity_recons = samurai::reconstruction(velocity);

                std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
                samurai::save(path, fmt::format("{}{}", filename, suffix), velocity.mesh(), velocity);
                samurai::save(path, fmt::format("{}_recons{}", filename, suffix), velocity_recons.mesh(), velocity_recons);
            }
        }
    }
    else
    {
        std::cerr << "Unknown test case. Allowed options are 's' = stationary, 'ns' = non-stationary, 'ldc' = lid-driven cavity."
                  << std::endl;
    }

    PetscFinalize();
    return 0;
}
