// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <iostream>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>

static constexpr double pi = M_PI;

template <class Field>
[[maybe_unused]] bool check_nan_or_inf(const Field& f)
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

template <class Solver>
void configure_direct_solver(Solver& solver)
{
    KSP ksp = solver.Ksp();
    PC pc;
    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPPREONLY); // (equiv. '-ksp_type preonly')
    PCSetType(pc, PCQR);         // (equiv. '-pc_type qr')
    // PetscBool use_superlu = PETSC_FALSE;
    // #if defined(PETSC_HAVE_SUPERLU)
    //     use_superlu = PETSC_TRUE;
    // #endif
    //     PetscBool use_mumps             = PETSC_FALSE;
    //     PetscBool solver_package_is_set = PETSC_FALSE;
    //     std::string pc_factor_mat_solver_type_str(100, '\0');
    //     PetscOptionsGetString(NULL,
    //                           NULL,
    //                           "-pc_factor_mat_solver_type",
    //                           pc_factor_mat_solver_type_str.data(),
    //                           pc_factor_mat_solver_type_str.size(),
    //                           &solver_package_is_set);
    //     if (solver_package_is_set)
    //     {
    //         pc_factor_mat_solver_type_str = pc_factor_mat_solver_type_str.substr(0, pc_factor_mat_solver_type_str.find('\0'));
    //     }
    // #if defined(PETSC_HAVE_MUMPS)
    //     if (!use_superlu || pc_factor_mat_solver_type_str == MATSOLVERMUMPS)
    //     {
    //         use_mumps   = PETSC_TRUE;
    //         use_superlu = PETSC_FALSE;
    //     }
    // #endif
    //     if (use_superlu)
    //     {
    // #if defined(PETSC_HAVE_SUPERLU)
    //         PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU); // (equiv. '-pc_factor_mat_solver_type superlu')
    // #endif
    //     }
    //     else if (use_mumps)
    //     {
    // #if defined(PETSC_HAVE_MUMPS)
    //         PCFactorSetMatSolverType(pc, MATSOLVERMUMPS); // (equiv. '-pc_factor_mat_solver_type mumps')
    // #endif
    //     }
    // KSP and PC overwritten by user value if needed
    KSPSetFromOptions(ksp);
    // If neither SuperLU nor MUMPS is installed, you can try: -ksp_type gmres -pc_type ilu
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

    block_solver.assemble_matrix(); // if nested matrix, must be called before calling set_pc_fieldsplit().

    block_solver.set_pc_fieldsplit(pc);
    PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR); // Schur complement preconditioner (equiv. '-pc_fieldsplit_type schur')
    PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, PETSC_NULLPTR); // (equiv. '-pc_fieldsplit_schur_precondition selfp')
    PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_FULL);           // (equiv. '-pc_fieldsplit_schur_fact_type full')

    // Configure the sub-solvers
    block_solver.setup(); // KSPSetUp() or PCSetUp() must be called before calling PCFieldSplitSchurGetSubKSP(), because the matrices are
                          // needed.
    KSP* sub_ksp;
    PCFieldSplitSchurGetSubKSP(pc, nullptr, &sub_ksp);
    KSP A_ksp     = sub_ksp[0];
    KSP schur_ksp = sub_ksp[1];

    // Set LU by default for the A block (diffusion). Consider using 'hypre' for large problems,
    // using the option '-fieldsplit_velocity_[np1]_pc_type hypre'.
    PC A_pc;
    KSPGetPC(A_ksp, &A_pc);
    KSPSetType(A_ksp, KSPPREONLY); // (equiv. '-fieldsplit_velocity_[np1]_ksp_type preonly')
    PCSetType(A_pc, PCLU);         // (equiv. '-fieldsplit_velocity_[np1]_pc_type lu')
    KSPSetFromOptions(A_ksp);      // KSP and PC overwritten by user value if needed

    PC schur_pc;
    KSPGetPC(schur_ksp, &schur_pc);
    KSPSetType(schur_ksp, KSPPREONLY); // (equiv. '-fieldsplit_pressure_[np1]_ksp_type preonly')
    PCSetType(schur_pc, PCQR);         // (equiv. '-fieldsplit_pressure_[np1]_pc_type qr')
    // PCSetType(schur_pc, PCJACOBI); // (equiv. '-fieldsplit_pressure_[np1]_pc_type none')
    KSPSetFromOptions(schur_ksp); // KSP and PC overwritten by user value if needed

    // If a tolerance is set by the user ('-ksp-rtol XXX'), then we set that
    // tolerance to all the sub-solvers
    PetscReal ksp_rtol;
    KSPGetTolerances(ksp, &ksp_rtol, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE);
    KSPSetTolerances(A_ksp, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);     // (equiv. '-fieldsplit_velocity_ksp_rtol XXX')
    KSPSetTolerances(schur_ksp, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); // (equiv. '-fieldsplit_pressure_ksp_rtol XXX')
}

template <class Solver>
void configure_solver(Solver& solver)
{
    if constexpr (Solver::is_monolithic)
    {
        configure_direct_solver(solver);
    }
    else
    {
        configure_saddle_point_solver(solver); // works also for monolithic
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Stokes problem", argc, argv);

    constexpr std::size_t dim        = 2;
    using Config                     = samurai::MRConfig<dim, 2>;
    using Mesh                       = samurai::MRMesh<Config>;
    using mesh_id_t                  = typename Mesh::mesh_id_t;
    static constexpr bool is_soa     = false;
    static constexpr bool monolithic = true;

    //----------------//
    //   Parameters   //
    //----------------//

    std::string test_case = "ns";

    std::size_t min_level = 5;
    std::size_t max_level = 5;
    double Tf             = 1.;
    double dt             = Tf / 100;

    double mr_epsilon    = 1e-1; // Threshold used by multiresolution
    double mr_regularity = 3;    // Regularity guess for multiresolution
    std::size_t nfiles   = 50;

    fs::path path        = fs::current_path();
    std::string filename = "";

    app.add_option("--test-case", test_case, "Test case (s = stationary, ns = non-stationary)")
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
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    PetscInitialize(&argc, &argv, 0, nullptr);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
    PetscOptionsSetValue(NULL, "-options_left", "off"); // disable warning for unused options

    auto box  = samurai::Box<double, dim>({0, 0}, {1, 1});
    auto mesh = Mesh(box, min_level, max_level);

    //--------------------//
    // Stationary problem //
    //--------------------//

    std::cout << "Problem solved: ";

    if (test_case == "s")
    {
        std::cout << "stationary" << std::endl;
        if (filename.empty())
        {
            filename = "stokes";
        }

        // 2 equations: -Lap(v) + Grad(p) = f
        //              -Div(v)           = 0
        // where v = velocity
        //       p = pressure

        // Unknowns
        auto velocity = samurai::make_field<dim, is_soa>("velocity", mesh);
        auto pressure = samurai::make_field<1, is_soa>("pressure", mesh);

        using VelocityField = decltype(velocity);
        using PressureField = decltype(pressure);

        // Boundary conditions
        samurai::make_bc<samurai::Dirichlet<1>>(velocity,
                                                [](const auto&, const auto&, const auto& coord)
                                                {
                                                    const auto& x = coord[0];
                                                    const auto& y = coord[1];
                                                    double v_x    = 1 / (pi * pi) * sin(pi * (x + y));
                                                    double v_y    = -v_x;
                                                    return samurai::Array<double, dim, is_soa>{v_x, v_y};
                                                });

        samurai::make_bc<samurai::Neumann<1>>(pressure,
                                              [](const auto&, const auto&, const auto& coord)
                                              {
                                                  const auto& x = coord[0];
                                                  const auto& y = coord[1];
                                                  int normal    = (x == 0 || y == 0) ? -1 : 1;
                                                  return normal * (1 / pi) * cos(pi * (x + y));
                                              });

        // Stokes operator
        //             |  Diff  Grad |
        //             | -Div     0  |
        auto diff    = samurai::make_diffusion_order2<VelocityField>();
        auto grad    = samurai::make_gradient_order2<PressureField>();
        auto div     = samurai::make_divergence_order2<VelocityField>();
        auto zero_op = samurai::make_zero_operator<PressureField>();

        // clang-format off
        auto stokes = samurai::make_block_operator<2, 2>(diff, grad,
                                                         -div, zero_op);
        // clang-format on

        // Right-hand side
        auto f    = samurai::make_field<dim, is_soa>("f",
                                                  mesh,
                                                  [](const auto& coord)
                                                  {
                                                      const auto& x = coord[0];
                                                      const auto& y = coord[1];
                                                      double f_x    = 2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
                                                      double f_y    = -2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
                                                      return samurai::Array<double, dim, is_soa>{f_x, f_y};
                                                  });
        auto zero = samurai::make_field<1, is_soa>("zero", mesh);
        zero.fill(0);

        // Linear solver
        std::cout << "Solving Stokes system..." << std::endl;
        auto stokes_solver = samurai::petsc::make_solver<monolithic>(stokes);

        stokes_solver.set_unknowns(velocity, pressure);
        configure_solver(stokes_solver);

        stokes_solver.solve(f, zero);
        std::cout << stokes_solver.iterations() << " iterations" << std::endl << std::endl;

        // Error
        double error = L2_error(velocity,
                                [](const auto& coord)
                                {
                                    const auto& x = coord[0];
                                    const auto& y = coord[1];
                                    auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                                    auto v_y      = -v_x;
                                    return samurai::Array<double, dim, is_soa>{v_x, v_y};
                                });
        std::cout.precision(2);
        std::cout << "L2-error on the velocity: " << std::scientific << error << std::endl;

        // Save solution
        std::cout << "Saving solution..." << std::endl;

        samurai::save(path, filename, mesh, velocity);
        samurai::save(path, "pressure", mesh, pressure);

        auto exact_velocity = samurai::make_field<dim, is_soa>("exact_velocity",
                                                               mesh,
                                                               [](const auto& coord)
                                                               {
                                                                   const auto& x = coord[0];
                                                                   const auto& y = coord[1];
                                                                   auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                                                                   auto v_y      = -v_x;
                                                                   return samurai::Array<double, dim, is_soa>{v_x, v_y};
                                                               });
        samurai::save(path, "exact_velocity", mesh, exact_velocity);

        /*auto err = samurai::make_field<dim, is_soa>("error", mesh);
        for_each_cell(err.mesh(), [&](const auto& cell)
            {
                err[cell] = exact_velocity[cell] - velocity[cell];
            });
        samurai::save(path, "error_velocity", mesh, err);*/

        auto exact_pressure = samurai::make_field<1, is_soa>("exact_pressure",
                                                             mesh,
                                                             [](const auto& coord)
                                                             {
                                                                 const auto& x = coord[0];
                                                                 const auto& y = coord[1];
                                                                 return 1 / (pi * pi) * sin(pi * (x + y));
                                                             });
        samurai::save(path, "exact_pressure", mesh, exact_pressure);
    }

    //------------------------//
    // Non stationary problem //
    //------------------------//

    else if (test_case == "ns")
    {
        std::cout << "non stationary" << std::endl;

        if (filename.empty())
        {
            filename = "stokes_ns_velocity";
        }

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
            return samurai::Array<double, dim, is_soa>{f_x, f_y};
        };

        // Exact solution
        auto exact_velocity = [&](double t, const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            double v_x    = std::sin(t) * std::sin(2 * pi * x) * std::cos(2 * pi * y);
            double v_y    = -std::sin(t) * std::cos(2 * pi * x) * std::sin(2 * pi * y);
            return samurai::Array<double, dim, is_soa>{v_x, v_y};
        };
        auto exact_normal_grad_pressure = [&](double t, const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            int normal    = (x == 0 || y == 0) ? -1 : 1;
            return normal * (-pi * std::sin(t) * std::sin(t) * std::sin(pi * x) * std::sin(pi * y));
        };

        // Unknowns
        auto velocity     = samurai::make_field<dim, is_soa>("velocity", mesh);
        auto velocity_np1 = samurai::make_field<dim, is_soa>("velocity_np1", mesh);
        auto pressure_np1 = samurai::make_field<1, is_soa>("pressure_np1", mesh);

        using VelocityField = decltype(velocity);
        using PressureField = decltype(pressure_np1);

        // Right-hand side
        auto rhs  = samurai::make_field<dim, is_soa>("rhs", mesh);
        auto zero = samurai::make_field<1, is_soa>("zero", mesh);

        // Boundary conditions
        samurai::make_bc<samurai::Dirichlet<1>>(velocity_np1,
                                                [&](const auto&, const auto&, const auto& coord)
                                                {
                                                    return exact_velocity(0, coord);
                                                });
        samurai::make_bc<samurai::Neumann<1>>(pressure_np1,
                                              [&](const auto&, const auto&, const auto& coord)
                                              {
                                                  return exact_normal_grad_pressure(0, coord);
                                              });

        // Stokes operator
        //             |  Diff  Grad |
        //             | -Div     0  |
        auto diff    = samurai::make_diffusion_order2<VelocityField>(diff_coeff);
        auto grad    = samurai::make_gradient_order2<PressureField>();
        auto div     = samurai::make_divergence_order2<VelocityField>();
        auto zero_op = samurai::make_zero_operator<PressureField>();
        auto id      = samurai::make_identity<VelocityField>();

        // clang-format off

        // Stokes with backward Euler
        //             | I + dt*Diff    dt*Grad |
        //             |       -Div        0    |
        auto stokes = samurai::make_block_operator<2, 2>(id + dt * diff, dt * grad,
                                                                   -div,   zero_op);
        // clang-format on

        // Linear solver
        auto stokes_solver = samurai::petsc::make_solver<monolithic>(stokes);

        stokes_solver.set_unknowns(velocity_np1, pressure_np1);
        configure_solver(stokes_solver);

        // Initial condition
        velocity.fill(0);

        velocity_np1.fill(0);
        pressure_np1.fill(0);

        // Time iteration
        auto MRadaptation = samurai::make_MRAdapt(velocity);

        std::size_t nsave = 0, nt = 0;
        if (nfiles != 1)
        {
            std::string suffix = fmt::format("_ite_{}", nsave++);
            samurai::save(path, fmt::format("{}{}", filename, suffix), velocity.mesh(), velocity);
        }

        bool mesh_has_changed = false;
        bool dt_has_changed   = false;

        std::size_t min_level_n   = mesh[mesh_id_t::cells].min_level();
        std::size_t max_level_n   = mesh[mesh_id_t::cells].max_level();
        std::size_t min_level_np1 = min_level_n;
        std::size_t max_level_np1 = max_level_n;

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

            // Mesh adaptation
            if (min_level != max_level)
            {
                MRadaptation(mr_epsilon, mr_regularity);
                min_level_np1    = mesh[mesh_id_t::cells].min_level();
                max_level_np1    = mesh[mesh_id_t::cells].max_level();
                mesh_has_changed = !(samurai::is_uniform(mesh) && min_level_n == min_level_np1 && max_level_n == max_level_np1);
                if (mesh_has_changed)
                {
                    velocity_np1.resize();
                    pressure_np1.resize();
                    rhs.resize();
                    zero.resize();
                }
                std::cout << ", levels " << min_level_np1 << "-" << max_level_np1;
            }
            std::cout.flush();

            // Boundary conditions
            velocity_np1.get_bc().clear();
            samurai::make_bc<samurai::Dirichlet<1>>(velocity_np1,
                                                    [&](const auto&, const auto&, const auto& coord)
                                                    {
                                                        return exact_velocity(t_np1, coord);
                                                    });
            pressure_np1.get_bc().clear();
            samurai::make_bc<samurai::Neumann<1>>(pressure_np1,
                                                  [&](const auto&, const auto&, const auto& coord)
                                                  {
                                                      return exact_normal_grad_pressure(t_np1, coord);
                                                  });

            // Update solver
            if (mesh_has_changed || dt_has_changed)
            {
                if (dt_has_changed)
                {
                    stokes = samurai::make_block_operator<2, 2>(id + dt * diff, dt * grad, -div, zero_op);
                }
                stokes_solver = samurai::petsc::make_solver<monolithic>(stokes);
                stokes_solver.set_unknowns(velocity_np1, pressure_np1);
                configure_solver(stokes_solver);
            }

            // Solve the linear equation
            //                [I + dt*Diff] v_np1 + dt*p_np1 = v_n + dt*f
            //                         -Div v_np1            = 0
            auto f = samurai::make_field<dim, is_soa>("f",
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
            t_n         = t_np1;
            min_level_n = min_level_np1;
            max_level_n = max_level_np1;

            // Error
            double error = L2_error(velocity,
                                    [&](auto& coord)
                                    {
                                        return exact_velocity(t_n, coord);
                                    });
            std::cout.precision(2);
            std::cout << ", L2-error: " << std::scientific << error;

            // Divergence
            auto div_velocity = div(velocity);

            // Save the result
            if (nfiles != 1)
            {
                samurai::save(path, fmt::format("{}_ite_{}", filename, nsave), velocity.mesh(), velocity, div_velocity);
            }

            if (min_level != max_level)
            {
                // Reconstruction on the finest level
                samurai::update_ghost_mr(velocity);
                auto velocity_recons = samurai::reconstruction(velocity);
                // Error
                double error_recons = samurai::L2_error(velocity_recons,
                                                        [&](auto& coord)
                                                        {
                                                            return exact_velocity(t_n, coord);
                                                        });
                std::cout.precision(2);
                std::cout << ", L2-error (recons): " << std::scientific << error_recons;
                // Save
                if (nfiles != 1)
                {
                    samurai::save(path, fmt::format("{}_recons_ite_{}", filename, nsave), velocity_recons.mesh(), velocity_recons);
                }
            }
            if (nfiles != 1)
            {
                nsave++;
            }
            std::cout << std::endl;
        }

        if (nfiles == 1)
        {
            samurai::save(path, filename, velocity.mesh(), velocity);
        }
    }
    else
    {
        std::cerr << "Unknown test case. Allowed options are 's' = stationary, 'ns' = non-stationary." << std::endl;
    }

    PetscFinalize();
    samurai::finalize();
    return 0;
}
