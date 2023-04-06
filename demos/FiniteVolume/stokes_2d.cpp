// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

static char help[] = "Solution of the Stokes problem in the domain [0,1]^2.\n"
                     "Important: use argument '-fieldsplit_pressure_pc_type none' for the stationary problem,\n"
                     "                        '-fieldsplit_pressure_np1_pc_type none' for the non stationary problem.\n"
                     "\n";

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
void configure_petsc_solver(Solver& block_solver)
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
    constexpr bool is_soa = false;

    //------------------//
    // Petsc initialize //
    //------------------//

    PetscInitialize(&argc, &argv, 0, help);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

    //----------------//
    //   Parameters   //
    //----------------//

    // Default values
    PetscInt min_level      = 2;
    PetscInt max_level      = 2;
    PetscBool save_solution = PETSC_FALSE;
    PetscBool save_mesh     = PETSC_FALSE;
    fs::path path           = fs::current_path();
    std::string filename    = "velocity";

    // Get user options
    PetscOptionsGetInt(NULL, NULL, "--min-level", &min_level, NULL);
    PetscOptionsGetInt(NULL, NULL, "--max-level", &max_level, NULL);

    PetscOptionsGetBool(NULL, NULL, "--save_sol", &save_solution, NULL);
    PetscOptionsGetBool(NULL, NULL, "--save_mesh", &save_mesh, NULL);

    PetscBool path_is_set = PETSC_FALSE;
    std::string path_str(100, '\0');
    PetscOptionsGetString(NULL, NULL, "--path", path_str.data(), path_str.size(), &path_is_set);
    if (path_is_set)
    {
        path = path_str.substr(0, path_str.find('\0'));
        if (!fs::exists(path))
        {
            fs::create_directory(path);
        }
    }

    PetscBool filename_is_set = PETSC_FALSE;
    std::string filename_str(100, '\0');
    PetscOptionsGetString(NULL, NULL, "--filename", filename_str.data(), filename_str.size(), &filename_is_set);
    if (path_is_set)
    {
        filename = filename_str.substr(0, filename_str.find('\0'));
    }

    auto box = samurai::Box<double, dim>({0, 0}, {1, 1});
    // auto mesh = Mesh(box, start_level, min_level, max_level); // amr::Mesh
    auto mesh = Mesh(box, static_cast<std::size_t>(min_level), static_cast<std::size_t>(max_level)); // MRMesh

    bool stationary = false;

    //--------------------//
    // Stationary problem //
    //--------------------//
    if (stationary)
    {
        //----------------//
        // Create problem //
        //----------------//

        // 2 equations: -Lap(v) + Grad(p) = f
        //              -Div(v)           = 0
        // where v = velocity
        //       p = pressure

        // Unknowns
        auto velocity = samurai::make_field<double, dim, is_soa>("velocity", mesh);
        auto pressure = samurai::make_field<double, 1, is_soa>("pressure", mesh);

        // Boundary conditions
        velocity
            .set_dirichlet(
                [](const auto& coord)
                {
                    const auto& x = coord[0];
                    const auto& y = coord[1];
                    double v_x    = 1 / (pi * pi) * sin(pi * (x + y));
                    double v_y    = -v_x;
                    return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                })
            .everywhere();

        pressure
            .set_neumann(
                [](const auto& coord)
                {
                    const auto& x = coord[0];
                    const auto& y = coord[1];
                    int normal    = (x == 0 || y == 0) ? -1 : 1;
                    return normal * (1 / pi) * cos(pi * (x + y));
                })
            .everywhere();

        // Block operator
        auto diff_v      = samurai::petsc::make_diffusion_FV(velocity);
        auto grad_p      = samurai::petsc::make_gradient_FV(pressure);
        auto minus_div_v = -1 * samurai::petsc::make_divergence_FV(velocity);
        auto zero_p      = samurai::petsc::make_zero_operator_FV<1>(pressure);

        auto stokes = samurai::petsc::make_block_operator<2, 2>(diff_v, grad_p, minus_div_v, zero_p);

        // Right-hand side
        auto f = samurai::make_field<double, dim, is_soa>(
            "f",
            mesh,
            [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double f_x    = 2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
                double f_y    = -2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
                return xt::xtensor_fixed<double, xt::xshape<dim>>{f_x, f_y};
            },
            0);
        auto zero = samurai::make_field<double, 1, is_soa>("zero", mesh);
        zero.fill(0);

        //-------------------//
        //   Linear solver   //
        //-------------------//

        std::cout << "Solving Stokes system..." << std::endl;
        auto block_solver = samurai::petsc::make_block_solver(stokes);
        configure_petsc_solver(block_solver);
        block_solver.solve(f, zero);
        std::cout << block_solver.iterations() << " iterations" << std::endl << std::endl;

        //--------------------//
        //       Error        //
        //--------------------//

        std::cout.precision(2);

        double error = diff_v.L2Error(velocity,
                                      [](auto& coord)
                                      {
                                          const auto& x = coord[0];
                                          const auto& y = coord[1];
                                          auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                                          auto v_y      = -v_x;
                                          return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                                      });

        std::cout << "L2-error on the velocity: " << std::scientific << error << std::endl;

        //--------------------//
        //   Save solution    //
        //--------------------//

        if (save_solution)
        {
            std::cout << "Saving solution..." << std::endl;

            samurai::save(path, filename, mesh, velocity);
            samurai::save(path, "pressure", mesh, pressure);

            auto exact_velocity = samurai::make_field<double, dim, is_soa>(
                "exact_velocity",
                mesh,
                [](const auto& coord)
                {
                    const auto& x = coord[0];
                    const auto& y = coord[1];
                    auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                    return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                },
                0);
            samurai::save(path, "exact_velocity", mesh, exact_velocity);

            /*auto err = samurai::make_field<double, dim, is_soa>("error", mesh);
            for_each_cell(err.mesh(), [&](const auto& cell)
                {
                    err[cell] = exact_velocity[cell] - velocity[cell];
                });
            samurai::save(path, "error_velocity", mesh, err);*/

            auto exact_pressure = samurai::make_field<double, 1, is_soa>(
                "exact_pressure",
                mesh,
                [](const auto& coord)
                {
                    const auto& x = coord[0];
                    const auto& y = coord[1];
                    return 1 / (pi * pi) * sin(pi * (x + y));
                },
                0);
            samurai::save(path, "exact_pressure", mesh, exact_pressure);
        }
        block_solver.destroy_petsc_objects();
    }
    //------------------------//
    // Non stationary problem //
    //------------------------//
    else
    {
        //----------------//
        // Create problem //
        //----------------//

        // 2 equations: v_np1 + dt * (-diff_coeff*Lap(v_np1) + Grad(p_np1)) = dt*f_n + v_n
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

        // Boundary conditions
        // (new BC for MR)
        return xt::xtensor_fixed<double, xt::xshape<dim>>{1, 0};
    })
            .where(
                [](const auto& coord)
                {
        const auto& y = coord[1];
        return y == 1;
                });
    velocity_np1
        .set_dirichlet(
            [](const auto&)
            {
                return xt::xtensor_fixed<double, xt::xshape<dim>>{0, 0};
            })
        .where(
            [](const auto& coord)
            {
                const auto& y = coord[1];
                return y != 1;
            });

    pressure_np1
        .set_neumann(
            [](const auto&)
            {
                return 0.0;
            })
        .everywhere();

    // Initial condition
    velocity.fill(0);

    velocity_np1.fill(0);
    pressure_np1.fill(0);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    double Tf = 1.;
    double dt = Tf / 100;

    double mr_epsilon    = 1e-1; // Threshold used by multiresolution
    double mr_regularity = 3;    // Regularity guess for multiresolution

    auto MRadaptation = samurai::make_MRAdapt(velocity);

    std::size_t nfiles = 50;

    samurai::save(path, fmt::format("{}{}", filename, "_init"), mesh, velocity);
    double dt_save    = dt; // Tf/static_cast<double>(nfiles);
    std::size_t nsave = 1, nt = 0;

    double t = 0;
    while (t != Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt);

        if (min_level != max_level)
        {
            // Mesh adaptation
            MRadaptation(mr_epsilon, mr_regularity);
            // samurai::update_ghost_mr(velocity);
            velocity_np1.resize();
            pressure_np1.resize();
            zero.resize();
            zero.fill(0);

            // Min and max levels actually used
            std::size_t actual_min_level = 999;
            std::size_t actual_max_level = 0;
            samurai::for_each_level(velocity.mesh(),
                                    [&](auto level)
                                    {
                                        actual_min_level = std::min(actual_min_level, level);
                                        actual_max_level = std::max(actual_max_level, level);
                                    });
            std::cout << ", levels " << actual_min_level << "-" << actual_max_level;
        }
        std::cout << std::endl;

        // clang-format off

            // Stokes operator
            //             |  Diff  Grad |
            //             | -Div     0  |
            auto diff_v      = diff_coeff * samurai::petsc::make_diffusion_FV(velocity_np1);
            auto grad_p      =              samurai::petsc::make_gradient_FV(pressure_np1);
            auto minus_div_v =         -1 * samurai::petsc::make_divergence_FV(velocity_np1);
            auto zero_p      =              samurai::petsc::make_zero_operator_FV<1>(pressure_np1);

            // Stokes with backward Euler
            //             | I + dt*Diff    dt*Grad |
            //             |       -Div        0    |
            auto id_v            = samurai::petsc::make_identity_FV(velocity_np1);
            auto id_plus_dt_diff = id_v + dt * diff_v;
            auto dt_grad_p       = dt * grad_p;

            auto stokes = samurai::petsc::make_block_operator<2, 2>(id_plus_dt_diff, dt_grad_p,
                                                                        minus_div_v,    zero_p);
        // clang-format on

        // Linear solver
        auto block_solver = samurai::petsc::make_block_solver(stokes);
        configure_petsc_solver(block_solver);

        // Solve the linear equation
        //                [I + dt*Diff] v_np1 + dt*p_np1 = v_n
        //                         -Div v_np1            = 0
        block_solver.solve(velocity, zero);

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

//--------------------//
//     Finalize       //
//--------------------//

PetscFinalize();

return 0;
}