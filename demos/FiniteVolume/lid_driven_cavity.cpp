// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <CLI/CLI.hpp>
#include <iostream>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>
#include <samurai/reconstruction.hpp>

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

template <class Solver>
void configure_direct_solver(Solver& solver)
{
    KSP ksp = solver.Ksp();
    PC pc;
    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPPREONLY); // (equiv. '-ksp_type preonly')
    PCSetType(pc, PCQR);         // (equiv. '-pc_type qr')
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
    PCSetType(pc, PCFIELDSPLIT);                 // (equiv. '-pc_type fieldsplit')
    PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR); // Schur complement preconditioner (equiv. '-pc_fieldsplit_type schur')
    PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP, PETSC_NULLPTR); // (equiv. '-pc_fieldsplit_schur_precondition selfp')
    PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_FULL);           // (equiv. '-pc_fieldsplit_schur_fact_type full')

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
        configure_saddle_point_solver(solver);
    }
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim        = 2;
    using Config                     = samurai::MRConfig<dim, 1>;
    using Mesh                       = samurai::MRMesh<Config>;
    using mesh_id_t                  = typename Mesh::mesh_id_t;
    static constexpr bool is_soa     = false;
    static constexpr bool monolithic = true;

    //----------------//
    //   Parameters   //
    //----------------//

    std::size_t min_level = 5;
    std::size_t max_level = 5;
    double Tf             = 5.;
    double dt             = Tf / 100;
    double cfl            = 0.95;

    double mr_epsilon    = 1e-1; // Threshold used by multiresolution
    double mr_regularity = 3;    // Regularity guess for multiresolution
    std::size_t nfiles   = 50;

    fs::path path = fs::current_path();

    CLI::App app{"Lid-driven cavity"};
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
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

    auto box  = samurai::Box<double, dim>({0, 0}, {1, 1});
    auto mesh = Mesh(box, static_cast<std::size_t>(min_level), static_cast<std::size_t>(max_level));

    auto ink_mesh = Mesh(box, static_cast<std::size_t>(max_level), static_cast<std::size_t>(max_level));

    //--------------------//
    // Stationary problem //
    //--------------------//

    std::cout << "lid-driven cavity" << std::endl;

    // 2 equations: v_np1 + dt * (-diff_coeff*Lap(v_np1) + Grad(p_np1)) = v_n
    //                                        Div(v_np1)                = 0
    // where v = velocity
    //       p = pressure

    double diff_coeff = 1. / 100;

    // Fields for the Navier-Stokes equation
    auto velocity     = samurai::make_field<dim, is_soa>("velocity", mesh);
    auto velocity_np1 = samurai::make_field<dim, is_soa>("velocity_np1", mesh);
    auto pressure_np1 = samurai::make_field<1, is_soa>("pressure_np1", mesh);

    using VelocityField = decltype(velocity);
    using PressureField = decltype(pressure_np1);

    // Fields for the ink convection
    auto ink          = samurai::make_field<1, is_soa>("ink", ink_mesh);
    auto ink_np1      = samurai::make_field<1, is_soa>("ink_np1", ink_mesh);
    auto ink_velocity = samurai::make_field<dim, is_soa>("ink_velocity", ink_mesh);

    using InkField = decltype(ink);

    // Right-hand side of the Stokes system
    auto rhs  = samurai::make_field<dim, is_soa>("rhs", mesh);
    auto zero = samurai::make_field<1, is_soa>("zero", mesh);
    zero.fill(0);

    // Boundary conditions (n)
    samurai::DirectionVector<dim> left   = {-1, 0};
    samurai::DirectionVector<dim> right  = {1, 0};
    samurai::DirectionVector<dim> bottom = {0, -1};
    samurai::DirectionVector<dim> top    = {0, 1};
    samurai::make_bc<samurai::Dirichlet>(velocity, 1., 0.)->on(top);
    samurai::make_bc<samurai::Dirichlet>(velocity, 0., 0.)->on(left, bottom, right);

    samurai::make_bc<samurai::Dirichlet>(ink, 0.);

    // Boundary conditions (n+1)
    samurai::make_bc<samurai::Dirichlet>(velocity_np1, 1., 0.)->on(top);
    samurai::make_bc<samurai::Dirichlet>(velocity_np1, 0., 0.)->on(left, bottom, right);

    samurai::make_bc<samurai::Neumann>(pressure_np1, 0.);

    // Initial condition
    velocity.fill(0);

    velocity_np1.fill(0);
    pressure_np1.fill(0);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               double x              = cell.center(0);
                               double y              = cell.center(1);
                               const double center_x = 0.5;
                               const double center_y = 0.5;
                               const double radius   = 0.1;

                               ink[cell] = (pow(x - center_x, 2) + pow(y - center_y, 2) <= pow(radius, 2)) ? 1 : 0;
                           });

    // Operators for the Navier-Stokes equation
    auto diff    = samurai::make_diffusion<VelocityField>(diff_coeff);
    auto grad    = samurai::make_gradient<PressureField>();
    auto conv    = samurai::make_convection<VelocityField>();
    auto div     = samurai::make_divergence<VelocityField>();
    auto zero_op = samurai::make_zero_operator<PressureField>();
    auto id      = samurai::make_identity<VelocityField>();

    // clang-format off

    // Stokes with backward Euler
    //             | I + dt*Diff    dt*Grad |
    //             |       -Div        0    |
    auto stokes = samurai::make_block_operator<2, 2>(id + dt * diff, dt * grad,
                                                               -div,   zero_op);
    // clang-format on

    // Convection operator for the ink
    auto ink_conv = samurai::make_convection<InkField>(ink_velocity);

    // Linear solver for the Stokes system
    auto stokes_solver = samurai::petsc::make_solver<monolithic>(stokes);

    stokes_solver.set_unknowns(velocity_np1, pressure_np1);
    configure_solver(stokes_solver);

    // Time iteration
    double dx                 = samurai::cell_length(mesh.max_level());
    double sum_max_velocities = 2;
    dt                        = cfl * dx / sum_max_velocities;

    auto MRadaptation = samurai::make_MRAdapt(velocity);

    double dt_save    = dt; // Tf/static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;
    samurai::save(path, fmt::format("ldc_velocity_ite_{}", nsave), velocity.mesh(), velocity);
    samurai::save(path, fmt::format("ldc_ink_ite_{}", nsave), ink.mesh(), ink);
    nsave++;

    bool mesh_has_changed = false;
    bool dt_has_changed   = false;

    std::size_t min_level_n   = mesh[mesh_id_t::cells].min_level();
    std::size_t max_level_n   = mesh[mesh_id_t::cells].max_level();
    std::size_t min_level_np1 = min_level_n;
    std::size_t max_level_np1 = max_level_n;

    double t = 0;
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
                zero.resize();
                rhs.resize();
            }
            std::cout << ", levels " << min_level_np1 << "-" << max_level_np1;
        }
        std::cout << std::endl;

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

        // Solve Stokes system
        rhs.fill(0);
        auto conv_v = conv(velocity);
        rhs         = velocity - dt * conv_v;
        zero.fill(0);
        stokes_solver.solve(rhs, zero);

        // Ink convection
        samurai::update_ghost_mr(velocity_np1);
        samurai::transfer(velocity_np1, ink_velocity);

        auto conv_ink = ink_conv(ink);
        ink_np1       = ink - dt * conv_ink;

        // Prepare next step
        std::swap(velocity.array(), velocity_np1.array());
        std::swap(ink.array(), ink_np1.array());
        min_level_n = min_level_np1;
        max_level_n = max_level_np1;

        // Save the result
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            samurai::update_ghost_mr(velocity);
            auto velocity_recons = samurai::reconstruction(velocity);
            auto div_velocity    = div(velocity);

            samurai::save(path, fmt::format("ldc_velocity_ite_{}", nsave), velocity.mesh(), velocity, div_velocity);
            samurai::save(path, fmt::format("ldc_velocity_recons_ite_{}", nsave), velocity_recons.mesh(), velocity_recons);

            samurai::save(path, fmt::format("ldc_ink_velocity_ite_{}", nsave), ink_velocity.mesh(), ink_velocity);
            samurai::save(path, fmt::format("ldc_ink_ite_{}", nsave), ink.mesh(), ink);
            nsave++;
        }

        // srand(time(NULL));
        // auto x_velocity = samurai::make_field<dim, is_soa>("x_velocity", mesh);
        // auto x_pressure = samurai::make_field<1, is_soa>("x_pressure", mesh);
        // samurai::for_each_cell(mesh[mesh_id_t::reference],
        //                        [&](auto cell)
        //                        {
        //                            x_velocity[cell] = rand() % 10 + 1;
        //                            x_pressure[cell] = rand() % 10 + 1;
        //                        });
        // auto x = stokes.tie_unknowns(x_velocity, x_pressure);

        // auto monolithicAssembly = samurai::petsc::make_assembly<true>(stokes);
        // Mat monolithicA;
        // monolithicAssembly.create_matrix(monolithicA);
        // monolithicAssembly.assemble_matrix(monolithicA);
        // Vec mono_x                = monolithicAssembly.create_applicable_vector(x); // copy
        // auto result_velocity_mono = samurai::make_field<dim, is_soa>("result_velocity", mesh);
        // auto result_pressure_mono = samurai::make_field<1, is_soa>("result_pressure", mesh);
        // auto result_mono          = stokes.tie_rhs(result_velocity_mono, result_pressure_mono);
        // Vec mono_result           = monolithicAssembly.create_rhs_vector(result_mono); // copy
        // MatMult(monolithicA, mono_x, mono_result);
        // monolithicAssembly.update_result_fields(mono_result, result_mono); // copy

        // auto nestedAssembly = samurai::petsc::make_assembly<false>(stokes);
        // Mat nestedA;
        // nestedAssembly.create_matrix(nestedA);
        // nestedAssembly.assemble_matrix(nestedA);
        // Vec nest_x                = nestedAssembly.create_applicable_vector(x);
        // auto result_velocity_nest = samurai::make_field<dim, is_soa>("result_velocity", mesh);
        // auto result_pressure_nest = samurai::make_field<1, is_soa>("result_pressure", mesh);
        // auto result_nest          = stokes.tie_rhs(result_velocity_nest, result_pressure_nest);
        // Vec nest_result           = nestedAssembly.create_rhs_vector(result_nest);
        // MatMult(nestedA, nest_x, nest_result);

        // std::cout << std::setprecision(15);
        // std::cout << std::fixed;
        // samurai::for_each_cell(
        //     mesh[mesh_id_t::reference],
        //     [&](auto cell)
        //     {
        //         std::cout << round(result_velocity_mono[cell][0] * 1.e8) / 1.e8 << std::endl;
        //         if (round(result_velocity_mono[cell][0] * 1.e5) / 1.e5 != round(result_velocity_nest[cell][0] * 1.e5) / 1.e5)
        //         {
        //             std::cout << result_velocity_mono[cell][0] << " =? " << result_velocity_nest[cell][0] << std::endl;
        //         }
        //         if (result_pressure_mono[cell] != result_pressure_nest[cell])
        //         {
        //             std::cout << result_pressure_mono[cell] << " =? " << result_pressure_nest[cell] << std::endl;
        //         }
        //     });
    }

    stokes_solver.destroy_petsc_objects();
    PetscFinalize();
    return 0;
}
