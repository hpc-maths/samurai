// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <iostream>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>

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
    auto& app = samurai::initialize("Lid-driven cavity", argc, argv);

    constexpr std::size_t dim = 2;
    using Config              = samurai::MRConfig<dim, 2>;
    using Mesh                = samurai::MRMesh<Config>;
    using mesh_id_t           = typename Mesh::mesh_id_t;

    using Config2 = samurai::MRConfig<dim, 3>;
    using Mesh2   = samurai::MRMesh<Config2>;

    static constexpr bool is_soa     = false;
    static constexpr bool monolithic = true;

    //----------------//
    //   Parameters   //
    //----------------//

    std::size_t min_level = 3;
    std::size_t max_level = 6;
    double Tf             = 5.;
    double dt             = Tf / 100;
    double cfl            = 0.95;

    int ink_init = 20;

    double mr_epsilon    = 1e-1; // Threshold used by multiresolution
    double mr_regularity = 3;    // Regularity guess for multiresolution

    std::size_t nfiles      = 0;
    bool export_velocity    = false;
    bool export_reconstruct = false;

    fs::path path        = fs::current_path();
    std::string filename = "ldc_ink";

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
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.add_flag("--export-velocity", export_velocity, "Export velocity field")->capture_default_str()->group("Output");
    app.add_flag("--export-reconstruct", export_reconstruct, "Export reconstructed fields")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::times::timers.start("petsc init");
    PetscInitialize(&argc, &argv, 0, nullptr);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
    PetscOptionsSetValue(NULL, "-options_left", "off"); // disable warning for unused options
    samurai::times::timers.stop("petsc init");

    auto box = samurai::Box<double, dim>({0, 0}, {1, 1});

    std::cout << "lid-driven cavity" << std::endl;

    //-------------------- 1 -----------------------------------------------------------------
    //
    // Incompressible Navier-Stokes equations:
    //              dv/dt + diff(v) + conv(v) + grad(p) = 0
    //                                           div(v) = 0
    // where v = velocity
    //       p = pressure

    auto mesh = Mesh(box, min_level, max_level);

    // Fields for the Navier-Stokes equations
    auto velocity     = samurai::make_field<dim, is_soa>("velocity", mesh);
    auto velocity_np1 = samurai::make_field<dim, is_soa>("velocity_np1", mesh);
    auto pressure_np1 = samurai::make_field<1, is_soa>("pressure_np1", mesh);

    using VelocityField = decltype(velocity);
    using PressureField = decltype(pressure_np1);

    // Multi-resolution: the mesh will be adapted according to the velocity
    auto MRadaptation = samurai::make_MRAdapt(velocity);

    // Boundary conditions (n)
    samurai::DirectionVector<dim> left   = {-1, 0};
    samurai::DirectionVector<dim> right  = {1, 0};
    samurai::DirectionVector<dim> bottom = {0, -1};
    samurai::DirectionVector<dim> top    = {0, 1};
    samurai::make_bc<samurai::Dirichlet<1>>(velocity, 1., 0.)->on(top);
    samurai::make_bc<samurai::Dirichlet<1>>(velocity, 0., 0.)->on(left, bottom, right);

    // Boundary conditions (n+1)
    samurai::make_bc<samurai::Dirichlet<1>>(velocity_np1, 1., 0.)->on(top);
    samurai::make_bc<samurai::Dirichlet<1>>(velocity_np1, 0., 0.)->on(left, bottom, right);
    samurai::make_bc<samurai::Neumann<1>>(pressure_np1, 0.);

    // Initial condition
    velocity.fill(0);
    velocity_np1.fill(0);
    pressure_np1.fill(0);

    // Operators for the Navier-Stokes equation solved with:
    //         - backward Euler,
    //         - implicit diffusion,
    //         - explicit convection.
    //
    // ==> We solve the following equations:
    //
    //              v_np1 + dt * (diff(v_np1) + grad(p_np1)) = v_n - dt * conv(v_n)
    //                             div(v_np1)                = 0
    //
    // which gives the algebraic Stokes system
    //
    //             | I + dt*diff    dt*grad | |v_np1| = |v_n -dt * conv(v_n)|
    //             |       -div        0    | |p_np1|   |         0         |

    double diff_coeff = 1. / 100;

    auto diff    = samurai::make_diffusion_order2<VelocityField>(diff_coeff); // diff: v ---> -diff_coeff*Lap(v)
    auto grad    = samurai::make_gradient_order2<PressureField>();
    auto conv    = samurai::make_convection_upwind<VelocityField>(); // conv: v ---> v.grad(v)
    auto div     = samurai::make_divergence_order2<VelocityField>();
    auto zero_op = samurai::make_zero_operator<PressureField>();
    auto id      = samurai::make_identity<VelocityField>(); // id: v ---> v

    // clang-format off
    auto stokes = samurai::make_block_operator<2, 2>(id + dt * diff, dt * grad,
                                                               -div,   zero_op);
    // clang-format on

    // Fields for the right-hand side of the system
    auto rhs  = samurai::make_field<dim, is_soa>("rhs", mesh);
    auto zero = samurai::make_field<1, is_soa>("zero", mesh);

    // Linear solver
    auto stokes_solver = samurai::petsc::make_solver<monolithic>(stokes);
    stokes_solver.set_unknowns(velocity_np1, pressure_np1);
    configure_solver(stokes_solver);

    //-------------------- 2 ----------------------------------------------------------------
    //
    // Convection of the ink concentration 'i' using the Navier-Stokes velocity 'v':
    //
    //               d(i)/dt + conv(i) = 0,       where conv(i) = v.grad(i).

    // 2nd mesh
    auto mesh2 = Mesh2(box, 1, max_level);

    // Ink data fields
    auto ink     = samurai::make_field<1, is_soa>("ink", mesh2);
    auto ink_np1 = samurai::make_field<1, is_soa>("ink_np1", mesh2);
    // Field to store the Navier-Stokes velocity transferred to the 2nd mesh
    auto velocity2 = samurai::make_field<dim, is_soa>("velocity2", mesh2);

    using InkField = decltype(ink);

    // Multi-resolution: the mesh will be adapted according to the ink concentration
    auto MRadaptation2 = samurai::make_MRAdapt(ink);

    // Here, 'velocity2' is used as the velocity parameter of the convection operator
    auto conv2 = samurai::make_convection_weno5<InkField>(velocity2);

    // Boundary condition
    samurai::make_bc<samurai::Dirichlet<3>>(ink, 0.);

    // Initial condition
    samurai::for_each_cell(mesh2,
                           [&](auto& cell)
                           {
                               double x = cell.center(0);
                               double y = cell.center(1);

                               const double center1_x = 0.5;
                               const double center1_y = 0.5;
                               const double center2_x = 0.8;
                               const double center2_y = 0.2;
                               const double center3_x = 0.2;
                               const double center3_y = 0.8;
                               const double center4_x = 0.8;
                               const double center4_y = 0.8;
                               const double center5_x = 0.2;
                               const double center5_y = 0.2;
                               const double radius    = 0.1;

                               ink[cell] = (pow(x - center1_x, 2) + pow(y - center1_y, 2) <= pow(radius, 2))
                                                || (pow(x - center2_x, 2) + pow(y - center2_y, 2) <= pow(radius, 2))
                                                || (pow(x - center3_x, 2) + pow(y - center3_y, 2) <= pow(radius, 2))
                                                || (pow(x - center4_x, 2) + pow(y - center4_y, 2) <= pow(radius, 2))
                                                || (pow(x - center5_x, 2) + pow(y - center5_y, 2) <= pow(radius, 2))
                                             ? ink_init
                                             : 0;
                           });

    //-------------------- 3 --------------------------------------------------------------
    //
    // Time iteration

    double dx                 = mesh.cell_length(mesh.max_level());
    double sum_max_velocities = 2;
    dt                        = cfl * dx / sum_max_velocities;

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    if (nfiles != 1)
    {
        if (export_velocity)
        {
            samurai::save(path, fmt::format("ldc_velocity_ite_{}", nsave), velocity.mesh(), velocity);
        }
        samurai::save(path, fmt::format("ldc_ink_ite_{}", nsave), ink.mesh(), ink);
        nsave++;
    }

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

        // Mesh adaptation for Navier-Stokes
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
        samurai::update_ghost_mr(velocity);
        rhs = velocity - dt * conv(velocity);
        zero.fill(0);
        stokes_solver.solve(rhs, zero);

        // Mesh adaptation for the ink
        MRadaptation2(mr_epsilon, mr_regularity);
        ink_np1.resize();

        // Transfer velocity_np1 to the 2nd mesh (--> velocity2)
        samurai::update_ghost_mr(velocity_np1);
        samurai::transfer(velocity_np1, velocity2);

        // Ink convection
        samurai::update_ghost_mr(ink);
        samurai::update_ghost_mr(velocity2);
        ink_np1 = ink - dt * conv2(ink);

        // Prepare next step
        std::swap(velocity.array(), velocity_np1.array());
        std::swap(ink.array(), ink_np1.array());
        min_level_n = min_level_np1;
        max_level_n = max_level_np1;

        // Save the results
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            if (nfiles != 1)
            {
                samurai::save(path, fmt::format("ldc_ink_ite_{}", nsave), ink.mesh(), ink);
            }
            else
            {
                samurai::save(path, filename, ink.mesh(), ink);
            }

            if (export_reconstruct)
            {
                samurai::update_bc(ink);
                samurai::update_ghost_mr(ink);
                auto ink_recons = samurai::reconstruction(ink);
                samurai::save(path, fmt::format("ldc_ink_recons_ite_{}", nsave), ink_recons.mesh(), ink_recons);
            }

            if (export_velocity)
            {
                // auto div_velocity    = div(velocity);
                samurai::save(path, fmt::format("ldc_velocity_ite_{}", nsave), velocity.mesh(), velocity); //, div_velocity);

                if (export_reconstruct)
                {
                    samurai::update_ghost_mr(velocity);
                    auto velocity_recons = samurai::reconstruction(velocity);
                    samurai::save(path, fmt::format("ldc_velocity_recons_ite_{}", nsave), velocity_recons.mesh(), velocity_recons);
                }
                // samurai::save(path, fmt::format("ldc_velocity2_ite_{}", nsave), velocity2.mesh(), velocity2);
            }
            nsave++;

        } // end time loop

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
    samurai::times::timers.start("petsc finalize");
    PetscFinalize();
    samurai::times::timers.stop("petsc finalize");
    samurai::finalize();
    return 0;
}
