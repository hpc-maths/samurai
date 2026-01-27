// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <cmath>
#include <iostream>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Differentially heated cavity", argc, argv);

    constexpr std::size_t dim = 2;

    //----------------//
    //   Parameters   //
    //----------------//

    double Ra = 1e5;  // Rayleigh number
    double Pr = 0.71; // Prandtl number (air)

    double Tf = 10; // Final time
    double dt = 1e-1;

    std::size_t nfiles   = 0;
    fs::path path        = fs::current_path();
    std::string filename = "dhc";

    app.add_option("--Ra", Ra, "Rayleigh number")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    std::cout << "Differentially heated cavity (Ra = " << Ra << ")" << std::endl;

    // Dimensionless incompressible Navier-Stokes equations (see, e.g., https://www.mdpi.com/2673-3951/6/3/66 eq. (19)-(22)):
    //
    //              ∂V/∂t -   √(Pr/Ra)ΔV + V·∇V + ∇P - T*e_y = 0
    //                                                   ∇·V = 0
    //              ∂T/∂t - 1/√(Pr·Ra)ΔT + V·∇T              = 0
    //
    // where V = velocity,
    //       P = pressure,
    //       T = temperature,
    // and Ra  = Rayleigh number,
    //     Pr  = Prandtl number,
    //     e_y = unit vector in the y direction

    //-----------------//
    // Field creations //
    //-----------------//

    // Mesh creation
    auto box    = samurai::Box<double, dim>({0, 0}, {1, 1});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(7).max_stencil_size(2);
    auto mesh   = samurai::mra::make_mesh(box, config);

    // Fields for the Navier-Stokes equations
    auto velocity        = samurai::make_vector_field<double, dim>("velocity", mesh);
    auto velocity_np1    = samurai::make_vector_field<double, dim>("velocity_np1", mesh);
    auto pressure        = samurai::make_scalar_field<double>("pressure", mesh);
    auto pressure_np1    = samurai::make_scalar_field<double>("pressure_np1", mesh);
    auto temperature     = samurai::make_scalar_field<double>("temperature", mesh);
    auto temperature_np1 = samurai::make_scalar_field<double>("temperature_np1", mesh);

    // Fields for the right-hand side of the system
    auto zero_pressure = samurai::make_scalar_field<double>("zero_pressure", mesh);

    // Fields for the null space of the system (constant pressure)
    auto constant_pressure = samurai::make_scalar_field<double>("constant_pressure", mesh, 1.);
    auto zero_velocity     = samurai::make_vector_field<double, dim>("zero_velocity", mesh, 0.);
    auto zero_temperature  = samurai::make_scalar_field<double>("zero_temperature", mesh, 0.);

    using VelocityField    = decltype(velocity);
    using PressureField    = decltype(pressure_np1);
    using TemperatureField = decltype(temperature);

    //---------------------//
    // Boundary conditions //
    //---------------------//

    samurai::DirectionVector<dim> left   = {-1, 0};
    samurai::DirectionVector<dim> right  = {1, 0};
    samurai::DirectionVector<dim> bottom = {0, -1};
    samurai::DirectionVector<dim> top    = {0, 1};

    // No-slip velocity on all walls
    samurai::make_bc<samurai::Dirichlet<1>>(velocity, 0., 0.);

    // Temperature boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(temperature, 1.)->on(left);      // Hot wall (T = 1)
    samurai::make_bc<samurai::Dirichlet<1>>(temperature, 0.)->on(right);     // Cold wall (T = 0)
    samurai::make_bc<samurai::Neumann<1>>(temperature, 0.)->on(top, bottom); // Adiabatic walls (∂T/∂n = 0)

    samurai::make_bc<samurai::Neumann<1>>(pressure, 0.);

    velocity_np1.copy_bc_from(velocity);
    pressure_np1.copy_bc_from(pressure);
    temperature_np1.copy_bc_from(temperature);

    // Multi-resolution: the mesh will be adapted according to the velocity
    auto MRadaptation = samurai::make_MRAdapt(velocity);
    auto mra_config   = samurai::mra_config().epsilon(1e-3).regularity(2);

    //--------------------//
    // Initial conditions //
    //--------------------//

    velocity.fill(0);
    pressure.fill(0);
    zero_pressure.fill(0);
    // Initial temperature: linear profile from cold (right) to hot (left)
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               double x          = cell.center(0);
                               temperature[cell] = 1. - x; // Linear profile: T = 1 at x=0, T = 0 at x=1
                           });

    //---------------------------------------//
    // Non-linear system with backward Euler //
    //---------------------------------------//

    // The Navier-Stokes equation solved with backward Euler:
    //
    // ==> We solve the following equations:
    //
    //              V_np1 + dt * (diff(V_np1) + conv(V_np1) + grad(P_np1) + buoy(T_np1)) = V_n
    //                            -div(V_np1)                                            = 0
    //              T_np1 + dt * (diff(T_np1) + conv(T_np1))                             = T_n
    //
    // The buoyancy force is: F_buoyancy = -Ra*T*e_y (in y-direction only)
    //
    // which gives the algebraic system
    //
    //             | I + dt*(diff+conv)    dt*grad          dt*buoy       | |V_np1|   |V_n|
    //             |       -div               0                0          | |P_np1| = | 0 |
    //             |         0                0        I + dt*(diff+conv) | |T_np1|   |T_n|

    auto id_V   = samurai::make_identity<VelocityField>();                                 // id:   V ---> V
    auto diff_V = samurai::make_diffusion_order2<VelocityField>(std::sqrt(Pr / Ra));       // diff: V ---> -√(Pr/Ra)ΔV
    auto conv_V = samurai::make_convection_smooth_rusanov_incompressible<VelocityField>(); // conv: V ---> V·∇V
    auto div_V  = samurai::make_divergence_order2<VelocityField>();                        // div:  V ---> ∇·V

    auto grad_P = samurai::make_gradient_order2<PressureField>(); // grad: P ---> ∇P
    auto zero_P = samurai::make_zero_operator<PressureField>();   // zero: P ---> 0

    auto id_T   = samurai::make_identity<TemperatureField>();                                             // id:   T ---> T
    auto diff_T = samurai::make_diffusion_order2<TemperatureField>(1. / std::sqrt(Pr * Ra));              // diff: T ---> -1/√(Pr·Ra)ΔT
    auto conv_T = samurai::make_convection_smooth_rusanov_incompressible<TemperatureField>(velocity_np1); // conv: T ---> V·∇T
    auto buoy_T = samurai::make_buoyancy<VelocityField, TemperatureField>(); // buoy: T ---> -T*e_y (acts only in y-direction)
    // used for the assembly of the Jacobian matrix: it fills the block ∂(conv_T)/∂V
    auto conv_dual = samurai::make_dual_convection_smooth_rusanov_incompressible<TemperatureField, VelocityField>(temperature_np1);

    // Note that we use 'velocity_np1' in conv_T and 'temperature_np1' in buoy_T to assemble the Jacobian matrix because during the Newton
    // iterations, these are the unknowns we are solving for.
    // If we used 'velocity' and 'temperature' instead, the non-linear function would always use the same values.
    // Luckily, in the Newton solver, we reuse the unknown fields to evaluate the non-linear function at each iteration.

    // Remark that for the convection operators, we use smooth versions of the Rusanov scheme to ensure differentiability (simple upwind and
    // WENO schemes are not differentiable).
    // It is crucial for the Newton solver to have a differentiable operator.

    auto navier_stokes_euler = [&](double dt)
    {
        // clang-format off
        return samurai::make_block_operator<3, 3>(
                               id_V + dt * (diff_V + conv_V),   dt * grad_P,           dt * buoy_T         ,
                                                      -div_V,        zero_P,                0              ,
                                              dt * conv_dual,         0    ,   id_T + dt * (diff_T + conv_T)
                );
        // clang-format on
    };

    //-------------------//
    // Non-linear solver //
    //-------------------//

    auto nonlin_solver = samurai::petsc::make_solver(navier_stokes_euler(dt));
    nonlin_solver.set_unknowns(velocity_np1, pressure_np1, temperature_np1);

    nonlin_solver.configure = [&](SNES& snes, KSP& ksp, PC& pc)
    {
        SNESLineSearch linesearch;
        SNESGetLineSearch(snes, &linesearch);
        SNESLineSearchSetType(linesearch, SNESLINESEARCHCP); // (equiv. '-snes_linesearch_type cp')
        // We use MUMPS because it can handle null spaces (unlike the default LU solver)
        KSPSetType(ksp, KSPPREONLY);                  // (equiv. '-ksp_type preonly')
        PCSetType(pc, PCLU);                          // (equiv. '-pc_type lu')
        PCFactorSetMatSolverType(pc, MATSOLVERMUMPS); // (equiv. '-pc_factor_mat_solver_type mumps')
        // We set the same tolerance as that of the multiresolution
        PetscReal atol = mesh.min_level() == mesh.max_level() ? 1e-4 : mra_config.epsilon();
        SNESSetTolerances(snes, atol, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE); // (equiv. '-snes_atol [atol]')
    };

    nonlin_solver.after_matrix_assembly = [&](SNES&, KSP&, PC& pc, Mat& A)
    {
        // Set the null space (constant pressure) so that iterative solvers can orthogonalize residuals against it
        Vec constant_pressure_vector = nonlin_solver.assembly().create_vector(zero_velocity, constant_pressure, zero_temperature);
        VecNormalize(constant_pressure_vector, NULL);

        MatNullSpace nullspace;
        MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &constant_pressure_vector, &nullspace);
        MatSetNullSpace(A, nullspace);
        MatNullSpaceDestroy(&nullspace);
        VecDestroy(&constant_pressure_vector);

        // If using MUMPS, set option ICNTL(24)=1 to enable the detection of null pivots (because the matrix is singular)
        PetscBool is_mumps = PETSC_FALSE;
        const char* stype;
        PCFactorGetMatSolverType(pc, &stype);
        PetscStrcmp(stype, MATSOLVERMUMPS, &is_mumps);
        if (is_mumps)
        {
            Mat F;
            PCFactorGetMatrix(pc, &F);
            MatMumpsSetIcntl(F, 24, 1); // (equiv. '-mat_mumps_icntl_24 1')
        }
    };

    // When the solver doesn't converge, for debugging purposes, the following code might be useful.
    // It saves the residual fields at each SNES iteration so that one can identify which variable doesn't decrease.
    /*
    nonlin_solver.monitor = [&](PetscInt it, PetscReal residual_norm, const auto& residual_fields)
    {
        std::cout << "\tSNES iter " << it << ", residual norm = " << residual_norm << std::endl;
        const auto& res_velocity    = std::get<0>(residual_fields);
        const auto& res_pressure    = std::get<1>(residual_fields);
        const auto& res_temperature = std::get<2>(residual_fields);
        samurai::save(path, fmt::format("snes_residual_{}", it), res_velocity.mesh(), res_velocity, res_pressure, res_temperature);
    };
    */

    nonlin_solver.stop_program_on_divergence(false);

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    if (nfiles != 1)
    {
        samurai::save(path, fmt::format("dhc_velocity_ite_{}", nsave), mesh, velocity);
        samurai::save(path, fmt::format("dhc_temperature_ite_{}", nsave), mesh, temperature);
        samurai::save(path, fmt::format("dhc_pressure_ite_{}", nsave), mesh, pressure);
        nsave++;
    }

    //----------------//
    // Time iteration //
    //----------------//

    double t = 0;
    while (t < Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            if (dt < 1e-10)
            {
                break;
            }
            t = Tf;
            // Reconstruct the block operator with the new dt
            nonlin_solver.set_block_operator(navier_stokes_euler(dt));
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}\n", nt++, t, dt);

        // Mesh adaptation for Navier-Stokes
        if (mesh.min_level() != mesh.max_level())
        {
            // Current pressure and temperature values must be conserved on the new mesh
            // to be used in the right-hand side of the non-linear system and as initial guess for the Newton method.
            MRadaptation(mra_config, temperature, pressure);

            velocity_np1.resize();
            pressure_np1.resize();
            temperature_np1.resize();

            zero_pressure.resize();
            zero_pressure.fill(0);

            constant_pressure.resize();
            zero_velocity.resize();
            zero_temperature.resize();

            constant_pressure.fill(1.0);
            zero_velocity.fill(0.0);
            zero_temperature.fill(0.0);

            // Reset the solver to re-assemble the matrices on the new mesh
            nonlin_solver.reset();
        }

        // Set initial guess for the non-linear solver
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   velocity_np1[cell]    = velocity[cell];
                                   pressure_np1[cell]    = pressure[cell];
                                   temperature_np1[cell] = temperature[cell];
                               });
        // Solve the non-linear system F(V_np1, P_np1, T_np1) = (V_n, 0, T_n)
        SNESConvergedReason reason_code = nonlin_solver.solve(velocity, zero_pressure, temperature);
        if (reason_code < 0)
        {
            std::cout << "    Non-linear solver diverged. Reducing time step and retrying..." << std::endl;
            t -= dt;                                                   // rollback time
            dt *= 0.5;                                                 // reduce time step by half
            nonlin_solver.set_block_operator(navier_stokes_euler(dt)); // Reconstruct the block operator with the new dt
            nt--;
            continue;
        }

        std::cout << "\tNewton its = " << nonlin_solver.iterations() << std::endl;

        // Remove the average pressure to avoid drift
        double avg_pressure = 0.0;
        double sum_volumes  = 0.0;
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                   double volume = std::pow(cell.length, dim);
                                   avg_pressure += volume * pressure_np1[cell];
                                   sum_volumes += volume;
                               });
        avg_pressure /= sum_volumes;
        pressure_np1 = pressure_np1 - avg_pressure;

        // Prepare next step
        samurai::swap(velocity, velocity_np1);
        samurai::swap(pressure, pressure_np1);
        samurai::swap(temperature, temperature_np1);

        // Save the results
        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            if (nfiles != 1)
            {
                samurai::save(path, fmt::format("dhc_velocity_ite_{}", nsave), mesh, velocity);
                samurai::save(path, fmt::format("dhc_temperature_ite_{}", nsave), mesh, temperature);
                samurai::save(path, fmt::format("dhc_pressure_ite_{}", nsave), mesh, pressure);
            }
            else
            {
                samurai::save(path, filename, mesh, temperature);
            }
            nsave++;
        }
    } // end time loop

    nonlin_solver.destroy_petsc_objects();
    samurai::finalize();
    return 0;
}
