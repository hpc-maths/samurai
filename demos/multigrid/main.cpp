// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/print.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include "Timer.hpp"
#include "samurai_new/utils.cpp"
#include "test_cases.hpp"

static char help[] = "Solution of the Poisson problem in the domain [0,1]^d.\n"
                     "Geometric multigrid using the samurai meshes.\n"
                     "\n"
                     "-------- General\n"
                     "\n"
                     "--level        <int>           Level used to set the problem size\n"
                     "--tc           <enum>          Test case:\n"
                     "                                   poly  - The solution is a polynomial "
                     "function.\n"
                     "                                           Homogeneous Dirichlet b.c.\n"
                     "                                   exp   - The solution is an exponential "
                     "function.\n"
                     "                                           Non-homogeneous Dirichlet "
                     "b.c.\n"
                     "--save_sol     [0|1]           Save solution (default 0)\n"
                     "--save_mesh    [0|1]           Save mesh (default 0)\n"
                     "--path         <string>        Output path\n"
                     "--filename     <string>        Solution file name\n"
                     "\n"
                     "-------- Samurai Multigrid ('-pc_type mg' to activate)\n"
                     "\n"
                     "--samg_smooth       <enum>     Smoother used in the samurai multigrid:\n"
                     "                                   sgs   - symmetric Gauss-Seidel\n"
                     "                                   gs    - Gauss-Seidel (pre: lexico., "
                     "post: antilexico.)\n"
                     "                                   petsc - defined by Petsc options "
                     "(default: Chebytchev polynomials)\n"
                     "--samg_transfer_ops [1:4]      Samurai multigrid transfer operators "
                     "(default: 1):\n"
                     "                                   1 - P assembled, R assembled\n"
                     "                                   2 - P assembled, R = P^T\n"
                     "                                   3 - P mat-free, R mat-free (via "
                     "double*)\n"
                     "                                   4 - P mat-free, R mat-free (via "
                     "Fields)\n"
                     "--samg_pred_order   [0|1]      Prediction order used in the prolongation "
                     "operator\n"
                     "\n"
                     "-------- Useful Petsc options\n"
                     "\n"
                     "-pc_type [mg|gamg|hypre...]    Sets the preconditioner ('mg' for the "
                     "samurai multigrid)\n"
                     "-ksp_monitor ascii             Prints the residual at each iteration\n"
                     "-ksp_view ascii                View the solver's parametrization\n"
                     "-ksp_rtol          <double>    Sets the solver tolerance\n"
                     "-ksp_max_it        <int>       Sets the maximum number of iterations\n"
                     "-pc_mg_levels      <int>       Sets the number of multigrid levels\n"
                     "-mg_levels_up_pc_sor_its <int> Sets the number of post-smoothing "
                     "iterations\n"
                     "-log_view -pc_mg_log           Monitors the multigrid performance\n"
                     "\n";

template <std::size_t dim>
auto create_uniform_mesh(std::size_t level)
{
    using Box = samurai::Box<double, dim>;

    Box box;
    if constexpr (dim == 1)
    {
        box = Box({0}, {1});
    }
    else if constexpr (dim == 2)
    {
        box = Box({0, 0}, {1, 1});
    }
    else if constexpr (dim == 3)
    {
        box = Box({0, 0, 0}, {1, 1, 1});
    }
    std::size_t start_level, min_level, max_level;
    start_level = level;
    min_level   = level;
    max_level   = level;

    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level).start_level(start_level).disable_minimal_ghost_width();
    return samurai::amr::make_mesh(box, config); // amr::Mesh
    // return samurai::mra::make_mesh(box, config); // MRMesh
}

template <std::size_t dim>
[[maybe_unused]] auto create_refined_mesh(std::size_t level)
{
    std::size_t min_level, max_level;
    min_level = level - 1;
    max_level = level;

    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level).disable_minimal_ghost_width();

    using cl_type = typename decltype(config)::cl_type;

    int i = 1 << min_level;

    cl_type cl;
    if constexpr (dim == 1)
    {
        cl[min_level][{}].add_interval({0, i / 2});
        cl[max_level][{}].add_interval({i, 2 * i});
    }
    static_assert(dim == 1, "create_refined_mesh() not implemented for this dimension");

    return samurai::amr::make_mesh(cl, config); // amr::Mesh
    // return samurai::mra::make_mesh(cl, config); // MRMesh
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim     = 2;
    constexpr unsigned int n_comp = 1;
    constexpr bool is_soa         = true;
    using Mesh                    = decltype(create_uniform_mesh<dim>(1));
    using Field                   = samurai::VectorField<Mesh, double, n_comp, is_soa>;

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

    //----------------//
    //   Parameters   //
    //----------------//

    // Default values
    PetscInt level             = 2;
    std::string test_case_code = "poly";
    PetscBool save_solution    = PETSC_FALSE;
    PetscBool save_mesh        = PETSC_FALSE;
    fs::path path              = fs::current_path();
    std::string filename       = "solution";

    // Get user options
    PetscOptionsGetInt(NULL, NULL, "--level", &level, NULL);

    TestCase<Field>* test_case = nullptr;
    PetscBool test_case_is_set = PETSC_FALSE;
    char test_case_char_array[10];
    PetscOptionsGetString(NULL, NULL, "--tc", test_case_char_array, 10, &test_case_is_set);
    if (test_case_is_set)
    {
        test_case_code = test_case_char_array;
    }
    if (test_case_code == "poly")
    {
        test_case = new PolynomialTestCase<Field>();
    }
    else if (test_case_code == "exp")
    {
        test_case = new ExponentialTestCase<Field>();
    }
    else
    {
        samurai::io::eprint("unknown value for argument --tc\n");
    }
    samurai::io::print("Test case: {}\n", test_case_code);

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

    //---------------//
    // Mesh creation //
    //---------------//

    auto mesh = create_uniform_mesh<dim>(static_cast<std::size_t>(level));
    // print_mesh(mesh);

    if (save_mesh)
    {
        samurai::io::print("Saving mesh...\n");
        samurai::save(path, "mesh", mesh);
    }

    samurai::io::print("Unknowns: {}\n", (mesh.nb_cells() * n_comp));

    //----------------//
    // Create problem //
    //----------------//

    auto source   = samurai::make_vector_field<double, n_comp, is_soa>("source", mesh, test_case->source());
    auto solution = samurai::make_vector_field<double, n_comp, is_soa>("solution", mesh);
    solution.fill(0);

    // Boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(solution, test_case->dirichlet());
    //  Other possibilities:
    /*samurai::DirectionVector<dim> left   = {-1, 0};
    samurai::DirectionVector<dim> right  = {1, 0};
    samurai::DirectionVector<dim> bottom = {0, -1};
    samurai::DirectionVector<dim> top    = {0, 1};
    if constexpr (dim == 2)
    {
        samurai::make_bc<samurai::Dirichlet<1>>(solution, test_case->dirichlet())->on(left, right);
        samurai::make_bc<samurai::Neumann<1>>(solution, test_case->neumann())->on(bottom, top);
    }*/

    //---------------------//
    // Solve linear system //
    //---------------------//

    auto diff        = samurai::make_diffusion_cell_based<decltype(solution)>();
    auto diff_nonlin = samurai::make_diffusion_cell_based_nonlin<decltype(solution)>();

    auto solver = samurai::petsc::make_solver(diff);
    solver.set_unknown(solution);

    Timer setup_timer, solve_timer, total_timer;

    total_timer.Start();

    samurai::io::print("Setup solver...\n");
    setup_timer.Start();
    solver.setup();
    setup_timer.Stop();

    samurai::io::print("Solving...\n");
    solve_timer.Start();
    solver.solve(source);
    solve_timer.Stop();

    total_timer.Stop();

    samurai::io::print("{} iterations\n\n", solver.iterations());

    solver.destroy_petsc_objects();

    //--------------------//
    //  Print exec times  //
    //--------------------//

    samurai::io::print("---- Setup ----\n");
    samurai::io::print("CPU time    : {}\n", fmt::streamed(setup_timer.CPU()));
    samurai::io::print("Elapsed time: {}\n", fmt::streamed(setup_timer.Elapsed()));
    samurai::io::print("---- Solve ----\n");
    samurai::io::print("CPU time    : {}\n", fmt::streamed(solve_timer.CPU()));
    samurai::io::print("Elapsed time: {}\n", fmt::streamed(solve_timer.Elapsed()));
    samurai::io::print("---- Total ----\n");
    samurai::io::print("CPU time    : {}\n", fmt::streamed(total_timer.CPU()));
    samurai::io::print("Elapsed time: {}\n", fmt::streamed(total_timer.Elapsed()));
    samurai::io::print("\n");

    /*auto right_fluxes = samurai::make_vector_field<double, n_comp, is_soa>("fluxes", mesh);
    samurai::DirectionVector<dim> right = {1, 0};
    samurai::Stencil<2, dim> comput_stencil = {{0, 0}, {1, 0}};
    samurai::for_each_interface(mesh, right, comput_stencil,
    [&](auto& interface_cells, auto& comput_cells)
    {
        const double& h = comput_cells[0].length;
        auto flux = (solution[comput_cells[1]] - solution[comput_cells[0]]) / h;
        right_fluxes[interface_cells[0]] = flux;
    });*/

    //--------------------//
    //       Error        //
    //--------------------//

    if (test_case->solution_is_known())
    {
        double error = L2_error(solution, test_case->solution());
        samurai::io::print("L2-error: {:.2e}\n", error);

        if (test_case_code == "poly")
        {
            // double hidden_constant             = samurai::compute_error_bound_hidden_constant<order>(h, error);
            // samurai::io::print("hidden_constant: {}\n", hidden_constant);
            static constexpr std::size_t order = 2;
            double h                           = mesh.cell_length(mesh.min_level());
            double hidden_constant             = 5e-2;
            double theoretical_bound           = samurai::theoretical_error_bound<order>(n_comp * hidden_constant, h);
            // samurai::io::print("theoretical_bound: {}\n", theoretical_bound);
            if (error > theoretical_bound)
            {
                samurai::io::eprint("Convergence order failure: the error must be < {}.\n", theoretical_bound);
            }
        }
    }

    // Save solution
    if (save_solution)
    {
        samurai::io::print("Saving solution...\n");
        samurai::save(path, filename, mesh, solution);
    }

    delete test_case;

    samurai::finalize();
    return 0;
}
