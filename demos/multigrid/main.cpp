// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <iostream>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/subset/subset_op.hpp>
#include <samurai/amr/mesh.hpp>

#include "test_cases.hpp"
#include "Laplacian.hpp"
#include "LaplacianSolver.hpp"
#include "Timer.hpp"

static char help[] = "Geometric multigrid using the samurai meshes.\n"
            "\n"
            "-------- General\n"
            "\n"
            "-n <int>                     problem size of the Laplacian problem on the domain [0,1]^d\n"
            "-tc <...>                    test case:\n"
            "                             poly  - The solution is a polynomial function.\n"
            "                                     Homogeneous Dirichlet b.c.\n"
            "                             exp   - The solution is an exponential function.\n"
            "                                     Non-homogeneous Dirichlet b.c.\n"
            "-save_sol [0|1]              save solution\n"
            "-save_mesh [0|1]             save mesh\n"
            "\n"
            "-------- Samurai Multigrid ('-pc_type mg' to activate)\n"
            "\n"
            "-samg_smooth <...>           smoother used in the samurai multigrid:\n"
            "                             sgs   - symmetric Gauss-Seidel\n"
            "                             gs    - Gauss-Seidel (pre: lexico., post: antilexico.)\n"
            "                             petsc - defined by Petsc options (default: Chebytchev polynomials)\n"
            "-samg_transfer_ops [1:4]     samurai multigrid transfer operators (default: 1):\n"
            "                             1 - P assembled, R assembled\n"
            "                             2 - P assembled, R = P^T\n"
            "                             3 - P mat-free, R mat-free (via double*)\n"
            "                             4 - P mat-free, R mat-free (via Fields)\n"
            "-samg_pred_order [0|1]       prediction order used in the prolongation operator\n"
            "\n"
            "-------- Useful Petsc options\n"
            "\n"
            "-pc_type [mg|gamg|hypre...]  sets the preconditioner ('mg' for the samurai multigrid)\n"
            "-ksp_monitor ascii           prints the residual at each iteration\n"
            "-ksp_view ascii              view the solver's parametrization\n"
            "-ksp_rtol <double>           sets the solver tolerance\n"
            "-ksp_max_it <int>            sets the maximum number of iterations\n"
            "-pc_mg_levels <int>          sets the number of multigrid levels\n"
            "-mg_levels_up_pc_sor_its <int> sets the number of post-smoothing iterations\n"
            "-log_view -pc_mg_log         monitors the multigrid performance\n"
            "\n";



template<class Mesh>
Mesh create_mesh(std::size_t n)
{
    using Box = samurai::Box<double, Mesh::dim>;
    /*constexpr std::size_t start_level = 2;

    Box leftBox({0}, {0.5});
    Box rightBox({0.5}, {1});

    Mesh m;
    m[start_level] = {start_level, leftBox};
    m[start_level+1] = {start_level+1, rightBox};
    return m;*/

    //using cl_type = typename Mesh::cl_type;

    Box box;
    if constexpr(Mesh::dim == 1)
    {
        box = Box({0}, {1});
    }
    else if constexpr(Mesh::dim == 2)
    {
        box = Box({0,0}, {1,1});
    }
    else if constexpr(Mesh::dim == 3)
    {
        box = Box({0,0,0}, {1,1,1});
    }

    /*cl_type cl;
    cl[start_level] = {start_level, leftBox};
    cl[start_level+1] = {start_level+1, rightBox};*/
    std::size_t start_level, min_level, max_level;
    start_level = n;
    min_level = n;
    max_level = n;

    return Mesh(box, start_level, min_level, max_level);
}




int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;
    using Config = samurai::amr::Config<dim>;
    using Mesh = samurai::amr::Mesh<Config>;
    using Field = samurai::Field<Mesh, double, 1>;

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

    PetscInt n = 2;
    PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);

    TestCase<dim>* test_case = nullptr;
    PetscBool test_case_is_set = PETSC_FALSE;
    char test_case_char_array[10];
    PetscOptionsGetString(NULL, NULL, "-tc", test_case_char_array, 10, &test_case_is_set);
    std::string test_case_code = test_case_is_set ? test_case_char_array : "poly";
    if (test_case_code == "poly")
    {
        test_case = new PolynomialTestCase<dim>();
    }
    else if (test_case_code == "exp")
    {
        test_case = new ExponentialTestCase<dim>();
    }
    else
    {
        fatal_error("unknown value for argument -tc");
    }
    std::cout << "Test case: " << test_case_code << std::endl;


    PetscBool save_solution = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-save_sol", &save_solution, NULL);

    PetscBool save_mesh = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-save_mesh", &save_mesh, NULL);

    //---------------//
    // Mesh creation //
    //---------------//

    Mesh mesh = create_mesh<Mesh>(static_cast<std::size_t>(n));
    std::cout << "Unknowns: " << mesh.nb_cells() << std::endl;

    if (save_mesh)
    {
        std::cout << "Saving mesh..." << std::endl;
        samurai::save("mesh", mesh);
    }

    //----------------//
    // Create problem //
    //----------------//

    Laplacian<Field> laplacian(mesh);

    Field rhs_field = laplacian.create_rhs(test_case->source(), test_case->source_poly_degree());
    laplacian.enforce_dirichlet_bc(rhs_field, test_case->dirichlet());

    Vec b = samurai_new::petsc::create_petsc_vector_from(rhs_field);
    PetscObjectSetName(reinterpret_cast<PetscObject>(b), "b");
    //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

    //---------------------//
    // Solve linear system //
    //---------------------//

    LaplacianSolver<Laplacian<Field>> solver(laplacian, mesh);

    Timer setup_timer, solve_timer, total_timer;

    total_timer.Start();

    std::cout << "Setup solver..." << std::endl;
    setup_timer.Start();
    solver.setup();
    setup_timer.Stop();

    std::cout << "Solving..." << std::endl;
    solve_timer.Start();
    Field sol("solution", mesh);
    solver.solve(b, sol);
    solve_timer.Stop();

    total_timer.Stop();

    //--------------------//
    //  Print exec times  //
    //--------------------//

    std::cout << "---- Setup ----" << std::endl;
    std::cout << "CPU time    : " << setup_timer.CPU() << std::endl;
    std::cout << "Elapsed time: " << setup_timer.Elapsed() << std::endl;
    std::cout << "---- Solve ----" << std::endl;
    std::cout << "CPU time    : " << solve_timer.CPU() << std::endl;
    std::cout << "Elapsed time: " << solve_timer.Elapsed() << std::endl;
    std::cout << "---- Total ----" << std::endl;
    std::cout << "CPU time    : " << total_timer.CPU() << std::endl;
    std::cout << "Elapsed time: " << total_timer.Elapsed() << std::endl;
    std::cout << std::endl;

    //--------------------//
    //       Error        //
    //--------------------//

    if (test_case->solution_is_known())
    {
        auto exact_solution = test_case->solution();

        samurai_new::GaussLegendre gl(test_case->solution_poly_degree());
        double error_norm = 0;
        double solution_norm = 0;
        samurai::for_each_cell(mesh, [&](const auto& cell)
        {
            error_norm += gl.quadrature(cell, [&](const auto& point)
            {
                return pow(exact_solution(point) - sol(cell.index), 2);
            });

            solution_norm += gl.quadrature(cell, [&](const auto& point)
            {
                return pow(exact_solution(point), 2);
            });
        });

        error_norm = sqrt(error_norm);
        solution_norm = sqrt(solution_norm);
        double relative_error = error_norm/solution_norm;

        std::cout.precision(2);
        std::cout << "L2-error: " << std::scientific << relative_error << std::endl;
    }

    // Save solution
    if (save_solution)
    {
        std::cout << "Saving solution..." << std::endl;
        samurai::save("solution", mesh, sol);
    }

    //--------------------//
    //     Finalize       //
    //--------------------//

    delete test_case;

    // Destroy Petsc objects
    VecDestroy(&b);
    solver.destroy_petsc_objects();
    PetscFinalize();

    return 0;
}