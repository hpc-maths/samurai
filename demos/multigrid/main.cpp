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

#include "Laplacian.hpp"
#include "LaplacianSolver.hpp"
#include "Timer.hpp"

static char help[] = "Geometric multigrid using the samurai meshes.\n"
            "\n"
            "-------- General\n"
            "\n"
            "-n <int>                     problem size\n"
            "-enforce_dbc [p|e|o]         enforcement of Dirichlet b.c.\n"
            "                             p  - penalization\n"
            "                             e  - elimination\n"
            "                             o  - ones on the diagonal (non-symmetric)\n"
            "-save_sol [0|1]              save solution\n"
            "-save_mesh [0|1]             save mesh\n"
            "\n"
            "-------- Samurai Multigrid ('-pc_type mg' to activate)\n"
            "\n"
            "-samg_smooth <...>           smoother used in the samurai multigrid\n"
            "                             sgs   - symmetric Gauss-Seidel\n"
            "                             gs    - Gauss-Seidel (pre: lexico., post: antilexico.)\n"
            "                             petsc - defined by Petsc options (default: Chebytchev polynomials)\n"
            "-samg_transfer_ops [1:4]     samurai multigrid transfer operators (default: 4):\n"
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
        //constexpr std::size_t start_level = 2;
        //constexpr std::size_t min_level = 2;
        //constexpr std::size_t max_level = 3;
        box = Box({0}, {1});
        //Box leftBox({0}, {0.5});
        //Box rightBox({0.5}, {1});
    }
    else if constexpr(Mesh::dim == 2)
    {
        //constexpr std::size_t start_level = 2;
        //constexpr std::size_t min_level = 2;
        //constexpr std::size_t max_level = 3;
        box = Box({0,0}, {1,1});
        //Box leftBox({0}, {0.5});
        //Box rightBox({0.5}, {1});
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

//-----------------------------------------//
// Test cases:                             //
//      exact solution and source function //
//-----------------------------------------//

template<std::size_t dim>
auto solution()
{
    if constexpr(dim == 1)
    {
        return [](const auto& coord) 
        {
            const auto& x = coord[0];
            return x * (1 - x);
        };
    }
    else if constexpr(dim == 2)
    {
        return [](const auto& coord) 
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            return x * (1 - x) * y*(1 - y);
        };
    }
    else if constexpr(dim == 3)
    {
        return [](const auto& coord) 
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            const auto& z = coord[2];
            return x * (1 - x)*y*(1 - y)*z*(1 - z);
        };
    }
}
template<std::size_t dim>
int solution_poly_degree()
{
    return static_cast<int>(pow(2, dim));
}

template<std::size_t dim>
auto source()
{
    if constexpr(dim == 1)
    {
        return [](const auto&) 
        { 
            return 2.0; 
        };
    }
    else if constexpr(dim == 2)
    {
        return [](const auto& coord) 
        { 
            const auto& x = coord[0];
            const auto& y = coord[1];
            return 2 * (y*(1 - y) + x * (1 - x)); 
        };
    }
    else if constexpr(dim == 3)
    {
        return [](const auto& coord) 
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            const auto& z = coord[2];
            return 2 * ((y*(1 - y)*z*(1 - z) + x * (1 - x)*z*(1 - z) + x * (1 - x)*y*(1 - y)));
        };
    }
}
template<std::size_t dim>
int source_poly_degree()
{
    int solution_degree = solution_poly_degree<dim>();
    return solution_degree < 0 ? -1 : max(solution_degree - 2, 0);
}








int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
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

    PetscBool save_solution = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-save_sol", &save_solution, NULL);

    PetscBool save_mesh = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-save_mesh", &save_mesh, NULL);

    PetscBool enforce_dbc_is_set = PETSC_FALSE;
    DirichletEnforcement enforce_dbc = Elimination;
    char enforce_dbc_char[2];
    PetscOptionsGetString(NULL, NULL, "-enforce_dbc", enforce_dbc_char, 2, &enforce_dbc_is_set);
    if (enforce_dbc_is_set)
    {
        if (enforce_dbc_char[0] == 'p')
            enforce_dbc = Penalization;
        else if (enforce_dbc_char[0] == 'e')
            enforce_dbc = Elimination;
        else if (enforce_dbc_char[0] == 'o')
            enforce_dbc = OnesOnDiagonal;
        else
            fatal_error("unknown value for argument -enforce_dbc");
    }
    std::cout << "Dirichlet b.c. enforcement: ";
    if (enforce_dbc == Penalization)
        std::cout << "penalization" << std::endl;
    else if (enforce_dbc == Elimination)
        std::cout << "elimination" << std::endl;
    else if (enforce_dbc == OnesOnDiagonal)
        std::cout << "ones on the diagonal (--> non-symmetric)" << std::endl;

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

    /*std::cout << "fine_mesh:" << endl << fine_mesh << std::endl;
    samurai::save("fine_mesh", fine_mesh);
    samurai::for_each_cell(fine_mesh, [](const auto& cell)
    {
        cout << cell.level << " " << cell.center(0) << endl; 
    });

    Mesh coarse_mesh = coarsen(fine_mesh);
    std::cout << "coarse_mesh:" << endl << coarse_mesh << std::endl;
    samurai::save("coarse_mesh", coarse_mesh);

    auto fine_field = samurai::make_field<double, 1>("f", fine_mesh);
    fine_field.fill(1.);
    std::cout << "fine_field:" << endl << fine_field << std::endl;

    auto coarse_field = restrict(fine_field, coarse_mesh);
    std::cout << "coarse_field (after restrict):" << endl << coarse_field << std::endl;

    fine_field = prolong(coarse_field, fine_mesh);
    std::cout << "fine_field (after prolong):" << endl << fine_field << std::endl;*/

    //----------------//
    // Create problem //
    //----------------//

    Laplacian<Field> laplacian(mesh, enforce_dbc);

    Field rhs_field = laplacian.create_rhs(source<dim>(), source_poly_degree<dim>());

    Vec b = laplacian.assemble_rhs(rhs_field);
    PetscObjectSetName((PetscObject)b, "b");
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

    auto exact_solution = solution<dim>();
    int solution_degree = solution_poly_degree<dim>();

    samurai_new::GaussLegendre gl(solution_degree);
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

    //--------------------//
    //     Finalize       //
    //--------------------//

    // Destroy Petsc objects
    VecDestroy(&b);
    solver.destroy_petsc_objects();
    PetscFinalize();

    // Save solution
    if (save_solution)
    {
        std::cout << "Saving solution..." << std::endl;
        samurai::save("solution", mesh, sol);
    }

    return 0;
}