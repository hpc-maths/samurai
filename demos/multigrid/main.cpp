// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <iostream>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/subset/subset_op.hpp>

#include "Laplacian1D.hpp"
#include "LaplacianSolver.hpp"
#include "Timer.hpp"

using namespace std;

template<class Mesh>
Mesh create_mesh(int n)
{
    using Box = samurai::Box<double, Mesh::dim>;
    /*constexpr std::size_t start_level = 2;

    Box leftBox({0}, {0.5});
    Box rightBox({0.5}, {1});

    Mesh m;
    m[start_level] = {start_level, leftBox};
    m[start_level+1] = {start_level+1, rightBox};
    return m;*/

    using cl_type = typename Mesh::cl_type;

    //constexpr std::size_t start_level = 2;
    //constexpr std::size_t min_level = 2;
    //constexpr std::size_t max_level = 3;
    Box box({0}, {1});
    //Box leftBox({0}, {0.5});
    //Box rightBox({0.5}, {1});

    /*cl_type cl;
    cl[start_level] = {start_level, leftBox};
    cl[start_level+1] = {start_level+1, rightBox};*/
    std::size_t start_level, min_level, max_level;
    start_level = n;
    min_level = n;
    max_level = n;

    return Mesh(box, start_level, min_level, max_level);
}


static char help[] = "Multigrid program.\n\n"
            "-n 15 -ksp_view ascii -ksp_monitor ascii -ksp_rtol 1e-9 -ksp_max_it 10 -pc_type mg -pc_mg_levels 3 -mg_levels_up_pc_sor_its 3";

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;
    //using Mesh = samurai::CellArray<dim>;
    using Config = samurai::amr::Config<dim>;
    using Mesh = samurai::amr::Mesh<Config>;
    using Field = samurai::Field<Mesh, double, 1>;

    
    //------------------//
    // Petsc initialize //
    //------------------//
    
    PetscInitialize(&argc,&argv,(char*)0, help);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size)); 
    PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
    PetscInt n = 2;
    PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);

    //---------------//
    // Mesh creation //
    //---------------//

    Mesh fine_mesh = create_mesh<Mesh>(n);
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


    PetscErrorCode ierr;

    //----------------//
    // Create problem //
    //----------------//

    Laplacian1D<Mesh, Field> laplacian(fine_mesh, false);
    
    //Mat A;
    //laplacian.create_and_assemble_matrix(A);
    //PetscObjectSetName((PetscObject)A, "A");

    //PetscViewer viewer;
    //PetscViewerCreate(PETSC_COMM_SELF, &viewer);
    //PetscViewerCreate_ASCII(&viewer);

    //MatView(A, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
    //std::cout << std::endl;

    Field rhs_field("f", fine_mesh);
    rhs_field.fill(1);
    Vec b = laplacian.assemble_rhs(rhs_field);
    PetscObjectSetName((PetscObject)b, "b");
    //VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
    //std::cout << std::endl;

    //---------------------//
    // Solve linear system //
    //---------------------//


    //LaplacianSolver<Laplacian1D<Mesh, Field>> solver(A);
    LaplacianSolver<Laplacian1D<Mesh, Field>> solver(laplacian, fine_mesh);

    Timer setup_timer, solve_timer, total_timer;

    total_timer.Start();

    setup_timer.Start();
    solver.setup();
    setup_timer.Stop();

    solve_timer.Start();
    Field& x = rhs_field;
    solver.solve(b, x);
    solve_timer.Stop();

    total_timer.Stop();

    std::cout << "---- Setup ----" << endl;
    std::cout << "CPU time    : " << setup_timer.CPU() << endl;
    std::cout << "Elapsed time: " << setup_timer.Elapsed() << endl;
    std::cout << "---- Solve ----" << endl;
    std::cout << "CPU time    : " << solve_timer.CPU() << endl;
    std::cout << "Elapsed time: " << solve_timer.Elapsed() << endl;
    std::cout << "---- Total ----" << endl;
    std::cout << "CPU time    : " << total_timer.CPU() << endl;
    std::cout << "Elapsed time: " << total_timer.Elapsed() << endl;

    //-----------------//
    // Destroy objects //
    //-----------------//

    //ierr = MatDestroy(&A);      CHKERRQ(ierr);
    VecDestroy(&b);
    solver.destroy();

    PetscCall(PetscFinalize());
    return 0;
}