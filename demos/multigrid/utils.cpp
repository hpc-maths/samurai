#pragma once
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <petsc.h>

template<class Field>
Vec create_petsc_vector_from(Field& f)
{
    Vec v;
    std::size_t n = f.mesh().nb_cells();
    VecCreateSeqWithArray(MPI_COMM_SELF, 1, n, f.array().data(), &v);
    return v;
}

template<class Field>
void copy(Field& f, Vec& v)
{
    std::size_t n = f.mesh().nb_cells();
    for (PetscInt i; i<n; ++i)
    {
        double value = f.array().data()[i];
        VecSetValues(v, 1, &i, &value, INSERT_VALUES);
    }
}



template<class Field>
void copy(Vec& v, Field& f)
{
    std::size_t n = f.mesh().nb_cells();
    const double *arr;
    VecGetArrayRead(v, &arr);

    for(std::size_t i=0; i<n; ++i)
        f(i) = arr[i];

    VecRestoreArrayRead(v, &arr);
}


template<class Mesh>
void print_mesh(Mesh& mesh)
{
    std::cout << mesh << std::endl;
    samurai::for_each_cell(mesh, [](const auto& cell)
    {
        std::cout << cell.level << " " << cell.center(0) << std::endl; 
    });
}