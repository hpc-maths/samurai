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

    PetscInt n_vec;
    VecGetSize(v, &n_vec);
    assert(n == n_vec);

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

    PetscInt n_vec;
    VecGetSize(v, &n_vec);
    assert(n == n_vec);

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

bool check_nan_or_inf(const Vec& v)
{
    PetscInt n;
    VecGetSize(v, &n);
    const double *arr;
    VecGetArrayRead(v, &arr);

    bool is_nan_or_inf = false;
    for(std::size_t i=0; i<n; ++i)
    {
        if (std::isnan(arr[i]) || std::isinf(arr[i]))
        {
            is_nan_or_inf = true;
            break;
        }
    }
    VecRestoreArrayRead(v, &arr);
    return !is_nan_or_inf;
}

template<class Field>
bool check_nan_or_inf(const Field& f)
{
    std::size_t n = f.mesh().nb_cells();
    bool is_nan_or_inf = false;
    for (std::size_t i = 0; i<n; ++i)
    {
        double value = f.array().data()[i];
        if (std::isnan(value) || std::isinf(value))
        {
            is_nan_or_inf = true;
            break;
        }
    }
    return !is_nan_or_inf;
}

void error(std::string msg)
{
    std::string beginRed = "\033[1;31m";
    std::string endColor = "\033[0m";
    std::cout << beginRed << "Error: " << msg << endColor << std::endl;
}
void fatal_error(std::string msg)
{
    std::string beginRed = "\033[1;31m";
    std::string endColor = "\033[0m";
    std::cout << beginRed << "Error: " << msg << endColor << std::endl;
    std::cout << "------------------------- FAILURE -------------------------" << std::endl;
    assert(false);
    exit(EXIT_FAILURE);
}
void warning(std::string msg)
{
    std::string beginYellow = "\033[1;33m";
    std::string endColor = "\033[0m";
    std::cout << beginYellow << "Warning: " << msg << endColor << std::endl;
}