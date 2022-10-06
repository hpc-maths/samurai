#pragma once
#include <petsc.h>

namespace samurai_new
{
    namespace petsc
    {
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
    } // namespace petsc
} // namespace samurai_new