#pragma once
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        template <class Field>
        Vec create_petsc_vector_from(Field& f)
        {
            Vec v;
            auto n = static_cast<PetscInt>(f.mesh().nb_cells() * Field::size);
            VecCreateSeqWithArray(MPI_COMM_SELF, 1, n, f.array().data(), &v);
            return v;
        }

        template <class Field>
        void copy(Field& f, Vec& v)
        {
            auto n = static_cast<PetscInt>(f.mesh().nb_cells() * Field::size);

            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(n == n_vec);

            for (PetscInt i = 0; i < n; ++i)
            {
                double value = f.array().data()[i];
                VecSetValues(v, 1, &i, &value, INSERT_VALUES);
            }
        }

        template <class Field>
        void copy(Field& f, Vec& v, PetscInt shift)
        {
            auto n = static_cast<PetscInt>(f.mesh().nb_cells() * Field::size);

            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(shift + n <= n_vec);

            for (PetscInt i = 0; i < n; ++i)
            {
                double value = f.array().data()[i];
                VecSetValue(v, shift + i, value, INSERT_VALUES);
            }
        }

        template <class Field>
        void copy(Vec& v, Field& f)
        {
            std::size_t n = f.mesh().nb_cells() * Field::size;

            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(static_cast<PetscInt>(n) == n_vec);

            const double* arr;
            VecGetArrayRead(v, &arr);

            for (std::size_t i = 0; i < n; ++i)
            {
                f.array().data()[i] = arr[i];
            }

            VecRestoreArrayRead(v, &arr);
        }

        template <class Field>
        void copy(PetscInt shift, Vec& v, Field& f)
        {
            std::size_t n = f.mesh().nb_cells() * Field::size;

            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(shift + static_cast<PetscInt>(n) <= n_vec);

            const double* arr;
            VecGetArrayRead(v, &arr);

            for (std::size_t i = 0; i < n; ++i)
            {
                f.array().data()[i] = arr[static_cast<std::size_t>(shift) + i];
            }

            VecRestoreArrayRead(v, &arr);
        }

        bool check_nan_or_inf(const Vec& v)
        {
            PetscInt n;
            VecGetSize(v, &n);
            const double* arr;
            VecGetArrayRead(v, &arr);

            bool is_nan_or_inf = false;
            for (PetscInt i = 0; i < n; ++i)
            {
                if (std::isnan(arr[i]) || std::isinf(arr[i]) || (abs(arr[i]) < 1e-300 && abs(arr[i]) != 0))
                {
                    is_nan_or_inf = true;
                    VecView(v, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                    std::cout << std::endl;
                    break;
                }
            }
            VecRestoreArrayRead(v, &arr);
            return !is_nan_or_inf;
        }
    } // end namespace petsc

    template <class Mesh>
    bool is_uniform(const Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        return mesh[mesh_id_t::cells].min_level() == mesh[mesh_id_t::cells].max_level();
    }
} // end namespace samurai
