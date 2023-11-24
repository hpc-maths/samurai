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
            PetscObjectSetName(reinterpret_cast<PetscObject>(v), f.name().data());
            return v;
        }

        template <class Field>
        void copy(Field& f, Vec& v)
        {
            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(static_cast<PetscInt>(f.mesh().nb_cells() * Field::size) == n_vec);

            double* v_data;
            VecGetArray(v, &v_data);
            std::copy(f.array().begin(), f.array().end(), v_data);
            VecRestoreArray(v, &v_data);
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

        template <class Field>
        void copy(Field& f, const typename Field::cell_t& cell, Vec& v)
        {
            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(static_cast<PetscInt>(Field::size) == n_vec);

            double* v_data;
            VecGetArray(v, &v_data);

            if constexpr (Field::size == 1)
            {
                v_data[0] = f[cell];
            }
            else
            {
                for (std::size_t i = 0; i < Field::size; ++i)
                {
                    v_data[i] = f[cell](i);
                }
            }

            VecRestoreArray(v, &v_data);
        }

        template <class Field>
        void copy(Vec& v, Field& f, const typename Field::cell_t& cell)
        {
            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(static_cast<PetscInt>(Field::size) == n_vec);

            const double* v_data;
            VecGetArrayRead(v, &v_data);

            if constexpr (Field::size == 1)
            {
                f[cell] = v_data[0];
            }
            else
            {
                for (std::size_t i = 0; i < Field::size; ++i)
                {
                    f[cell][i] = v_data[i];
                }
            }

            VecRestoreArrayRead(v, &v_data);
        }

        void copy(PetscScalar value, Vec& v)
        {
            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(n_vec == 1);

            PetscScalar* v_data;
            VecGetArray(v, &v_data);
            v_data[0] = value;
            VecRestoreArray(v, &v_data);
        }

        template <std::size_t size>
        void copy(xt::xtensor_fixed<PetscScalar, xt::xshape<size>>& container, Vec& v)
        {
            PetscInt n_vec;
            VecGetSize(v, &n_vec);
            assert(n_vec == size);

            PetscScalar* v_data;
            VecGetArray(v, &v_data);
            std::copy(container.linear_begin(), container.linear_end(), v_data);
            VecRestoreArray(v, &v_data);
        }

    } // end namespace petsc

    template <class Mesh>
    bool is_uniform(const Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        return mesh[mesh_id_t::cells].min_level() == mesh[mesh_id_t::cells].max_level();
    }
} // end namespace samurai
