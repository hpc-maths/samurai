#pragma once
#include "petsc_assembly.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../boundary.hpp"

namespace samurai { namespace petsc
{
    namespace detail
    {
        /**
         * Local square matrix to store the coefficients of a vectorial field.
        */
        template<class value_type, std::size_t size>
        struct LocalMatrix
        {
            using Type = xt::xtensor_fixed<value_type, xt::xshape<size, size>>;
        };

        /**
         * Template specialization: if size=1, then just a scalar coefficient
        */
        template<class value_type>
        struct LocalMatrix<value_type, 1>
        {
            using Type = value_type;
        };
    }

    template<class matrix_type>
    matrix_type eye()
    {
        static constexpr auto s = typename matrix_type::shape_type();
        return xt::eye(s[0]);
    }

    template<>
    double eye<double>()
    {
        return 1;
    }




    

    template<class cfg, class Field>
    class PetscCellBasedSchemeAssembly : public PetscAssembly
    {
    public:
        using cfg_t = cfg;

        using Mesh = typename Field::mesh_t;
        using field_value_type = typename Field::value_type;
        static constexpr std::size_t field_size = Field::size;
        using local_matrix_t = typename detail::LocalMatrix<field_value_type, field_size>::Type;
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;

        using stencil_t = Stencil<cfg::scheme_stencil_size, dim>;
        using GetCoefficientsFunc = std::function<std::array<local_matrix_t, cfg::scheme_stencil_size>(double)>;
        using boundary_condition_t = typename Field::boundary_condition_t;
    
        using PetscAssembly::assemble_matrix;
    protected:
        Mesh& _mesh;
        std::size_t _n_cells;
        stencil_t _stencil;
        GetCoefficientsFunc _get_coefficients;
        const std::vector<boundary_condition_t>& _boundary_conditions;
    public:
        PetscCellBasedSchemeAssembly(Mesh& m, stencil_t s, GetCoefficientsFunc get_coeffs, const std::vector<boundary_condition_t>& boundary_conditions) :
            _mesh(m), _stencil(s), _get_coefficients(get_coeffs), _boundary_conditions(boundary_conditions)
        {
            _n_cells = _mesh.nb_cells();
        }

        auto& mesh() const
        {
            return _mesh;
        }

        auto& stencil()
        {
            return _stencil;
        }

        const auto& boundary_conditions()
        {
            return _boundary_conditions;
        }

        PetscInt matrix_size() const override
        {
            return static_cast<PetscInt>(_n_cells * field_size);
        }

    private:
        // Global data index
        inline PetscInt data_index(PetscInt cell_index, unsigned int field_i) const
        {
            if constexpr (field_size == 1)
            {
                return cell_index;
            }
            else if constexpr (Field::is_soa)
            {
                return static_cast<PetscInt>(field_i * _n_cells) + cell_index;
            }
            else
            {
                return static_cast<PetscInt>(cell_index * field_size + field_i);
            }
        }
        template <class CellT>
        inline auto data_index(const CellT& cell, unsigned int field_i) const
        {
            if constexpr (field_size == 1)
            {
                return cell.index;
            }
            else if constexpr (Field::is_soa)
            {
                return field_i * _n_cells + cell.index;
            }
            else
            {
                return cell.index * field_size + field_i;
            }
        }

        // Data index in the given stencil
        inline auto local_data_index(unsigned int cell_local_index, unsigned int field_i) const
        {
            if constexpr (field_size == 1)
            {
                return cell_local_index;
            }
            else if constexpr (Field::is_soa)
            {
                return field_i * cfg::scheme_stencil_size + cell_local_index;
            }
            else
            {
                return cell_local_index * field_size + field_i;
            }
        }

    public:
        std::vector<PetscInt> sparsity_pattern() const override
        {
            // Number of non-zeros per row. 
            // 1 by default (for the unused ghosts outside of the domain).
            std::vector<PetscInt> nnz(_n_cells * field_size, 1);


            // Cells
            auto coeffs = _get_coefficients(cell_length(0));
            for (unsigned int field_i = 0; field_i < field_size; ++field_i)
            {
                PetscInt scheme_nnz_i = cfg::scheme_stencil_size * field_size;
                if constexpr (Field::is_soa)
                {
                    scheme_nnz_i = 0;
                    for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                    {
                        if constexpr(cfg::contiguous_indices_start > 0)
                        {
                            for (unsigned int c=0; c<cfg::contiguous_indices_start; ++c)
                            {
                                double coeff;
                                if constexpr (field_size == 1)
                                {
                                    coeff = coeffs[c];
                                }
                                else
                                {
                                    coeff = coeffs[c](field_i, field_j);
                                }
                                if (coeff != 0)
                                {
                                    scheme_nnz_i++;
                                }
                            }
                        }
                        if constexpr(cfg::contiguous_indices_size > 0)
                        {
                            for (unsigned int c=0; c<cfg::contiguous_indices_size; ++c)
                            {
                                double coeff;
                                if constexpr (field_size == 1)
                                {
                                    coeff = coeffs[cfg::contiguous_indices_start + c];
                                }
                                else
                                {
                                    coeff = coeffs[cfg::contiguous_indices_start + c](field_i, field_j);
                                }
                                if (coeff != 0)
                                {
                                    scheme_nnz_i += cfg::contiguous_indices_size;
                                    break;
                                }
                            }
                        }
                        if constexpr(cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                        {
                            for (unsigned int c=cfg::contiguous_indices_start + cfg::contiguous_indices_size; c<cfg::scheme_stencil_size; ++c)
                            {
                                double coeff;
                                if constexpr (field_size == 1)
                                {
                                    coeff = coeffs[c];
                                }
                                else
                                {
                                    coeff = coeffs[c](field_i, field_j);
                                }
                                if (coeff != 0)
                                {
                                    scheme_nnz_i++;
                                }
                            }
                        }

                    }
                }
                for_each_cell(_mesh, [&](auto& cell)
                {
                    nnz[data_index(cell, field_i)] = scheme_nnz_i;
                });
            }


            // Boundary ghosts on the Dirichlet boundary (if Elimination, nnz=1, the default value)
            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Equation)
            {
                for_each_stencil_center_and_outside_ghost(_mesh, _stencil, [&](const auto& cells, const auto& towards_ghost)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(_boundary_conditions, boundary_point);
                    if (bc.is_dirichlet())
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            nnz[data_index(ghost, field_i)] = 2;
                        }
                    }
                });
            }

            // Boundary ghosts on the Neumann boundary
            if (has_neumann(_boundary_conditions))
            {
                for_each_stencil_center_and_outside_ghost(_mesh, _stencil, [&](const auto& cells, const auto& towards_ghost)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(_boundary_conditions, boundary_point);
                    if (bc.is_neumann())
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            nnz[data_index(ghost, field_i)] = 2;
                        }
                    }
                });
            }

            // Projection
            for_each_projection_ghost(_mesh, [&](auto& ghost)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    nnz[data_index(ghost, field_i)] = cfg::proj_stencil_size;
                }
            });

            // Prediction
            for_each_prediction_ghost(_mesh, [&](auto& ghost)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    nnz[data_index(ghost, field_i)] = cfg::pred_stencil_size;
                }
            });

            return nnz;
        }

    private:
        void assemble_scheme_on_uniform_grid(Mat& A) const override
        {
            // Add 1 on the diagonal
            for (PetscInt i = 0; i < matrix_size(); ++i)
            {
                MatSetValue(A, i, i, 1, INSERT_VALUES);
            }

            // Apply the given coefficents to the given stencil
            for_each_stencil(_mesh, _stencil, _get_coefficients,
            [&] (const auto& cells, const auto& coeffs)
            {
                // Global indices
                std::array<PetscInt, cfg::scheme_stencil_size * field_size> indices;
                for (unsigned int c = 0; c < cfg::scheme_stencil_size; ++c)
                {
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        indices[local_data_index(c, field_i)] = static_cast<PetscInt>(data_index(cells[c], field_i));
                    }
                }

                // The stencil coefficients are stored as an array of matrices.
                // For instance, vector diffusion in 2D:
                //
                //                       L     R     C     B     T        (left, right, center, bottom, top)
                //     field_i -->    |-1   |-1   | 4   |-1   |-1   |
                //     field_j -->    |   -1|   -1|    4|   -1|   -1|
                
                // Coefficient insertion
                if constexpr(field_size == 1 || Field::is_soa)
                {
                    // In SOA, the indices are ordered in field_i for all cells, then field_j for all cells:
                    //
                    //            [         field_i        |         field_j        ]
                    //            [  L    R    C    B    T |  L    R    C    B    T ]
                    //  coupling: [ i j| i j| i j| i j| i j| i j| i j| i j| i j| i j]
                    //            [-1 0|-1 0| 4 0|-1 0|-1 0|0 -1|0 -1|0  4|0 -1|0 -1]
                    //
                    // For the cell of global index c:
                    //
                    //                field_i       ...       field_j 
                    //   row c*i: |-1 -1  4 -1 -1|  ...  | 0  0  0  0  0|
                    //
                    //   row c*j: | 0  0  0  0  0|  ...  |-1 -1  4 -1 -1|
                    //                |_______|              |_______|
                    //               contiguous              contiguous
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                        {

                            if constexpr(cfg::contiguous_indices_start > 0)
                            {
                                for (unsigned int c=0; c<cfg::contiguous_indices_start; ++c)
                                {
                                    double coeff;
                                    if constexpr (field_size == 1)
                                    {
                                        coeff = coeffs[c];
                                    }
                                    else
                                    {
                                        coeff = coeffs[c](field_i, field_j);
                                    }
                                    if (coeff != 0)
                                    {
                                        MatSetValue(A, indices[local_data_index(cfg::center_index, field_i)], indices[local_data_index(c, field_j)], coeff, INSERT_VALUES);
                                    }
                                }
                            }
                            if constexpr(cfg::contiguous_indices_size > 0)
                            {
                                std::array<double, cfg::contiguous_indices_size> contiguous_coeffs;
                                for (unsigned int c=0; c<cfg::contiguous_indices_size; ++c)
                                {
                                    if constexpr (field_size == 1)
                                    {
                                        contiguous_coeffs[c] = coeffs[cfg::contiguous_indices_start + c];
                                    }
                                    else
                                    {
                                        contiguous_coeffs[c] = coeffs[cfg::contiguous_indices_start + c](field_i, field_j);
                                    }
                                    
                                }
                                if (std::any_of(contiguous_coeffs.begin(), contiguous_coeffs.end(), [](auto coeff){ return coeff != 0; }))
                                {
                                    MatSetValues(A, 1, &indices[local_data_index(cfg::center_index, field_i)], static_cast<PetscInt>(cfg::contiguous_indices_size), &indices[local_data_index(cfg::contiguous_indices_start, field_j)], contiguous_coeffs.data(), INSERT_VALUES);
                                }
                            }
                            if constexpr(cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                            {
                                for (unsigned int c=cfg::contiguous_indices_start + cfg::contiguous_indices_size; c<cfg::scheme_stencil_size; ++c)
                                {
                                    double coeff;
                                    if constexpr (field_size == 1)
                                    {
                                        coeff = coeffs[c];
                                    }
                                    else
                                    {
                                        coeff = coeffs[c](field_i, field_j);
                                    }
                                    if (coeff != 0)
                                    {
                                        MatSetValue(A, indices[local_data_index(cfg::center_index, field_i)], indices[local_data_index(c, field_j)], coeff, INSERT_VALUES);
                                    }
                                }
                            }

                        }
                    }
                }
                else // AOS
                {
                    // In AOS, the indices are ordered as
                    //
                    //              L     |     R     |     C     |     B     |     T    
                    //         ii ij ji jj|ii ij ji jj|ii ij ji jj|ii ij ji jj|ii ij ji jj
                    //        [-1  0  0 -1|-1  0  0 -1| 4  0  0  4|-1  0  0 -1|-1  0  0 -1]
                    //
                    //                     i  j  i  j  i  j  i  j  i  j
                    // row (c*2)+i   --> [-1  0|-1  0| 4  0|-1  0|-1  0]
                    // row (c*2)+i+1 --> [ 0 -1| 0 -1| 0  4| 0 -1| 0 -1]

                    for (unsigned int c=0; c<cfg::scheme_stencil_size; ++c)
                    {
                        std::array<double, field_size*field_size> contiguous_coeffs; // matrix row major
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                            {
                                contiguous_coeffs[field_i * field_size + field_j] = coeffs[c](field_i, field_j);
                            }
                        }

                        MatSetValues(A, static_cast<PetscInt>(field_size), &indices[local_data_index(cfg::center_index, 0)], static_cast<PetscInt>(field_size), &indices[local_data_index(c, 0)], contiguous_coeffs.data(), INSERT_VALUES);
                    }
                }
            });
        }

        void assemble_boundary_conditions(Mat& A) const override
        {
            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
            {
                // Must flush to use ADD_VALUES instead of INSERT_VALUES
                MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
            }

            for_each_stencil_center_and_outside_ghost(_mesh, _stencil, _get_coefficients, 
            [&] (const auto& cells, const auto& towards_ghost, auto& ghost_coeff)
            {
                const auto& cell  = cells[0];
                const auto& ghost = cells[1];
                auto boundary_point = cell.face_center(towards_ghost);
                auto bc = find(_boundary_conditions, boundary_point);

                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    PetscInt cell_index = static_cast<PetscInt>(data_index(cell, field_i));
                    PetscInt ghost_index = static_cast<PetscInt>(data_index(ghost, field_i));
                    double coeff;
                    if constexpr (field_size == 1)
                    {
                        coeff = ghost_coeff;
                    }
                    else
                    {
                        coeff = ghost_coeff(field_i, field_i);
                    }
                    coeff = coeff == 0 ? 1 : coeff;
                    if (bc.is_dirichlet())
                    {
                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            MatSetValue(A, cell_index, cell_index,  -coeff, ADD_VALUES); // the coeff is added to the center of the stencil
                            MatSetValue(A, cell_index, ghost_index, -coeff, ADD_VALUES); // the coeff of the ghost is removed from the stencil (we want 0 so we substract the coeff we set before)
                        }
                        else
                        {
                            // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is [  1/2    1/2 ] = dirichlet_value
                            // which is equivalent to                                                         [-coeff -coeff] = -2 * coeff * dirichlet_value
                            MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                            MatSetValue(A, ghost_index, cell_index , -coeff, INSERT_VALUES);
                        }
                    }
                    else
                    {
                        // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is [  1/h  -1/h ] = neumann_value             
                        // However, to have symmetry, we want to have coeff as the off-diagonal coefficient, so     [-coeff coeff] = -coeff * h * neumann_value
                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            MatSetValue(A, ghost_index, ghost_index, -coeff -1, ADD_VALUES); // We want -coeff in the matrix, but we added 1 before, so we remove it
                            MatSetValue(A, ghost_index, cell_index,   coeff   , ADD_VALUES);
                        }
                        else
                        {
                            MatSetValue(A, ghost_index, ghost_index, -coeff, INSERT_VALUES);
                            MatSetValue(A, ghost_index, cell_index,   coeff, INSERT_VALUES);
                        }
                    }
                }
            });

            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
            {
                // Must flush to use INSERT_VALUES instead of ADD_VALUES
                MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
            }
        }


    public:
        virtual void enforce_bc(Vec& b, const Field& solution) const
        {
            for_each_stencil_center_and_outside_ghost(solution.mesh(), _stencil, _get_coefficients,
            [&] (const auto& cells, const auto& towards_ghost, auto& ghost_coeff)
            {
                auto& cell  = cells[0];
                auto& ghost = cells[1];
                auto boundary_point = cell.face_center(towards_ghost);
                auto bc = find(solution.boundary_conditions(), boundary_point);

                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    PetscInt cell_index = static_cast<PetscInt>(data_index(cell, field_i));
                    PetscInt ghost_index = static_cast<PetscInt>(data_index(ghost, field_i));
                    double coeff;
                    if constexpr (field_size == 1)
                    {
                        coeff = ghost_coeff;
                    }
                    else
                    {
                        coeff = ghost_coeff(field_i, field_i);
                    }
                    coeff = coeff == 0 ? 1 : coeff;

                    if (bc.is_dirichlet())
                    {
                        double dirichlet_value;
                        if constexpr (Field::size == 1)
                        {
                            dirichlet_value = bc.get_value(boundary_point);
                        }
                        else
                        {
                            dirichlet_value = bc.get_value(boundary_point)(field_i); // TODO: call get_value() only once instead of once per field_i
                        }
                        
                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            VecSetValue(b, cell_index, - 2 * coeff * dirichlet_value, ADD_VALUES);
                        }
                        else
                        {
                            VecSetValue(b, ghost_index, - 2 * coeff * dirichlet_value, INSERT_VALUES);
                        }
                    }
                    else
                    {
                        auto& h = cell.length;
                        double neumann_value;
                        if constexpr (Field::size == 1)
                        { 
                            neumann_value = bc.get_value(boundary_point);
                        }
                        else
                        {
                            neumann_value = bc.get_value(boundary_point)(field_i); // TODO: call get_value() only once instead of once per field_i
                        }
                        VecSetValue(b, ghost_index, -coeff * h * neumann_value, INSERT_VALUES);
                    }
                }
            });

            // Projection
            for_each_projection_ghost(solution.mesh(), [&](auto& ghost)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    VecSetValue(b, static_cast<PetscInt>(data_index(ghost, field_i)), 0, INSERT_VALUES);
                }
            });

            // Prediction
            for_each_prediction_ghost(solution.mesh(), [&](auto& ghost)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    VecSetValue(b, static_cast<PetscInt>(data_index(ghost, field_i)), 0, INSERT_VALUES);
                }
            });
        }


    private:
        void assemble_projection(Mat& A) const override
        {
            static constexpr PetscInt number_of_children = (1 << dim);

            for_each_projection_ghost_and_children_cells<PetscInt>(_mesh, 
            [&] (PetscInt ghost, const std::array<PetscInt, number_of_children>& children)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    PetscInt ghost_index = data_index(ghost, field_i);
                    MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);
                    for (unsigned int i=0; i<number_of_children; ++i)
                    {
                        MatSetValue(A, ghost_index, data_index(children[i], field_i), -1./number_of_children, INSERT_VALUES);
                    }
                }
            });
        }

        void assemble_prediction(Mat& A) const override
        {
            static_assert(dim >= 1 && dim <= 3, "assemble_prediction() is not implemented for this dimension.");
            if constexpr (dim == 1)
            {
                assemble_prediction_1D(A);
            }
            else if constexpr (dim == 2)
            {
                assemble_prediction_2D(A);
            }
            else if constexpr (dim == 3)
            {
                assemble_prediction_3D(A);
            }
        }

        void assemble_prediction_1D(Mat& A) const
        {
            std::array<double, 3> pred{{1./8, 0, -1./8}};
            for_each_prediction_ghost(_mesh, [&](auto& ghost)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    PetscInt ghost_index = static_cast<PetscInt>(data_index(ghost, field_i));
                    MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                    auto ii = ghost.indices(0);
                    int sign_i = (ii & 1) ? -1 : 1;

                    auto parent_index = data_index(static_cast<PetscInt>(_mesh.get_index(ghost.level - 1, ii/2)), field_i);
                    auto parent_left  = data_index(parent_index - 1, field_i);
                    auto parent_right = data_index(parent_index + 1, field_i);
                    MatSetValue(A, ghost_index, parent_index,                -1, INSERT_VALUES);
                    MatSetValue(A, ghost_index, parent_left,  -sign_i * pred[0], INSERT_VALUES);
                    MatSetValue(A, ghost_index, parent_right, -sign_i * pred[2], INSERT_VALUES);
                }
            });

            /*using mesh_id_t = typename Mesh::mesh_id_t;

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();
            for(std::size_t level=min_level+1; level<=max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                                mesh[mesh_id_t::cells][level-1])
                        .on(level);

                //std::array<double, 3> pred{{1./8, 0, -1./8}};
                set([&](const auto& i, const auto&)
                {
                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_cell = static_cast<int>(mesh.get_index(level, ii));
                        MatSetValue(A, i_cell, i_cell, 1., INSERT_VALUES);

                        int sign_i = (ii & 1)? -1: 1;

                        for(int is = -1; is<2; ++is)
                        {
                            auto i1 = static_cast<int>(mesh.get_index(level - 1, (ii>>1) + is));
                            double v = -sign_i*pred[static_cast<unsigned int>(is + 1)];
                            MatSetValue(A, i_cell, i1, v, INSERT_VALUES);
                        }

                        auto i0 = static_cast<int>(mesh.get_index(level - 1, (ii>>1)));
                        MatSetValue(A, i_cell, i0, -1., INSERT_VALUES);
                    }
                });
            }*/
        }

        void assemble_prediction_2D(Mat& A) const
        {
            std::array<double, 3> pred{{1./8, 0, -1./8}};
            for_each_prediction_ghost(_mesh, [&](auto& ghost)
            {
                for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                {
                    PetscInt ghost_index = static_cast<PetscInt>(data_index(ghost, field_i));
                    MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                    auto ii = ghost.indices(0);
                    auto  j = ghost.indices(1);
                    int sign_i = (ii & 1) ? -1 : 1;
                    int sign_j =  (j & 1) ? -1 : 1;

                    auto parent              = data_index(static_cast<PetscInt>(_mesh.get_index(ghost.level - 1, ii/2, j/2    )), field_i);
                    auto parent_bottom       = data_index(static_cast<PetscInt>(_mesh.get_index(ghost.level - 1, ii/2, j/2 - 1)), field_i);
                    auto parent_top          = data_index(static_cast<PetscInt>(_mesh.get_index(ghost.level - 1, ii/2, j/2 + 1)), field_i);
                    auto parent_left         = data_index(parent - 1       , field_i);
                    auto parent_right        = data_index(parent + 1       , field_i);
                    auto parent_bottom_left  = data_index(parent_bottom - 1, field_i);
                    auto parent_bottom_right = data_index(parent_bottom + 1, field_i);
                    auto parent_top_left     = data_index(parent_top - 1   , field_i);
                    auto parent_top_right    = data_index(parent_top + 1   , field_i);

                    MatSetValue(A, ghost_index, parent             ,                                  -1, INSERT_VALUES);
                    MatSetValue(A, ghost_index, parent_bottom      ,                   -sign_j * pred[0], INSERT_VALUES); //        sign_j * -1/8
                    MatSetValue(A, ghost_index, parent_top         ,                   -sign_j * pred[2], INSERT_VALUES); //        sign_j *  1/8
                    MatSetValue(A, ghost_index, parent_left        ,                   -sign_i * pred[0], INSERT_VALUES); // sign_i        * -1/8
                    MatSetValue(A, ghost_index, parent_right       ,                   -sign_i * pred[2], INSERT_VALUES); // sign_i        *  1/8
                    MatSetValue(A, ghost_index, parent_bottom_left , sign_i * sign_j * pred[0] * pred[0], INSERT_VALUES); // sign_i*sign_j *  1/64
                    MatSetValue(A, ghost_index, parent_bottom_right, sign_i * sign_j * pred[2] * pred[0], INSERT_VALUES); // sign_i*sign_j * -1/64
                    MatSetValue(A, ghost_index, parent_top_left    , sign_i * sign_j * pred[0] * pred[2], INSERT_VALUES); // sign_i*sign_j * -1/64
                    MatSetValue(A, ghost_index, parent_top_right   , sign_i * sign_j * pred[2] * pred[2], INSERT_VALUES); // sign_i*sign_j *  1/64
                }
            });

            /*using mesh_id_t = typename Mesh::mesh_id_t;

            auto min_level = _mesh[mesh_id_t::cells].min_level();
            auto max_level = _mesh[mesh_id_t::cells].max_level();
            for(std::size_t level=min_level+1; level<=max_level; ++level)
            {
                auto set = intersection(_mesh[mesh_id_t::cells_and_ghosts][level],
                                        _mesh[mesh_id_t::cells][level-1])
                        .on(level);

                std::array<double, 3> pred{{1./8, 0, -1./8}};
                set([&](const auto& i, const auto& index)
                {
                    auto j = index[0];
                    int sign_j = (j & 1)? -1: 1;

                    for(int ii=i.start; ii<i.end; ++ii)
                    {
                        auto i_cell = static_cast<PetscInt>(_mesh.get_index(level, ii, j));
                        MatSetValue(A, i_cell, i_cell, 1, INSERT_VALUES);

                        int sign_i = (ii & 1)? -1: 1;

                        for(int is = -1; is<2; ++is)
                        {
                            // is = -1 --> parent_bottom --> -sign_j*pred[0]
                            // is =  0 --> parent        --> -sign_j*pred[1]
                            // is =  1 --> parent_top    --> -sign_j*pred[2]
                            auto i1 = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1), (j>>1) + is));
                            MatSetValue(A, i_cell, i1, -sign_j*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);

                            // is = -1 --> parent_left  --> -sign_i*pred[0]
                            // is =  0 --> parent       --> -sign_i*pred[1]
                            // is =  1 --> parent_right --> -sign_i*pred[2]
                            i1 = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                            MatSetValue(A, i_cell, i1, -sign_i*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);
                        }

                        auto parent_bottom_left  = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) - 1));
                        auto parent_bottom_right = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) - 1));
                        auto parent_top_left     = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) + 1));
                        auto parent_top_right    = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) + 1));

                        MatSetValue(A, i_cell, parent_bottom_left , sign_i*sign_j*pred[0]*pred[0], INSERT_VALUES);
                        MatSetValue(A, i_cell, parent_bottom_right, sign_i*sign_j*pred[2]*pred[0], INSERT_VALUES);
                        MatSetValue(A, i_cell, parent_top_left    , sign_i*sign_j*pred[0]*pred[2], INSERT_VALUES);
                        MatSetValue(A, i_cell, parent_top_right   , sign_i*sign_j*pred[2]*pred[2], INSERT_VALUES);

                        auto i0 = static_cast<PetscInt>(_mesh.get_index(level - 1, (ii>>1), (j>>1)));
                        MatSetValue(A, i_cell, i0, -1, INSERT_VALUES);
                    }
                });
            }*/
        }

        void assemble_prediction_3D(Mat&) const
        {
            for_each_prediction_ghost(_mesh, [&](auto&)
            {
                assert(false && "assemble_prediction_3D() not implemented.");
            });
        }


    public:
        template<class Func>
        static double L2Error(const Field& approximate, Func&& exact)
        {
            // In FV, we want only 1 quadrature point.
            // This is equivalent to 
            //       error += pow(exact(cell.center()) - approximate(cell.index), 2) * cell.length;
            GaussLegendre gl(0);

            double error_norm = 0;
            //double solution_norm = 0;
            for_each_cell(approximate.mesh(), [&](const auto& cell)
            {
                error_norm += gl.quadrature<1>(cell, [&](const auto& point)
                {
                    auto e = exact(point) - approximate[cell];
                    double norm_square;
                    if constexpr (Field::size == 1)
                    {
                        norm_square = e * e;
                    }
                    else
                    {
                        norm_square = xt::sum(e * e)();
                    }
                    return norm_square;
                });

                /*solution_norm += gl.quadrature(cell, [&](const auto& point)
                {
                    return pow(exact(point), 2);
                });*/
            });

            error_norm = sqrt(error_norm);
            //solution_norm = sqrt(solution_norm);
            //double relative_error = error_norm/solution_norm;
            //return relative_error;
            return error_norm;
        }
    };

}} // end namespace