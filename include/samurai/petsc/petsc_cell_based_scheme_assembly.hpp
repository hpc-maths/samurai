#pragma once
#include "petsc_assembly.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "../boundary.hpp"

namespace samurai { namespace petsc
{
    template<class cfg, class Field>
    class PetscCellBasedSchemeAssembly : public PetscAssembly
    {
    public:
        using cfg_t = cfg;
        using Mesh = typename Field::mesh_t;
        using mesh_id_t = typename Mesh::mesh_id_t;
        static constexpr std::size_t dim = Mesh::dim;
        using stencil_t = Stencil<cfg::scheme_stencil_size, dim>;
        using GetCoefficientsFunc = std::function<std::array<double, cfg::scheme_stencil_size>(double)>;
        using boundary_condition_t = typename Field::boundary_condition_t;
    
        using PetscAssembly::assemble_matrix;
        Mesh& mesh;
    protected:
        stencil_t _stencil;
        GetCoefficientsFunc _get_coefficients;
        const std::vector<boundary_condition_t>& _boundary_conditions;
    public:
        PetscCellBasedSchemeAssembly(Mesh& m, stencil_t s, GetCoefficientsFunc get_coeffs, const std::vector<boundary_condition_t>& boundary_conditions) :
            mesh(m), _stencil(s), _get_coefficients(get_coeffs), _boundary_conditions(boundary_conditions)
        {}

        auto& stencil()
        {
            return _stencil;
        }

        const auto& boundary_conditions()
        {
            return _boundary_conditions;
        }

    private:
        PetscInt matrix_size() override
        {
            return static_cast<PetscInt>(mesh.nb_cells());
        }


        std::vector<PetscInt> sparsity_pattern() override
        {
            std::size_t n = mesh.nb_cells();

            // Number of non-zeros per row. 
            // 1 by default (for the unused ghosts outside of the domain).
            std::vector<PetscInt> nnz(n, 1);

            // Cells
            for_each_cell(mesh, [&](auto& cell)
            {
                nnz[cell.index] = cfg::scheme_stencil_size;
            });


            // Boundary ghosts on the Dirichlet boundary (if Elimination, nnz=1, the default value)
            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Equation)
            {
                for_each_stencil_center_and_outside_ghost(mesh, _stencil, [&](const auto& cells, const auto& towards_ghost)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(_boundary_conditions, boundary_point);
                    if (bc.is_dirichlet())
                    {
                        nnz[ghost.index] = 2;
                    }
                });
            }

            // Boundary ghosts on the Neumann boundary
            if (has_neumann(_boundary_conditions))
            {
                for_each_stencil_center_and_outside_ghost(mesh, _stencil, [&](const auto& cells, const auto& towards_ghost)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(_boundary_conditions, boundary_point);
                    if (bc.is_neumann())
                    {
                        nnz[ghost.index] = 2;
                    }
                });
            }

            // Projection
            for_each_projection_ghost(mesh, [&](auto& ghost)
            {
                nnz[ghost.index] = cfg::proj_stencil_size;
            });

            // Prediction
            for_each_prediction_ghost(mesh, [&](auto& ghost)
            {
                nnz[ghost.index] = cfg::pred_stencil_size;
            });

            return nnz;
        }


        void assemble_scheme_on_uniform_grid(Mat& A) override
        {
            // Add 1 on the diagonal
            std::size_t n = mesh.nb_cells();
            for (PetscInt i=0; i<static_cast<PetscInt>(n); ++i)
            {
                MatSetValue(A, i, i, 1, INSERT_VALUES);
            }

            // Apply the given coefficents on the given stencil
            for_each_stencil(mesh, _stencil, _get_coefficients,
            [&] (const auto& cells, const std::array<double, cfg::scheme_stencil_size>& coeffs)
            {
                std::array<PetscInt, cfg::scheme_stencil_size> indices;
                std::transform(cells.begin(), cells.end(), indices.begin(), [](auto& cell) { return cell.index; });

                if constexpr(cfg::contiguous_indices_start > 0)
                {
                    for (unsigned int i=0; i<cfg::contiguous_indices_start; ++i)
                    {
                        MatSetValue(A, indices[cfg::center_index], indices[i], coeffs[i], INSERT_VALUES);
                    }
                }

                if constexpr(cfg::contiguous_indices_size > 0)
                {
                    MatSetValues(A, 1, &indices[cfg::center_index], static_cast<PetscInt>(cfg::contiguous_indices_size), &indices[cfg::contiguous_indices_start], &coeffs[cfg::contiguous_indices_start], INSERT_VALUES);
                }

                if constexpr(cfg::contiguous_indices_start + cfg::contiguous_indices_size < cfg::scheme_stencil_size)
                {
                    for (unsigned int i=cfg::contiguous_indices_start + cfg::contiguous_indices_size; i<cfg::scheme_stencil_size; ++i)
                    {
                        MatSetValue(A, indices[cfg::center_index], indices[i], coeffs[i], INSERT_VALUES);
                    }
                }
            });
        }

        void assemble_boundary_conditions(Mat& A) override
        {
            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Equation)
            {
                assemble_boundary_conditions__dirich_by_equation(A);
            }
            else if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
            {
                assemble_boundary_conditions__dirich_by_elimination(A);
            }
        }

        void assemble_boundary_conditions__dirich_by_elimination(Mat& A)
        {
            // Must flush to use ADD_VALUES instead of INSERT_VALUES
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

            for_each_stencil_center_and_outside_ghost(mesh, _stencil, _get_coefficients, 
            [&] (const auto& cells, const auto& towards_ghost, double ghost_coeff)
            {
                const auto& cell  = cells[0];
                const auto& ghost = cells[1];
                auto boundary_point = cell.face_center(towards_ghost);
                auto bc = find(_boundary_conditions, boundary_point);

                if (bc.is_dirichlet())
                {
                    MatSetValue(A, static_cast<PetscInt>(cell.index), static_cast<PetscInt>(cell.index),  -ghost_coeff, ADD_VALUES); // the coeff is added to the center of the stencil
                    MatSetValue(A, static_cast<PetscInt>(cell.index), static_cast<PetscInt>(ghost.index), -ghost_coeff, ADD_VALUES); // the coeff of the ghost is removed from the stencil (we want 0 so we substract the coeff we set before)
                }
                else
                {
                    // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is   [     1/h        -1/h    ] = neumann_value             
                    // However, to have symmetry, we want to have ghost_coeff as the off-diagonal coefficient, so [-ghost_coeff ghost_coeff] = -ghost_coeff * h * neumann_value
                    MatSetValue(A, static_cast<PetscInt>(ghost.index), static_cast<PetscInt>(ghost.index), -ghost_coeff -1, ADD_VALUES); // We want -ghost_coeff in the matrix, but we added 1 before, so we remove it
                    MatSetValue(A, static_cast<PetscInt>(ghost.index), static_cast<PetscInt>(cell.index),   ghost_coeff   , ADD_VALUES);
                }
            });

            // Must flush to use INSERT_VALUES instead of ADD_VALUES
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        }


        void assemble_boundary_conditions__dirich_by_equation(Mat& A)
        {
            for_each_stencil_center_and_outside_ghost(mesh, _stencil, _get_coefficients, 
            [&] (const auto& cells, const auto& towards_ghost, double ghost_coeff)
            {
                const auto& cell  = cells[0];
                const auto& ghost = cells[1];
                auto boundary_point = cell.face_center(towards_ghost);
                auto bc = find(_boundary_conditions, boundary_point);
                ghost_coeff = ghost_coeff == 0 ? 1 : ghost_coeff;

                if (bc.is_dirichlet())
                {
                    // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is [     1/2          1/2    ] = dirichlet_value
                    // which is equivalent to                                                         [-ghost_coeff -ghost_coeff] = -2 * ghost_coeff * dirichlet_value
                    MatSetValue(A, static_cast<PetscInt>(ghost.index), static_cast<PetscInt>(ghost.index), -ghost_coeff, INSERT_VALUES);
                    MatSetValue(A, static_cast<PetscInt>(ghost.index), static_cast<PetscInt>(cell.index) , -ghost_coeff, INSERT_VALUES);
                }
                else
                {
                    // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is   [     1/h        -1/h    ] = neumann_value             
                    // However, to have symmetry, we want to have ghost_coeff as the off-diagonal coefficient, so [-ghost_coeff ghost_coeff] = -ghost_coeff * h * neumann_value
                    MatSetValue(A, static_cast<PetscInt>(ghost.index), static_cast<PetscInt>(ghost.index), -ghost_coeff, INSERT_VALUES);
                    MatSetValue(A, static_cast<PetscInt>(ghost.index), static_cast<PetscInt>(cell.index),   ghost_coeff, INSERT_VALUES);
                }
            });
        }


    public:
        virtual void enforce_bc(Vec& b, const Field& solution)
        {
            if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Equation)
            {
                for_each_stencil_center_and_outside_ghost(solution.mesh(), _stencil, _get_coefficients,
                [&] (const auto& cells, const auto& towards_ghost, double ghost_coeff)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(solution.boundary_conditions(), boundary_point);
                    ghost_coeff = ghost_coeff == 0 ? 1 : ghost_coeff;

                    if (bc.is_dirichlet())
                    {
                        auto dirichlet_value = bc.get_value(boundary_point);
                        VecSetValue(b, static_cast<PetscInt>(ghost.index), - 2 * ghost_coeff * dirichlet_value, INSERT_VALUES);
                    }
                    else
                    {
                        auto& h = cell.length;
                        auto neumann_value = bc.get_value(boundary_point);
                        VecSetValue(b, static_cast<PetscInt>(ghost.index), -ghost_coeff * h * neumann_value, INSERT_VALUES);
                    }
                });
            }
            else if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
            {
                for_each_stencil_center_and_outside_ghost(solution.mesh(), _stencil, _get_coefficients,
                [&] (const auto& cells, const auto& towards_ghost, double ghost_coeff)
                {
                    auto& cell  = cells[0];
                    auto& ghost = cells[1];
                    auto boundary_point = cell.face_center(towards_ghost);
                    auto bc = find(solution.boundary_conditions(), boundary_point);

                    if (bc.is_dirichlet())
                    {
                        auto dirichlet_value = bc.get_value(boundary_point);
                        VecSetValue(b, static_cast<PetscInt>(cell.index), - 2 * ghost_coeff * dirichlet_value, ADD_VALUES);
                    }
                    else
                    {
                        auto& h = cell.length;
                        auto neumann_value = bc.get_value(boundary_point);
                        VecSetValue(b, static_cast<PetscInt>(ghost.index), -ghost_coeff * h * neumann_value, ADD_VALUES);
                    }
                });
            }
        }


    private:
        void assemble_projection(Mat& A) override
        {
            static constexpr PetscInt number_of_children = (1 << dim);

            for_each_projection_ghost_and_children_cells<PetscInt>(mesh, 
            [&] (PetscInt ghost, const std::array<PetscInt, number_of_children>& children)
            {
                MatSetValue(A, ghost, ghost, 1, INSERT_VALUES);
                for (unsigned int i=0; i<number_of_children; ++i)
                {
                    MatSetValue(A, ghost, children[i], -1./number_of_children, INSERT_VALUES);
                }
            });
        }

        void assemble_prediction(Mat& A) override
        {
            assemble_prediction_impl(std::integral_constant<std::size_t, dim>{}, A, mesh);
        }


    public:
        template<class Func>
        static double L2Error(const Field& approximate, Func&& exact, int exact_polynomial_degree)
        {
            GaussLegendre gl(exact_polynomial_degree);
            double error_norm = 0;
            double solution_norm = 0;
            for_each_cell(approximate.mesh(), [&](const auto& cell)
            {
                error_norm += gl.quadrature(cell, [&](const auto& point)
                {
                    return pow(exact(point) - approximate(cell.index), 2);
                });

                solution_norm += gl.quadrature(cell, [&](const auto& point)
                {
                    return pow(exact(point), 2);
                });
            });

            error_norm = sqrt(error_norm);
            solution_norm = sqrt(solution_norm);
            double relative_error = error_norm/solution_norm;
            return relative_error;
        }
    };

}} // end namespace



//-----------------------------//
//     Assemble prediction     //
//          (order 1)          //
//-----------------------------//

// 1D

template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 1>, Mat& A, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                        mesh[mesh_id_t::cells][level-1])
                .on(level);

        std::array<double, 3> pred{{1./8, 0, -1./8}};
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
    }
}


// 2D

template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 2>, Mat& A, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                mesh[mesh_id_t::cells][level-1])
                .on(level);

        std::array<double, 3> pred{{1./8, 0, -1./8}};
        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            int sign_j = (j & 1)? -1: 1;

            for(int ii=i.start; ii<i.end; ++ii)
            {
                auto i_cell = static_cast<PetscInt>(mesh.get_index(level, ii, j));
                MatSetValue(A, i_cell, i_cell, 1, INSERT_VALUES);

                int sign_i = (ii & 1)? -1: 1;

                for(int is = -1; is<2; ++is)
                {
                    auto i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1), (j>>1) + is));
                    MatSetValue(A, i_cell, i1, -sign_j*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);

                    i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + is, (j>>1)));
                    MatSetValue(A, i_cell, i1, -sign_i*pred[static_cast<unsigned int>(is + 1)], INSERT_VALUES);
                }

                auto i1 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) - 1));
                auto i2 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) - 1));
                auto i3 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) - 1, (j>>1) + 1));
                auto i4 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1) + 1, (j>>1) + 1));

                MatSetValue(A, i_cell, i1, sign_i*sign_j*pred[0]*pred[0], INSERT_VALUES);
                MatSetValue(A, i_cell, i2, sign_i*sign_j*pred[2]*pred[0], INSERT_VALUES);
                MatSetValue(A, i_cell, i3, sign_i*sign_j*pred[0]*pred[2], INSERT_VALUES);
                MatSetValue(A, i_cell, i4, sign_i*sign_j*pred[2]*pred[2], INSERT_VALUES);

                auto i0 = static_cast<PetscInt>(mesh.get_index(level - 1, (ii>>1), (j>>1)));
                MatSetValue(A, i_cell, i0, -1, INSERT_VALUES);
            }
        });
    }
}


// 3D

template<class Mesh>
void assemble_prediction_impl(std::integral_constant<std::size_t, 3>, Mat& /*A*/, Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    for(std::size_t level=min_level+1; level<=max_level; ++level)
    {
        auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                mesh[mesh_id_t::cells][level-1])
                .on(level);

        //std::array<double, 3> pred{{1./8, 0, -1./8}};
        set([&](const auto&, const auto&)
        {
            assert(false && "non implemented");
        });
    }
}