#pragma once
#include "../boundary.hpp"
#include "../interface.hpp"
#include "../numeric/gauss_legendre.hpp"
#include "matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Defines how to compute a normal flux: e.g., Grad(u).n
         * - direction: e.g., right = {0, 1}
         * - stencil: e.g., current cell and right neighbour = {{0, 0}, {0, 1}}
         * - function returning the coefficients for the flux computation w.r.t. the stencil:
         *            auto get_flux_coeffs(double h)
         *            {
         *                // Grad(u).n = (u_1 - u_0)/h
         *                std::array<double, 2> flux_coeffs;
         *                flux_coeffs[0] = -1/h; // current cell
         *                flux_coeffs[1] =  1/h; // right neighbour
         *                return flux_coeffs;
         *            }
         */
        template <class Field, std::size_t stencil_size>
        struct NormalFluxComputation
        {
            static constexpr std::size_t dim        = Field::dim;
            static constexpr std::size_t field_size = Field::size;
            using field_value_type                  = typename Field::value_type;                      // double
            using flux_matrix_t = typename detail::LocalMatrix<field_value_type, 1, field_size>::Type; // 'double' if field_size = 1,
                                                                                                       // 'xtensor' representing a matrix
                                                                                                       // otherwise
            using flux_coeffs_t = std::array<flux_matrix_t, stencil_size>;

            DirectionVector<dim> direction;
            Stencil<stencil_size, dim> stencil;
            std::function<flux_coeffs_t(double)> get_flux_coeffs;
        };

        template <class Field, class Vector>
        auto normal_grad_order2(Vector& direction)
        {
            static constexpr std::size_t dim        = Field::dim;
            static constexpr std::size_t field_size = Field::size;
            using flux_computation_t                = NormalFluxComputation<Field, 2>;
            using flux_matrix_t                     = typename flux_computation_t::flux_matrix_t;

            flux_computation_t normal_grad;
            normal_grad.direction       = direction;
            normal_grad.stencil         = in_out_stencil<dim>(direction);
            normal_grad.get_flux_coeffs = [](double h)
            {
                std::array<flux_matrix_t, 2> coeffs;
                if constexpr (field_size == 1)
                {
                    coeffs[0] = -1 / h;
                    coeffs[1] = 1 / h;
                }
                else
                {
                    coeffs[0].fill(-1 / h);
                    coeffs[1].fill(1 / h);
                }
                return coeffs;
            };
            return normal_grad;
        }

        template <class Field>
        auto normal_grad_order2()
        {
            static constexpr std::size_t dim = Field::dim;
            using flux_computation_t         = NormalFluxComputation<Field, 2>;

            auto directions = positive_cartesian_directions<dim>();
            std::array<flux_computation_t, dim> normal_fluxes;
            for (std::size_t d = 0; d < dim; ++d)
            {
                normal_fluxes[d] = normal_grad_order2<Field>(xt::view(directions, d));
            }
            return normal_fluxes;
        }

        template <class Field, std::size_t output_field_size, std::size_t stencil_size>
        struct FluxBasedCoefficients
        {
            static constexpr std::size_t dim        = Field::dim;
            static constexpr std::size_t field_size = Field::size;

            using flux_computation_t = NormalFluxComputation<Field, stencil_size>;
            using field_value_type   = typename Field::value_type; // double
            using coeff_matrix_t     = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;
            using cell_coeffs_t      = std::array<coeff_matrix_t, stencil_size>;
            using flux_coeffs_t      = typename flux_computation_t::flux_coeffs_t; // std::array<flux_matrix_t, stencil_size>;

            flux_computation_t flux;
            std::function<cell_coeffs_t(flux_coeffs_t&, double, double)> get_cell1_coeffs;
            std::function<cell_coeffs_t(flux_coeffs_t&, double, double)> get_cell2_coeffs;
        };

        /**
         * Useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
         */
        template <PetscInt output_field_size_, PetscInt comput_stencil_size_, DirichletEnforcement dirichlet_enfcmt_ = Equation>
        struct FluxBasedAssemblyConfig
        {
            static constexpr PetscInt output_field_size            = output_field_size_;
            static constexpr PetscInt comput_stencil_size          = comput_stencil_size_;
            static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
        };

        template <class cfg, class Field>
        class FluxBasedScheme : public MatrixAssembly
        {
            template <class Scheme1, class Scheme2>
            friend class FluxBasedScheme_Sum_CellBasedScheme;

          public:

            using cfg_t   = cfg;
            using field_t = Field;

            using Mesh                                       = typename Field::mesh_t;
            static constexpr std::size_t field_size          = Field::size;
            static constexpr std::size_t output_field_size   = cfg::output_field_size;
            using mesh_id_t                                  = typename Mesh::mesh_id_t;
            static constexpr std::size_t dim                 = Mesh::dim;
            static constexpr std::size_t prediction_order    = Mesh::config::prediction_order;
            static constexpr std::size_t comput_stencil_size = cfg::comput_stencil_size;

            using coefficients_t       = FluxBasedCoefficients<Field, output_field_size, comput_stencil_size>;
            using boundary_condition_t = typename Field::boundary_condition_t;

            using MatrixAssembly::assemble_matrix;

          protected:

            Field& m_unknown;
            Mesh& m_mesh;
            std::size_t m_n_cells;
            std::array<coefficients_t, dim> m_scheme_coefficients;
            const std::vector<boundary_condition_t>& m_boundary_conditions;
            std::vector<bool> m_is_row_empty;

          public:

            FluxBasedScheme(Field& unknown, std::array<coefficients_t, dim> scheme_coefficients)
                : m_unknown(unknown)
                , m_mesh(unknown.mesh())
                , m_scheme_coefficients(scheme_coefficients)
                , m_boundary_conditions(unknown.boundary_conditions())
            {
                m_n_cells      = m_mesh.nb_cells();
                m_is_row_empty = std::vector(static_cast<std::size_t>(matrix_rows()), true);
            }

            auto& unknown() const
            {
                return m_unknown;
            }

            auto& mesh() const
            {
                return m_mesh;
            }

            auto& scheme_coefficients() const
            {
                return m_scheme_coefficients;
            }

            auto& scheme_coefficients()
            {
                return m_scheme_coefficients;
            }

            const auto& boundary_conditions() const
            {
                return m_boundary_conditions;
            }

            PetscInt matrix_rows() const override
            {
                return static_cast<PetscInt>(m_n_cells * output_field_size);
            }

            PetscInt matrix_cols() const override
            {
                return static_cast<PetscInt>(m_n_cells * field_size);
            }

          protected:

            // Global data index
            inline PetscInt col_index(PetscInt cell_index, unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return cell_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return static_cast<PetscInt>(field_j * m_n_cells) + cell_index;
                }
                else
                {
                    return cell_index * static_cast<PetscInt>(field_size) + static_cast<PetscInt>(field_j);
                }
            }

            inline PetscInt row_index(PetscInt cell_index, unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return cell_index;
                }
                else if constexpr (Field::is_soa)
                {
                    return static_cast<PetscInt>(field_i * m_n_cells) + cell_index;
                }
                else
                {
                    return cell_index * static_cast<PetscInt>(output_field_size) + static_cast<PetscInt>(field_i);
                }
            }

            template <class CellT>
            inline auto col_index(const CellT& cell, unsigned int field_j) const
            {
                if constexpr (field_size == 1)
                {
                    return cell.index;
                }
                else if constexpr (Field::is_soa)
                {
                    return field_j * m_n_cells + cell.index;
                }
                else
                {
                    return cell.index * field_size + field_j;
                }
            }

            template <class CellT>
            inline auto row_index(const CellT& cell, unsigned int field_i) const
            {
                if constexpr (output_field_size == 1)
                {
                    return cell.index;
                }
                else if constexpr (Field::is_soa)
                {
                    return field_i * m_n_cells + cell.index;
                }
                else
                {
                    return cell.index * output_field_size + field_i;
                }
            }

            template <class Coeffs>
            inline double
            cell_coeff(const Coeffs& coeffs, std::size_t cell_number_in_stencil, unsigned int field_i, unsigned int field_j) const
            {
                if constexpr (field_size == 1 && output_field_size == 1)
                {
                    return coeffs[cell_number_in_stencil];
                }
                else
                {
                    return coeffs[cell_number_in_stencil](field_i, field_j);
                }
            }

          public:

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir = m_scheme_coefficients[d];
                    for_each_interior_interface(m_mesh,
                                                scheme_coeffs_dir.flux.direction,
                                                scheme_coeffs_dir.flux.stencil,
                                                [&](auto& interface_cells, auto&)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            nnz[row_index(interface_cells[0], field_i)] += comput_stencil_size * field_size;
                                                            nnz[row_index(interface_cells[1], field_i)] += comput_stencil_size * field_size;
                                                        }
                                                    }
                                                });

                    for_each_boundary_interface(m_mesh,
                                                scheme_coeffs_dir.flux.direction,
                                                scheme_coeffs_dir.flux.stencil,
                                                [&](auto& interface_cells, auto&)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            nnz[row_index(interface_cells[0], field_i)] += comput_stencil_size * field_size;
                                                        }
                                                    }
                                                });

                    auto opposite_direction = xt::eval(-scheme_coeffs_dir.flux.direction);
                    auto opposite_stencil   = xt::eval(-scheme_coeffs_dir.flux.stencil);
                    for_each_boundary_interface(m_mesh,
                                                opposite_direction,
                                                opposite_stencil,
                                                [&](auto& interface_cells, auto&)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            nnz[row_index(interface_cells[0], field_i)] += comput_stencil_size * field_size;
                                                        }
                                                    }
                                                });
                }
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                std::array<Stencil<comput_stencil_size, dim>, 2 * dim> bdry_stencils;
                std::array<DirectionVector<dim>, 2 * dim> bdry_directions;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir   = m_scheme_coefficients[d];
                    bdry_directions[d]       = scheme_coeffs_dir.flux.direction;
                    bdry_stencils[d]         = scheme_coeffs_dir.flux.stencil;
                    bdry_directions[dim + d] = -scheme_coeffs_dir.flux.direction;
                    bdry_stencils[dim + d]   = -scheme_coeffs_dir.flux.stencil;
                }

                for_each_stencil_on_boundary(m_mesh,
                                             bdry_directions,
                                             bdry_stencils,
                                             [&](const auto& cells, const auto& towards_ghost)
                                             {
                                                 auto& cell = cells[0];
                                                 for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                 {
                                                     nnz[row_index(cell, field_i)]++;
                                                 }
                                                 auto boundary_point = cell.face_center(towards_ghost);
                                                 auto bc             = find(m_boundary_conditions, boundary_point);
                                                 for (std::size_t g = 1; g < comput_stencil_size; ++g)
                                                 {
                                                     auto& ghost = cells[g];
                                                     if (bc.is_dirichlet())
                                                     {
                                                         for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                         {
                                                             if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                                                             {
                                                                 nnz[row_index(ghost, field_i)] = 1;
                                                             }
                                                             else
                                                             {
                                                                 nnz[row_index(ghost, field_i)] = 2;
                                                             }
                                                         }
                                                     }
                                                     else
                                                     {
                                                         for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                         {
                                                             nnz[row_index(ghost, field_i)] = 2;
                                                         }
                                                     }
                                                 }
                                             });
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                // ----  Projection stencil size
                // cell + 2^dim children --> 1+2=3 in 1D
                //                           1+4=5 in 2D
                static constexpr std::size_t proj_stencil_size = 1 + (1 << dim);

                for_each_projection_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                              {
                                                  nnz[row_index(ghost, field_i)] = proj_stencil_size;
                                              }
                                          });
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                // ----  Prediction stencil size
                // Order 1: cell + hypercube of 3 coarser cells --> 1 + 3= 4 in 1D
                //                                                  1 + 9=10 in 2D
                // Order 2: cell + hypercube of 5 coarser cells --> 1 + 5= 6 in 1D
                //                                                  1 +25=21 in 2D
                static constexpr std::size_t pred_stencil_size = 1 + ce_pow(2 * prediction_order + 1, dim);

                for_each_prediction_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                              {
                                                  nnz[row_index(ghost, field_i)] = pred_stencil_size;
                                              }
                                          });
            }

          protected:

            void assemble_scheme(Mat& A) override
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir = m_scheme_coefficients[d];
                    for_each_interior_interface(
                        m_mesh,
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        scheme_coeffs_dir.flux.get_flux_coeffs,
                        scheme_coeffs_dir.get_cell1_coeffs,
                        scheme_coeffs_dir.get_cell2_coeffs,
                        [&](auto& interface_cells, auto& comput_cells, auto& cell1_coeffs, auto& cell2_coeffs)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto interface_cell1_row = static_cast<PetscInt>(row_index(interface_cells[0], field_i));
                                auto interface_cell2_row = static_cast<PetscInt>(row_index(interface_cells[1], field_i));
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    for (std::size_t c = 0; c < comput_stencil_size; ++c)
                                    {
                                        auto comput_cell_col = static_cast<PetscInt>(col_index(comput_cells[c], field_j));
                                        double cell1_coeff   = cell_coeff(cell1_coeffs, c, field_i, field_j);
                                        double cell2_coeff   = cell_coeff(cell2_coeffs, c, field_i, field_j);
                                        if (cell1_coeff != 0)
                                        {
                                            MatSetValue(A, interface_cell1_row, comput_cell_col, cell1_coeff, ADD_VALUES);
                                        }
                                        if (cell2_coeff != 0)
                                        {
                                            MatSetValue(A, interface_cell2_row, comput_cell_col, cell2_coeff, ADD_VALUES);
                                        }
                                    }
                                }
                                m_is_row_empty[static_cast<std::size_t>(interface_cell1_row)] = false;
                                m_is_row_empty[static_cast<std::size_t>(interface_cell2_row)] = false;
                            }
                        });

                    for_each_boundary_interface(
                        m_mesh,
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        scheme_coeffs_dir.flux.get_flux_coeffs,
                        scheme_coeffs_dir.get_cell1_coeffs,
                        [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto interface_cell0_row = static_cast<PetscInt>(row_index(interface_cells[0], field_i));
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    for (std::size_t c = 0; c < comput_stencil_size; ++c)
                                    {
                                        double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                        if (coeff != 0)
                                        {
                                            auto comput_cell_col = static_cast<PetscInt>(col_index(comput_cells[c], field_j));
                                            MatSetValue(A, interface_cell0_row, comput_cell_col, coeff, ADD_VALUES);
                                        }
                                    }
                                }
                                m_is_row_empty[static_cast<std::size_t>(interface_cell0_row)] = false;
                            }
                        });

                    auto opposite_direction                    = xt::eval(-scheme_coeffs_dir.flux.direction);
                    Stencil<comput_stencil_size, dim> reversed = xt::eval(xt::flip(scheme_coeffs_dir.flux.stencil, 0));
                    auto opposite_stencil                      = xt::eval(-reversed);
                    /*auto get_coeffs_opposite_direction = [&](FluxBasedCoefficients::Coeffs flux_coeffs, double h)
                    {
                        auto coeffs = scheme_coeffs_dir.get_cell1_coeffs(flux_coeffs, h);
                        for (auto& c : coeffs)
                        {
                            c *= -1;
                        }
                        auto neighbour_coeffs = scheme_coeffs_dir.get_neighbour_coeffs(coeffs);
                        return neighbour_coeffs;
                    };*/
                    for_each_boundary_interface(
                        m_mesh,
                        opposite_direction,
                        opposite_stencil,
                        scheme_coeffs_dir.flux.get_flux_coeffs,
                        scheme_coeffs_dir.get_cell2_coeffs, // get_coeffs_opposite_direction,
                        [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                auto interface_cell0_row = static_cast<PetscInt>(row_index(interface_cells[0], field_i));
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    for (std::size_t c = 0; c < comput_stencil_size; ++c)
                                    {
                                        double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                        if (coeff != 0)
                                        {
                                            auto comput_cell_col = static_cast<PetscInt>(col_index(comput_cells[c], field_j));
                                            MatSetValue(A, interface_cell0_row, comput_cell_col, coeff, ADD_VALUES);
                                        }
                                    }
                                }
                                m_is_row_empty[static_cast<std::size_t>(interface_cell0_row)] = false;
                            }
                        });
                }
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                std::array<Stencil<comput_stencil_size, dim>, 2 * dim> bdry_stencils;
                std::array<DirectionVector<dim>, 2 * dim> bdry_directions;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir   = m_scheme_coefficients[d];
                    bdry_directions[d]       = scheme_coeffs_dir.flux.direction;
                    bdry_stencils[d]         = scheme_coeffs_dir.flux.stencil;
                    bdry_directions[dim + d] = -scheme_coeffs_dir.flux.direction;
                    bdry_stencils[dim + d]   = -scheme_coeffs_dir.flux.stencil;
                }
                auto get_coeffs = [&](double h)
                {
                    auto flux_coeffs = m_scheme_coefficients[0].flux.get_flux_coeffs(h);
                    return m_scheme_coefficients[0].get_cell1_coeffs(flux_coeffs, h, h);
                };

                for_each_stencil_on_boundary(
                    m_mesh,
                    bdry_directions,
                    bdry_stencils,
                    get_coeffs,
                    [&](const auto& cells, const auto& towards_ghost, auto& coeffs)
                    {
                        const auto& cell    = cells[0];
                        auto boundary_point = cell.face_center(towards_ghost);
                        auto bc             = find(m_boundary_conditions, boundary_point);

                        for (std::size_t g = 1; g < comput_stencil_size; ++g)
                        {
                            const auto& ghost = cells[g];
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                                PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                                double coeff = cell_coeff(coeffs, g, field_i, field_i);
                                // Add missing flux to the cell
                                // MatSetValue(A, cell_index, cell_index , cell_coeff , ADD_VALUES);
                                // MatSetValue(A, cell_index, ghost_index, coeff, ADD_VALUES);

                                if (bc.is_dirichlet())
                                {
                                    if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                                    {
                                        // We have (u_ghost + u_cell)/2 = dirichlet_value ==> u_ghost = 2*dirichlet_value - u_cell
                                        // The equation on the cell row is
                                        //                     coeff*u_ghost + coeff_cell*u_cell + ... = f
                                        // Eliminating u_ghost, it gives
                                        //                           (coeff_cell - coeff)*u_cell + ... = f - 2*coeff*dirichlet_value
                                        // which means that:
                                        // - on the cell row, we have to 1) remove the coeff in the column of the ghost,
                                        //                               2) substract coeff in the column of the cell.
                                        // - on the cell row of the right-hand side, we have to add -2*coeff*dirichlet_value.
                                        MatSetValue(A, cell_index, ghost_index, -coeff, ADD_VALUES); // the coeff of the ghost is removed
                                                                                                     // from the stencil (we want 0 so we
                                                                                                     // substract the coeff we set before)
                                        MatSetValue(A, cell_index, cell_index, -coeff, ADD_VALUES);  // the coeff is substracted from the
                                                                                                     // center of the stencil
                                        MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES); // 1 is added to the diagonal of the ghost
                                    }
                                    else
                                    {
                                        coeff = coeff == 0 ? 1 : coeff;
                                        // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is [  1/2    1/2 ] =
                                        // dirichlet_value which is equivalent to [-coeff -coeff] = -2 * coeff * dirichlet_value
                                        MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES);
                                        MatSetValue(A, ghost_index, cell_index, -coeff, ADD_VALUES);
                                    }
                                }
                                else
                                {
                                    coeff = coeff == 0 ? 1 : coeff;
                                    // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is [  1/h  -1/h
                                    // ] = neumann_value However, to have symmetry, we want to have coeff as the off-diagonal coefficient,
                                    // so     [-coeff coeff] = -coeff * h * neumann_value
                                    if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                                    {
                                        MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES);
                                        MatSetValue(A, ghost_index, cell_index, coeff, ADD_VALUES);
                                    }
                                    else
                                    {
                                        MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES);
                                        MatSetValue(A, ghost_index, cell_index, coeff, ADD_VALUES);
                                    }
                                }

                                m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                            }
                        }
                    });
            }

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        MatSetValue(A, static_cast<PetscInt>(i), static_cast<PetscInt>(i), 1, ADD_VALUES);
                        m_is_row_empty[i] = false;
                    }
                }
            }

          public:

            virtual void enforce_bc(Vec& b) const
            {
                std::array<Stencil<comput_stencil_size, dim>, 2 * dim> bdry_stencils;
                std::array<DirectionVector<dim>, 2 * dim> bdry_directions;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto scheme_coeffs_dir   = m_scheme_coefficients[d];
                    bdry_directions[d]       = scheme_coeffs_dir.flux.direction;
                    bdry_stencils[d]         = scheme_coeffs_dir.flux.stencil;
                    bdry_directions[dim + d] = -scheme_coeffs_dir.flux.direction;
                    bdry_stencils[dim + d]   = -scheme_coeffs_dir.flux.stencil;
                }
                auto get_coeffs = [&](double h)
                {
                    auto flux_coeffs = m_scheme_coefficients[0].flux.get_flux_coeffs(h);
                    return m_scheme_coefficients[0].get_cell1_coeffs(flux_coeffs, h, h);
                };

                for_each_stencil_on_boundary(m_mesh,
                                             bdry_directions,
                                             bdry_stencils,
                                             get_coeffs,
                                             [&](const auto& cells, const auto& towards_ghost, auto& coeffs)
                                             {
                                                 auto& cell          = cells[0];
                                                 auto boundary_point = cell.face_center(towards_ghost);
                                                 auto bc             = find(m_boundary_conditions, boundary_point);
                                                 for (std::size_t g = 1; g < comput_stencil_size; ++g)
                                                 {
                                                     const auto& ghost = cells[g];
                                                     for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                     {
                                                         PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                                                         PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                                                         double coeff = cell_coeff(coeffs, g, field_i, field_i);
                                                         if (bc.is_dirichlet())
                                                         {
                                                             double dirichlet_value;
                                                             if constexpr (field_size == 1)
                                                             {
                                                                 dirichlet_value = bc.get_value(boundary_point);
                                                             }
                                                             else
                                                             {
                                                                 dirichlet_value = bc.get_value(boundary_point)(field_i); // TODO: call
                                                                                                                          // get_value()
                                                                                                                          // only once
                                                                                                                          // instead of once
                                                                                                                          // per field_i
                                                             }

                                                             if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                                                             {
                                                                 // std::cout << "ADD " << (- 2 * coeff * dirichlet_value) << " to row " <<
                                                                 // cell_index << " (field " << field_i << ", cell " << cell.index << ",
                                                                 // ghost " << ghost.index << ")" << std::endl;
                                                                 VecSetValue(b, cell_index, -2 * coeff * dirichlet_value, ADD_VALUES);
                                                             }
                                                             else
                                                             {
                                                                 coeff = coeff == 0 ? 1 : coeff;
                                                                 VecSetValue(b, ghost_index, -2 * coeff * dirichlet_value, INSERT_VALUES);
                                                             }
                                                         }
                                                         else
                                                         {
                                                             coeff   = coeff == 0 ? 1 : coeff;
                                                             auto& h = cell.length;
                                                             double neumann_value;
                                                             if constexpr (field_size == 1)
                                                             {
                                                                 neumann_value = bc.get_value(boundary_point);
                                                             }
                                                             else
                                                             {
                                                                 neumann_value = bc.get_value(boundary_point)(field_i); // TODO: call
                                                                                                                        // get_value() only
                                                                                                                        // once instead of
                                                                                                                        // once per field_i
                                                             }
                                                             // std::cout << "ADD " << (- coeff * h * neumann_value) << " to row " <<
                                                             // ghost_index << " (field " << field_i << ", cell " << cell.index << ", ghost
                                                             // " << ghost.index << ")" << std::endl;
                                                             VecSetValue(b, ghost_index, -coeff * h * neumann_value, INSERT_VALUES);
                                                         }
                                                     }
                                                 }
                                             });
            }

            void add_0_for_useless_ghosts(Vec& b) const
            {
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        VecSetValue(b, static_cast<PetscInt>(i), 0, INSERT_VALUES);
                    }
                }
            }

            virtual void enforce_projection_prediction(Vec& b) const
            {
                // Projection
                for_each_projection_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                                              {
                                                  VecSetValue(b, static_cast<PetscInt>(col_index(ghost, field_i)), 0, INSERT_VALUES);
                                              }
                                          });

                // Prediction
                for_each_prediction_ghost(m_mesh,
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                                              {
                                                  VecSetValue(b, static_cast<PetscInt>(col_index(ghost, field_i)), 0, INSERT_VALUES);
                                              }
                                          });
            }

          private:

            void assemble_projection(Mat& A) override
            {
                static constexpr PetscInt number_of_children = (1 << dim);

                for_each_projection_ghost_and_children_cells<PetscInt>(
                    m_mesh,
                    [&](PetscInt ghost, const std::array<PetscInt, number_of_children>& children)
                    {
                        for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            PetscInt ghost_index = row_index(ghost, field_i);
                            MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);
                            for (unsigned int i = 0; i < number_of_children; ++i)
                            {
                                MatSetValue(A, ghost_index, col_index(children[i], field_i), -1. / number_of_children, ADD_VALUES);
                            }
                            m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

            void assemble_prediction(Mat& A) override
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

            void assemble_prediction_1D(Mat& A)
            {
                /*std::array<double, 3> pred{{1./8, 0, -1./8}};
                for_each_prediction_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt ghost_index = static_cast<PetscInt>(row_index(ghost, field_i));
                        MatSetValue(A, ghost_index, ghost_index, 1, INSERT_VALUES);

                        auto ii = ghost.indices(0);
                        int sign_i = (ii & 1) ? -1 : 1;

                        auto parent_index = col_index(static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2)), field_i);
                        auto parent_left  = col_index(parent_index - 1, field_i);
                        auto parent_right = col_index(parent_index + 1, field_i);
                        MatSetValue(A, ghost_index, parent_index,                -1, INSERT_VALUES);
                        MatSetValue(A, ghost_index, parent_left,  -sign_i * pred[0], INSERT_VALUES);
                        MatSetValue(A, ghost_index, parent_right, -sign_i * pred[2], INSERT_VALUES);
                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                });*/
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    m_mesh,
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                            MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            double isign = (ii & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig)), field_i);
                            MatSetValue(A, ghost_index, parent_index, -1, ADD_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                if (ci != prediction_order)
                                {
                                    double value           = -interpx[ci];
                                    auto coarse_cell_index = this->col_index(
                                        static_cast<PetscInt>(
                                            this->m_mesh.get_index(ghost.level - 1, ig + static_cast<index_t>(ci - prediction_order))),
                                        field_i);
                                    MatSetValue(A, ghost_index, coarse_cell_index, value, ADD_VALUES);
                                }
                            }
                            this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

            void assemble_prediction_2D(Mat& A)
            {
                /*std::array<double, 3> pred{{1./8, 0, -1./8}};
                for_each_prediction_ghost(m_mesh, [&](auto& ghost)
                {
                    for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                    {
                        PetscInt ghost_index = static_cast<PetscInt>(row_index(ghost, field_i));
                        MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);

                        auto ii = ghost.indices(0);
                        auto  j = ghost.indices(1);
                        int sign_i = (ii & 1) ? -1 : 1;
                        int sign_j =  (j & 1) ? -1 : 1;

                        auto parent_index        = static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2, j/2    ));
                        auto parent_bottom_index = static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2, j/2 - 1));
                        auto parent_top_index    = static_cast<PetscInt>(m_mesh.get_index(ghost.level - 1, ii/2, j/2 + 1));
                        auto parent              = col_index(parent_index           , field_i);
                        auto parent_bottom       = col_index(parent_bottom_index    , field_i);
                        auto parent_top          = col_index(parent_top_index       , field_i);
                        auto parent_left         = col_index(parent_index - 1       , field_i);
                        auto parent_right        = col_index(parent_index + 1       , field_i);
                        auto parent_bottom_left  = col_index(parent_bottom_index - 1, field_i);
                        auto parent_bottom_right = col_index(parent_bottom_index + 1, field_i);
                        auto parent_top_left     = col_index(parent_top_index - 1   , field_i);
                        auto parent_top_right    = col_index(parent_top_index + 1   , field_i);

                        MatSetValue(A, ghost_index, parent             ,                                  -1, ADD_VALUES);
                        MatSetValue(A, ghost_index, parent_bottom      ,                   -sign_j * pred[0], ADD_VALUES); //        sign_j
                * -1/8 MatSetValue(A, ghost_index, parent_top         ,                   -sign_j * pred[2], ADD_VALUES); //        sign_j *
                1/8 MatSetValue(A, ghost_index, parent_left        ,                   -sign_i * pred[0], ADD_VALUES); // sign_i        *
                -1/8 MatSetValue(A, ghost_index, parent_right       ,                   -sign_i * pred[2], ADD_VALUES); // sign_i        *
                1/8 MatSetValue(A, ghost_index, parent_bottom_left , sign_i * sign_j * pred[0] * pred[0], ADD_VALUES); // sign_i*sign_j *
                1/64 MatSetValue(A, ghost_index, parent_bottom_right, sign_i * sign_j * pred[2] * pred[0], ADD_VALUES); // sign_i*sign_j *
                -1/64 MatSetValue(A, ghost_index, parent_top_left    , sign_i * sign_j * pred[0] * pred[2], ADD_VALUES); // sign_i*sign_j *
                -1/64 MatSetValue(A, ghost_index, parent_top_right   , sign_i * sign_j * pred[2] * pred[2], ADD_VALUES); // sign_i*sign_j *
                1/64 m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                });*/

                using index_t = int;
                samurai::for_each_prediction_ghost(
                    m_mesh,
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                            MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            auto j       = ghost.indices(1);
                            auto jg      = j / 2;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_order + 1>(jsign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig, jg)),
                                                                field_i);
                            MatSetValue(A, ghost_index, parent_index, -1, ADD_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                                {
                                    if (ci != prediction_order || cj != prediction_order)
                                    {
                                        double value           = -interpx[ci] * interpy[cj];
                                        auto coarse_cell_index = this->col_index(
                                            static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1,
                                                                                         ig + static_cast<index_t>(ci - prediction_order),
                                                                                         jg + static_cast<index_t>(cj - prediction_order))),
                                            field_i);
                                        MatSetValue(A, ghost_index, coarse_cell_index, value, ADD_VALUES);
                                    }
                                }
                            }
                            this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

            void assemble_prediction_3D(Mat& A)
            {
                using index_t = int;
                samurai::for_each_prediction_ghost(
                    m_mesh,
                    [&](auto& ghost)
                    {
                        for (unsigned int field_i = 0; field_i < field_size; ++field_i)
                        {
                            PetscInt ghost_index = static_cast<PetscInt>(this->row_index(ghost, field_i));
                            MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);

                            auto ii      = ghost.indices(0);
                            auto ig      = ii / 2;
                            auto j       = ghost.indices(1);
                            auto jg      = j / 2;
                            auto k       = ghost.indices(2);
                            auto kg      = k / 2;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;
                            double ksign = (k & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_order + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_order + 1>(jsign);
                            auto interpz = samurai::interp_coeffs<2 * prediction_order + 1>(ksign);

                            auto parent_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(ghost.level - 1, ig, jg, kg)),
                                                                field_i);
                            MatSetValue(A, ghost_index, parent_index, -1, ADD_VALUES);

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                                {
                                    for (std::size_t ck = 0; ck < interpz.size(); ++ck)
                                    {
                                        if (ci != prediction_order || cj != prediction_order || ck != prediction_order)
                                        {
                                            double value           = -interpx[ci] * interpy[cj] * interpz[ck];
                                            auto coarse_cell_index = this->col_index(static_cast<PetscInt>(this->m_mesh.get_index(
                                                                                         ghost.level - 1,
                                                                                         ig + static_cast<index_t>(ci - prediction_order),
                                                                                         jg + static_cast<index_t>(cj - prediction_order),
                                                                                         kg + static_cast<index_t>(ck - prediction_order))),
                                                                                     field_i);
                                            MatSetValue(A, ghost_index, coarse_cell_index, value, ADD_VALUES);
                                        }
                                    }
                                }
                            }
                            this->m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                        }
                    });
            }

          public:

            template <class Func>
            static double L2Error(const Field& approximate, Func&& exact)
            {
                // In FV, we want only 1 quadrature point.
                // This is equivalent to
                //       error += pow(exact(cell.center()) - approximate(cell.index), 2) * cell.length;
                GaussLegendre<0> gl;

                double error_norm = 0;
                // double solution_norm = 0;
                for_each_cell(approximate.mesh(),
                              [&](const auto& cell)
                              {
                                  error_norm += gl.quadrature<1>(cell,
                                                                 [&](const auto& point)
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

                                  /*solution_norm += gl.quadrature<1>(cell, [&](const auto& point)
                                  {
                                      auto v = exact(point);
                                      double v_square;
                                      if constexpr (Field::size == 1)
                                      {
                                          v_square = v * v;
                                      }
                                      else
                                      {
                                          v_square = xt::sum(v * v)();
                                      }
                                      return v_square;
                                  });*/
                              });

                error_norm = sqrt(error_norm);
                // solution_norm = sqrt(solution_norm);
                // double relative_error = error_norm/solution_norm;
                // return relative_error;
                return error_norm;
            }
        };

        template <typename, typename = void>
        constexpr bool is_FluxBasedScheme{};

        template <typename T>
        constexpr bool is_FluxBasedScheme<T, std::void_t<decltype(std::declval<T>().scheme_coefficients())>> = true;

    } // end namespace petsc
} // end namespace samurai
