#pragma once
#include "../interface.hpp"
#include "fv/FV_scheme.hpp"

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
        template <PetscInt output_field_size_, PetscInt neighbourhood_width_, PetscInt comput_stencil_size_, DirichletEnforcement dirichlet_enfcmt_ = Equation>
        struct FluxBasedAssemblyConfig
        {
            static constexpr PetscInt output_field_size            = output_field_size_;
            static constexpr PetscInt neighbourhood_width          = neighbourhood_width_;
            static constexpr PetscInt comput_stencil_size          = comput_stencil_size_;
            static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
        };

        template <class cfg, class Field>
        class FluxBasedScheme : public FVScheme<Field, cfg::output_field_size, cfg::neighbourhood_width>
        {
            template <class Scheme1, class Scheme2>
            friend class FluxBasedScheme_Sum_CellBasedScheme;

            using base_class = FVScheme<Field, cfg::output_field_size, cfg::neighbourhood_width>;
            using base_class::cell_coeff;
            using base_class::col_index;
            using base_class::m_is_row_empty;
            using base_class::m_mesh;
            using base_class::row_index;
            using base_class::set_current_insert_mode;

            using base_class::m_unknown;
            using dirichlet_t = typename base_class::dirichlet_t;
            using neumann_t   = typename base_class::neumann_t;

          public:

            using cfg_t   = cfg;
            using field_t = Field;

            using field_value_type                           = typename Field::value_type; // double
            static constexpr std::size_t dim                 = Field::dim;
            static constexpr std::size_t field_size          = Field::size;
            static constexpr std::size_t output_field_size   = cfg::output_field_size;
            static constexpr std::size_t comput_stencil_size = cfg::comput_stencil_size;

            using coefficients_t = FluxBasedCoefficients<Field, output_field_size, comput_stencil_size>;

          protected:

            std::array<coefficients_t, dim> m_scheme_coefficients;

          public:

            FluxBasedScheme(Field& unknown, std::array<coefficients_t, dim> scheme_coefficients)
                : base_class(unknown)
                , m_scheme_coefficients(scheme_coefficients)
            {
            }

            auto& scheme_coefficients() const
            {
                return m_scheme_coefficients;
            }

            auto& scheme_coefficients()
            {
                return m_scheme_coefficients;
            }

          public:

            //-------------------------------------------------------------//
            //                     Sparsity pattern                        //
            //-------------------------------------------------------------//

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

            /*void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                // Iterate over the boundary conditions set by the user
                for (auto& bc : m_unknown.get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out    = directions[d];
                        auto& boundary_cells = boundary_cells_directions[d];

                        // Find the direction 'towards_out' amongst the flux directions
                        Stencil<comput_stencil_size, dim> stencil = find_stencil_from_scheme(towards_out);

                        dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                        neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                        if (dirichlet)
                        {
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         stencil,
                                                         [&](auto& cells)
                                                         {
                                                             sparsity_pattern_dirichlet_bc(nnz, cells);
                                                         });
                        }
                        else if (neumann)
                        {
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         stencil,
                                                         [&](auto& cells)
                                                         {
                                                             sparsity_pattern_neumann_bc(nnz, cells);
                                                         });
                        }
                        else
                        {
                            std::cerr << "Unknown boundary condition type" << std::endl;
                        }
                    }
                }
            }

          protected:

            template <class CellList>
            void sparsity_pattern_dirichlet_bc(std::vector<PetscInt>& nnz, CellList& cells) const
            {
                auto& cell = cells[0];
                for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                {
                    nnz[row_index(cell, field_i)]++;
                }
                for (std::size_t g = 1; g < comput_stencil_size; ++g)
                {
                    auto& ghost = cells[g];
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
            }

            template <class CellList>
            void sparsity_pattern_neumann_bc(std::vector<PetscInt>& nnz, CellList& cells) const
            {
                auto& cell = cells[0];
                for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                {
                    nnz[row_index(cell, field_i)]++;
                }
                for (std::size_t g = 1; g < comput_stencil_size; ++g)
                {
                    auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        nnz[row_index(ghost, field_i)] = 2;
                    }
                }
            }*/

          protected:

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

            void assemble_scheme(Mat& A) override
            {
                set_current_insert_mode(ADD_VALUES);

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
                    for_each_boundary_interface(
                        m_mesh,
                        opposite_direction,
                        opposite_stencil,
                        scheme_coeffs_dir.flux.get_flux_coeffs,
                        scheme_coeffs_dir.get_cell2_coeffs,
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

            //-------------------------------------------------------------//
            //             Assemble the boundary conditions                //
            //-------------------------------------------------------------//

            /*auto find_stencil_from_scheme(const DirectionVector<dim>& direction) const
            {
                bool found = false;
                Stencil<comput_stencil_size, dim> stencil;
                for (std::size_t d2 = 0; d2 < dim; ++d2)
                {
                    if (direction == m_scheme_coefficients[d2].flux.direction)
                    {
                        found   = true;
                        stencil = m_scheme_coefficients[d2].flux.stencil;
                    }
                }
                if (!found)
                {
                    for (std::size_t d2 = 0; d2 < dim; ++d2)
                    {
                        if (direction == -m_scheme_coefficients[d2].flux.direction)
                        {
                            found   = true;
                            stencil = -m_scheme_coefficients[d2].flux.stencil;
                        }
                    }
                }
                assert(found);
                return stencil;
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                auto get_coeffs = [&](double h)
                {
                    auto flux_coeffs = m_scheme_coefficients[0].flux.get_flux_coeffs(h);
                    return m_scheme_coefficients[0].get_cell1_coeffs(flux_coeffs, h, h);
                };

                // Iterate over the boundary conditions set by the user
                for (auto& bc : m_unknown.get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out    = directions[d];
                        auto& boundary_cells = boundary_cells_directions[d];

                        // Find the direction 'towards_out' amongst the flux directions
                        Stencil<comput_stencil_size, dim> stencil = find_stencil_from_scheme(towards_out);

                        dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                        neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                        if (dirichlet)
                        {
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         stencil,
                                                         get_coeffs,
                                                         [&](auto& cells, auto& coeffs)
                                                         {
                                                             assemble_dirichlet_bc(A, cells, coeffs);
                                                         });
                        }
                        else if (neumann)
                        {
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         stencil,
                                                         get_coeffs,
                                                         [&](auto& cells, auto& coeffs)
                                                         {
                                                             assemble_neumann_bc(A, cells, coeffs);
                                                         });
                        }
                        else
                        {
                            std::cerr << "Unknown boundary condition type" << std::endl;
                        }
                    }
                }
            }

            template <class CellList, class CoeffList>
            void assemble_dirichlet_bc(Mat& A, CellList& cells, CoeffList& coeffs)
            {
                const auto& cell = cells[0];
                for (std::size_t g = 1; g < comput_stencil_size; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);

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

                            // the coeff of the ghost is removed from the stencil (we want 0 so we substract the coeff we set before)
                            MatSetValue(A, cell_index, ghost_index, -coeff, ADD_VALUES);
                            // the coeff is substracted from the center of the stencil
                            MatSetValue(A, cell_index, cell_index, -coeff, ADD_VALUES);
                            // 1 is added to the diagonal of the ghost
                            MatSetValue(A, ghost_index, ghost_index, 1, ADD_VALUES);
                        }
                        else
                        {
                            coeff = coeff == 0 ? 1 : coeff;
                            // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
                            //                        [  1/2    1/2 ] = dirichlet_value
                            // which is equivalent to
                            //                        [-coeff -coeff] = -2 * coeff * dirichlet_value
                            MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES);
                            MatSetValue(A, ghost_index, cell_index, -coeff, ADD_VALUES);
                        }
                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                }
            }

            template <class CellList, class CoeffList>
            void assemble_neumann_bc(Mat& A, CellList& cells, CoeffList& coeffs)
            {
                const auto& cell = cells[0];
                for (std::size_t g = 1; g < comput_stencil_size; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);
                        coeff        = coeff == 0 ? 1 : coeff;
                        // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
                        //                    [ 1/h  -1/h ] = neumann_value
                        // However, to have symmetry, we want to have coeff as the off-diagonal coefficient, so
                        //                   [-coeff coeff] = -coeff * h * neumann_value
                        MatSetValue(A, ghost_index, ghost_index, -coeff, ADD_VALUES);
                        MatSetValue(A, ghost_index, cell_index, coeff, ADD_VALUES);

                        m_is_row_empty[static_cast<std::size_t>(ghost_index)] = false;
                    }
                }
            }

          public:

            //-------------------------------------------------------------//
            //   Enforce the boundary conditions on the right-hand side    //
            //-------------------------------------------------------------//

            void enforce_bc(Vec& b) const override
            {
                auto get_coeffs = [&](double h)
                {
                    auto flux_coeffs = m_scheme_coefficients[0].flux.get_flux_coeffs(h);
                    return m_scheme_coefficients[0].get_cell1_coeffs(flux_coeffs, h, h);
                };

                // Iterate over the boundary conditions set by the user
                for (auto& bc : m_unknown.get_bc())
                {
                    auto bc_region                  = bc->get_region(); // get the region
                    auto& directions                = bc_region.first;
                    auto& boundary_cells_directions = bc_region.second;
                    // Iterate over the directions in that region
                    for (std::size_t d = 0; d < directions.size(); ++d)
                    {
                        auto& towards_out    = directions[d];
                        auto& boundary_cells = boundary_cells_directions[d];

                        // Find the direction 'towards_out' amongst the flux directions
                        Stencil<comput_stencil_size, dim> stencil = find_stencil_from_scheme(towards_out);

                        dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                        neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                        if (dirichlet)
                        {
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         stencil,
                                                         get_coeffs,
                                                         [&](auto& cells, auto& coeffs)
                                                         {
                                                             enforce_dirichlet_bc(b, cells, coeffs, *dirichlet, towards_out);
                                                         });
                        }
                        else if (neumann)
                        {
                            for_each_stencil_on_boundary(m_mesh,
                                                         boundary_cells,
                                                         stencil,
                                                         get_coeffs,
                                                         [&](auto& cells, auto& coeffs)
                                                         {
                                                             enforce_neumann_bc(b, cells, coeffs, *neumann, towards_out);
                                                         });
                        }
                        else
                        {
                            std::cerr << "Unknown boundary condition type" << std::endl;
                        }
                    }
                }
            }

            template <class CellList, class CoeffList>
            void enforce_dirichlet_bc(Vec& b,
                                      CellList& cells,
                                      CoeffList& coeffs,
                                      const dirichlet_t& dirichlet,
                                      const DirectionVector<dim>& towards_out) const
            {
                auto& cell = cells[0];

                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t g = 1; g < comput_stencil_size; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);

                        double dirichlet_value;
                        if constexpr (field_size == 1)
                        {
                            dirichlet_value = dirichlet.value(boundary_point);
                        }
                        else
                        {
                            dirichlet_value = dirichlet.value(boundary_point)(field_i); // TODO: call get_value() only once instead of once
                                                                                        // per field_i
                        }

                        if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                        {
                            VecSetValue(b, cell_index, -2 * coeff * dirichlet_value, ADD_VALUES);
                        }
                        else
                        {
                            coeff = coeff == 0 ? 1 : coeff;
                            VecSetValue(b, ghost_index, -2 * coeff * dirichlet_value, INSERT_VALUES);
                        }
                    }
                }
            }

            template <class CellList, class CoeffList>
            void
            enforce_neumann_bc(Vec& b, CellList& cells, CoeffList& coeffs, const neumann_t& neumann, const DirectionVector<dim>&
          towards_out) const
            {
                auto& cell          = cells[0];
                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t g = 1; g < comput_stencil_size; ++g)
                {
                    const auto& ghost = cells[g];
                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        // PetscInt cell_index  = static_cast<PetscInt>(col_index(cell, field_i));
                        PetscInt ghost_index = static_cast<PetscInt>(col_index(ghost, field_i));

                        double coeff = cell_coeff(coeffs, g, field_i, field_i);

                        coeff   = coeff == 0 ? 1 : coeff;
                        auto& h = cell.length;
                        double neumann_value;
                        if constexpr (field_size == 1)
                        {
                            neumann_value = neumann.value(boundary_point);
                        }
                        else
                        {
                            neumann_value = neumann.value(boundary_point)(field_i); // TODO: call get_value() only once instead of once per
                                                                                    // field_i
                        }
                        VecSetValue(b, ghost_index, -coeff * h * neumann_value, INSERT_VALUES);
                    }
                }
            }*/
        };

        template <typename, typename = void>
        constexpr bool is_FluxBasedScheme{};

        template <typename T>
        constexpr bool is_FluxBasedScheme<T, std::void_t<decltype(std::declval<T>().scheme_coefficients())>> = true;

    } // end namespace petsc
} // end namespace samurai
