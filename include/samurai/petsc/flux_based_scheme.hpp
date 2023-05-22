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
        template <PetscInt output_field_size_, PetscInt stencil_size_>
        struct FluxBasedAssemblyConfig
        {
            static constexpr PetscInt output_field_size = output_field_size_;
            static constexpr PetscInt stencil_size      = stencil_size_;
        };

        template <class cfg, class bdry_cfg, class Field>
        class FluxBasedScheme : public FVScheme<Field, cfg::output_field_size, bdry_cfg>
        {
            template <class Scheme1, class Scheme2>
            friend class FluxBasedScheme_Sum_CellBasedScheme;

            template <class Scheme>
            friend class Scalar_x_FluxBasedScheme;

            template <std::size_t rows, std::size_t cols, class... Operators>
            friend class MonolithicBlockAssembly;

          protected:

            using base_class = FVScheme<Field, cfg::output_field_size, bdry_cfg>;
            using base_class::cell_coeff;
            using base_class::col_index;
            using base_class::dim;
            using base_class::field_size;
            using base_class::m_mesh;
            using base_class::m_unknown;
            using base_class::row_index;
            using base_class::set_current_insert_mode;
            using base_class::set_is_row_not_empty;
            using dirichlet_t = typename base_class::dirichlet_t;
            using neumann_t   = typename base_class::neumann_t;

          public:

            using cfg_t                                    = cfg;
            using bdry_cfg_t                               = bdry_cfg;
            using field_t                                  = Field;
            static constexpr std::size_t output_field_size = cfg::output_field_size;
            static constexpr std::size_t stencil_size      = cfg::stencil_size;

            using coefficients_t = FluxBasedCoefficients<Field, output_field_size, stencil_size>;

          protected:

            std::array<coefficients_t, dim> m_scheme_coefficients;

          public:

            FluxBasedScheme(Field& unknown, std::array<coefficients_t, dim> scheme_coefficients)
                : base_class(unknown)
                , m_scheme_coefficients(scheme_coefficients)
            {
                set_current_insert_mode(ADD_VALUES);
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
                    for_each_interior_interface(
                        m_mesh,
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        [&](auto& interface_cells, auto&)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size * field_size;
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[1], field_i))] += stencil_size * field_size;
                                }
                            }
                        });

                    for_each_boundary_interface(
                        m_mesh,
                        scheme_coeffs_dir.flux.direction,
                        scheme_coeffs_dir.flux.stencil,
                        [&](auto& interface_cells, auto&)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size * field_size;
                                }
                            }
                        });

                    auto opposite_direction = xt::eval(-scheme_coeffs_dir.flux.direction);
                    auto opposite_stencil   = xt::eval(-scheme_coeffs_dir.flux.stencil);
                    for_each_boundary_interface(
                        m_mesh,
                        opposite_direction,
                        opposite_stencil,
                        [&](auto& interface_cells, auto&)
                        {
                            for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                            {
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    nnz[static_cast<std::size_t>(this->row_index(interface_cells[0], field_i))] += stencil_size * field_size;
                                }
                            }
                        });
                }
            }

          protected:

            //-------------------------------------------------------------//
            //             Assemble scheme in the interior                 //
            //-------------------------------------------------------------//

            void assemble_scheme(Mat& A) override
            {
                // std::cout << "assemble_scheme() of " << this->name() << std::endl;

                if (this->current_insert_mode() == INSERT_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(ADD_VALUES);
                }

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
                                auto interface_cell1_row = this->row_index(interface_cells[0], field_i);
                                auto interface_cell2_row = this->row_index(interface_cells[1], field_i);
                                for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                {
                                    for (std::size_t c = 0; c < stencil_size; ++c)
                                    {
                                        auto comput_cell_col = col_index(comput_cells[c], field_j);
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
                                set_is_row_not_empty(interface_cell1_row);
                                set_is_row_not_empty(interface_cell2_row);
                            }
                        });

                    for_each_boundary_interface(m_mesh,
                                                scheme_coeffs_dir.flux.direction,
                                                scheme_coeffs_dir.flux.stencil,
                                                scheme_coeffs_dir.flux.get_flux_coeffs,
                                                scheme_coeffs_dir.get_cell1_coeffs,
                                                [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        auto interface_cell0_row = this->row_index(interface_cells[0], field_i);
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            for (std::size_t c = 0; c < stencil_size; ++c)
                                                            {
                                                                double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                                                if (coeff != 0)
                                                                {
                                                                    auto comput_cell_col = col_index(comput_cells[c], field_j);
                                                                    MatSetValue(A, interface_cell0_row, comput_cell_col, coeff, ADD_VALUES);
                                                                }
                                                            }
                                                        }
                                                        set_is_row_not_empty(interface_cell0_row);
                                                    }
                                                });

                    auto opposite_direction             = xt::eval(-scheme_coeffs_dir.flux.direction);
                    Stencil<stencil_size, dim> reversed = xt::eval(xt::flip(scheme_coeffs_dir.flux.stencil, 0));
                    auto opposite_stencil               = xt::eval(-reversed);
                    for_each_boundary_interface(m_mesh,
                                                opposite_direction,
                                                opposite_stencil,
                                                scheme_coeffs_dir.flux.get_flux_coeffs,
                                                scheme_coeffs_dir.get_cell2_coeffs,
                                                [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                                                {
                                                    for (unsigned int field_i = 0; field_i < output_field_size; ++field_i)
                                                    {
                                                        auto interface_cell0_row = this->row_index(interface_cells[0], field_i);
                                                        for (unsigned int field_j = 0; field_j < field_size; ++field_j)
                                                        {
                                                            for (std::size_t c = 0; c < stencil_size; ++c)
                                                            {
                                                                double coeff = cell_coeff(coeffs, c, field_i, field_j);
                                                                if (coeff != 0)
                                                                {
                                                                    auto comput_cell_col = col_index(comput_cells[c], field_j);
                                                                    MatSetValue(A, interface_cell0_row, comput_cell_col, coeff, ADD_VALUES);
                                                                }
                                                            }
                                                        }
                                                        set_is_row_not_empty(interface_cell0_row);
                                                    }
                                                });
                }
            }
        };

        template <typename, typename = void>
        constexpr bool is_FluxBasedScheme{};

        template <typename T>
        constexpr bool is_FluxBasedScheme<T, std::void_t<decltype(std::declval<T>().scheme_coefficients())>> = true;

    } // end namespace petsc
} // end namespace samurai
