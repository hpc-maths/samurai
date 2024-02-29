#pragma once
// #include "../../../petsc/fv/flux_based_scheme_assembly.hpp"
#include "../explicit_FV_scheme.hpp"
#include "flux_based_scheme__lin_hom.hpp"
#include "flux_based_scheme__nonlin.hpp"

namespace samurai
{
    /**
     * LINEAR explicit schemes
     */
    template <class cfg, class bdry_cfg>
    class Explicit<FluxBasedScheme<cfg, bdry_cfg>,
                   std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous || cfg::scheme_type == SchemeType::LinearHeterogeneous>>
        : public ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>;

        using scheme_t              = typename base_class::scheme_t;
        using input_field_t         = typename base_class::input_field_t;
        using output_field_t        = typename base_class::output_field_t;
        using flux_stencil_coeffs_t = typename scheme_t::flux_stencil_coeffs_t;
        using base_class::scheme;

        static constexpr std::size_t field_size        = input_field_t::size;
        static constexpr std::size_t output_field_size = scheme_t::output_field_size;
        static constexpr std::size_t stencil_size      = cfg::stencil_size;

      public:

        explicit Explicit(const scheme_t& s)
            : base_class(s)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const override
        {
            /**
             * Implementation by matrix-vector multiplication
             */
            // Mat A;
            // auto assembly = petsc::make_assembly(scheme());
            // assembly.create_matrix(A);
            // assembly.assemble_matrix(A);
            // Vec vec_f   = petsc::create_petsc_vector_from(f);
            // Vec vec_res = petsc::create_petsc_vector_from(output_field);
            // MatMult(A, vec_f, vec_res);

            // Interior interfaces
            scheme().for_each_interior_interface(
                input_field.mesh(),
                [&](const auto& interface_cells, const auto& comput_cells, auto& left_cell_coeffs, auto& right_cell_coeffs)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                        {
                            for (std::size_t c = 0; c < stencil_size; ++c)
                            {
#ifdef SAMURAI_CHECK_NAN
                                if (std::isnan(field_value(input_field, comput_cells[c], field_j)))
                                {
                                    std::cerr << "NaN detected when computing the flux on the interior interfaces: " << comput_cells[c]
                                              << std::endl;
                                    assert(false);
                                }
#endif
                                double left_cell_coeff  = this->scheme().cell_coeff(left_cell_coeffs, c, field_i, field_j);
                                double right_cell_coeff = this->scheme().cell_coeff(right_cell_coeffs, c, field_i, field_j);
                                field_value(output_field, interface_cells[0], field_i) += left_cell_coeff
                                                                                        * field_value(input_field, comput_cells[c], field_j);
                                field_value(output_field, interface_cells[1], field_i) += right_cell_coeff
                                                                                        * field_value(input_field, comput_cells[c], field_j);
                            }
                        }
                    }
                });

            // Boundary interfaces
            scheme().for_each_boundary_interface(
                input_field.mesh(),
                [&](const auto& cell, const auto& comput_cells, auto& coeffs)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                        {
                            for (std::size_t c = 0; c < stencil_size; ++c)
                            {
#ifdef SAMURAI_CHECK_NAN
                                if (std::isnan(field_value(input_field, comput_cells[c], field_j)))
                                {
                                    std::cerr << "NaN detected when computing the flux on the boundary interfaces: " << comput_cells[c]
                                              << std::endl;
                                    assert(false);
                                }
#endif
                                double coeff = this->scheme().cell_coeff(coeffs, c, field_i, field_j);
                                field_value(output_field, cell, field_i) += coeff * field_value(input_field, comput_cells[c], field_j);
                            }
                        }
                    }
                });
        }
    };

    /**
     * NON-LINEAR explicit schemes
     */
    template <class cfg, class bdry_cfg>
    class Explicit<FluxBasedScheme<cfg, bdry_cfg>, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using base_class::scheme;

        static constexpr std::size_t output_field_size = scheme_t::output_field_size;

      public:

        explicit Explicit(const scheme_t& s)
            : base_class(s)
        {
        }

        void apply(output_field_t& output_field, input_field_t& input_field) const override
        {
            // Interior interfaces
            scheme().for_each_interior_interface(
                input_field,
                [&](const auto& interface_cells, auto& left_cell_contrib, auto& right_cell_contrib)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        field_value(output_field, interface_cells[0], field_i) += this->scheme().flux_value_cmpnent(left_cell_contrib,
                                                                                                                    field_i);
                        field_value(output_field, interface_cells[1], field_i) += this->scheme().flux_value_cmpnent(right_cell_contrib,
                                                                                                                    field_i);
                    }
                });

            // Boundary interfaces
            scheme().for_each_boundary_interface(
                input_field,
                [&](const auto& cell, auto& contrib)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        field_value(output_field, cell, field_i) += this->scheme().flux_value_cmpnent(contrib, field_i);
                    }
                });
        }
    };
} // end namespace samurai
