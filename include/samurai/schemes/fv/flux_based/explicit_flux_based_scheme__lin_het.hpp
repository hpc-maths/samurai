#pragma once
// #include "../../../petsc/fv/flux_based_scheme_assembly.hpp"
#include "../explicit_FV_scheme.hpp"
#include "flux_based_scheme__lin_het.hpp"

namespace samurai
{
    /**
     * LINEAR HETEROGENEOUS explicit schemes
     */
    template <class cfg, class bdry_cfg>
        requires(cfg::scheme_type == SchemeType::LinearHeterogeneous)
    class Explicit<FluxBasedScheme<cfg, bdry_cfg>> : public ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using size_type      = typename base_class::size_type;
        using base_class::scheme;

        static constexpr size_type n_comp         = input_field_t::n_comp;
        static constexpr size_type output_n_comp  = scheme_t::output_n_comp;
        static constexpr std::size_t stencil_size = cfg::stencil_size;

      public:

        using base_class::apply;

        explicit Explicit(scheme_t& s)
            : base_class(s)
        {
        }

        void apply(std::size_t d, output_field_t& output_field, input_field_t& input_field) override
        {
            assert(input_field.ghosts_updated());

            scheme().apply_directional_bc(input_field, d);

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
            scheme().for_each_interior_interface_and_coeffs(
                d,
                input_field,
                [&](const auto& interface_cells, const auto& comput_cells, auto& left_cell_coeffs, auto& right_cell_coeffs)
                {
                    for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
                    {
                        for (size_type field_j = 0; field_j < n_comp; ++field_j)
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
            if (scheme().include_boundary_fluxes())
            {
                scheme().for_each_boundary_interface_and_coeffs(
                    d,
                    input_field,
                    [&](const auto& cell, const auto& comput_cells, auto& coeffs)
                    {
                        for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
                        {
                            for (size_type field_j = 0; field_j < n_comp; ++field_j)
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
        }
    };

} // end namespace samurai
