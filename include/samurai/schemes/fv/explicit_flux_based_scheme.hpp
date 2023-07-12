#pragma once
// #include "../../petsc/fv/flux_based_scheme_assembly.hpp"
#include "../explicit_scheme.hpp"
#include "flux_based_scheme.hpp"

namespace samurai
{
    template <class Scheme>
    class Explicit<Scheme, std::enable_if_t<is_FluxBasedScheme_v<Scheme>>>
    {
        using field_t                                  = typename Scheme::field_t;
        static constexpr std::size_t dim               = field_t::dim;
        static constexpr std::size_t field_size        = field_t::size;
        static constexpr std::size_t output_field_size = Scheme::output_field_size;
        static constexpr std::size_t stencil_size      = Scheme::stencil_size;

      protected:

        const Scheme* m_scheme;

      public:

        explicit Explicit(const Scheme& scheme)
            : m_scheme(&scheme)
        {
        }

        auto& scheme() const
        {
            return *m_scheme;
        }

        auto apply_to(field_t& f)
        {
            auto result = make_field<typename field_t::value_type, Scheme::output_field_size, field_t::is_soa>(
                scheme().name() + "(" + f.name() + ")",
                f.mesh());
            result.fill(0);

            update_bc(f);

            /**
             * Implementation by matrix-vector multiplication
             */
            // Mat A;
            // auto assembly = petsc::make_assembly(scheme());
            // assembly.create_matrix(A);
            // assembly.assemble_matrix(A);
            // Vec vec_f   = petsc::create_petsc_vector_from(f);
            // Vec vec_res = petsc::create_petsc_vector_from(result);
            // MatMult(A, vec_f, vec_res);
            // return result;

            auto& mesh = f.mesh();
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto scheme_coeffs_dir = scheme().coefficients()[d];
                for_each_interior_interface(
                    mesh,
                    scheme_coeffs_dir.flux.direction,
                    scheme_coeffs_dir.flux.stencil,
                    scheme_coeffs_dir.flux.get_flux_coeffs,
                    scheme_coeffs_dir.get_cell1_coeffs,
                    scheme_coeffs_dir.get_cell2_coeffs,
                    [&](auto& interface_cells, auto& comput_cells, auto& cell1_coeffs, auto& cell2_coeffs)
                    {
                        for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                            {
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
                                    double cell1_coeff = scheme().cell_coeff(cell1_coeffs, c, field_i, field_j);
                                    double cell2_coeff = scheme().cell_coeff(cell2_coeffs, c, field_i, field_j);
                                    field_value(result, interface_cells[0], field_i) += cell1_coeff
                                                                                      * field_value(f, comput_cells[c], field_j);
                                    field_value(result, interface_cells[1], field_i) += cell2_coeff
                                                                                      * field_value(f, comput_cells[c], field_j);
                                }
                            }
                        }
                    });

                for_each_boundary_interface(
                    mesh,
                    scheme_coeffs_dir.flux.direction,
                    scheme_coeffs_dir.flux.stencil,
                    scheme_coeffs_dir.flux.get_flux_coeffs,
                    scheme_coeffs_dir.get_cell1_coeffs,
                    [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                    {
                        for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                            {
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
                                    double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                    field_value(result, interface_cells[0], field_i) += coeff * field_value(f, comput_cells[c], field_j);
                                }
                            }
                        }
                    });

                auto opposite_direction             = xt::eval(-scheme_coeffs_dir.flux.direction);
                Stencil<stencil_size, dim> reversed = xt::eval(xt::flip(scheme_coeffs_dir.flux.stencil, 0));
                auto opposite_stencil               = xt::eval(-reversed);
                for_each_boundary_interface(
                    mesh,
                    opposite_direction,
                    opposite_stencil,
                    scheme_coeffs_dir.flux.get_flux_coeffs,
                    scheme_coeffs_dir.get_cell2_coeffs,
                    [&](auto& interface_cells, auto& comput_cells, auto& coeffs)
                    {
                        for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                        {
                            for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                            {
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
                                    double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                    field_value(result, interface_cells[0], field_i) += coeff * field_value(f, comput_cells[c], field_j);
                                }
                            }
                        }
                    });
            }

            return result;
        }
    };
} // end namespace samurai
