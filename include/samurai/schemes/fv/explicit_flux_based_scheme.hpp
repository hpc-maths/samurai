#pragma once
// #include "../../petsc/fv/flux_based_scheme_assembly.hpp"
#include "../explicit_scheme.hpp"
#include "flux_based_scheme__lin_hom.hpp"
#include "flux_based_scheme__nonlin.hpp"

namespace samurai
{
    /**
     * LINEAR and HOMOGENEOUS explicit schemes
     */
    template <class Scheme>
    class Explicit<Scheme, std::enable_if_t<is_FluxBasedScheme_v<Scheme> && Scheme::cfg_t::flux_type == FluxType::LinearHomogeneous>>
    {
        using field_t                                  = typename Scheme::field_t;
        using scheme_definition_t                      = typename Scheme::scheme_definition_t;
        using flux_stencil_coeffs_t                    = typename scheme_definition_t::flux_stencil_coeffs_t;
        static constexpr std::size_t dim               = field_t::dim;
        static constexpr std::size_t field_size        = field_t::size;
        static constexpr std::size_t output_field_size = Scheme::output_field_size;
        static constexpr std::size_t stencil_size      = Scheme::stencil_size;

      protected:

        const Scheme* m_scheme = nullptr;

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

            // Interior interfaces
            scheme().for_each_interior_interface(
                f.mesh(),
                [&](auto& interface_cells, auto& comput_cells, auto& left_cell_coeffs, auto& right_cell_coeffs)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                        {
                            for (std::size_t c = 0; c < stencil_size; ++c)
                            {
                                double left_cell_coeff  = scheme().cell_coeff(left_cell_coeffs, c, field_i, field_j);
                                double right_cell_coeff = scheme().cell_coeff(right_cell_coeffs, c, field_i, field_j);
                                field_value(result, interface_cells[0], field_i) += left_cell_coeff
                                                                                  * field_value(f, comput_cells[c], field_j);
                                field_value(result, interface_cells[1], field_i) += right_cell_coeff
                                                                                  * field_value(f, comput_cells[c], field_j);
                            }
                        }
                    }
                });

            // Boundary interfaces
            scheme().for_each_boundary_interface(
                f.mesh(),
                [&](auto& cell, auto& comput_cells, auto& coeffs)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                        {
                            for (std::size_t c = 0; c < stencil_size; ++c)
                            {
                                double coeff = scheme().cell_coeff(coeffs, c, field_i, field_j);
                                field_value(result, cell, field_i) += coeff * field_value(f, comput_cells[c], field_j);
                            }
                        }
                    }
                });

            return result;
        }
    };

    /**
     * NON-LINEAR explicit schemes
     */
    template <class Scheme>
    class Explicit<Scheme, std::enable_if_t<is_FluxBasedScheme_v<Scheme> && Scheme::cfg_t::flux_type == FluxType::NonLinear>>
    {
        using field_t                                  = typename Scheme::field_t;
        using scheme_definition_t                      = typename Scheme::scheme_definition_t;
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

            // Interior interfaces
            scheme().for_each_interior_interface(
                f,
                [&](auto& interface_cells, auto& left_cell_contrib, auto& right_cell_contrib)
                {
                    for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                    {
                        field_value(result, interface_cells[0], field_i) += scheme().cell_coeff(left_cell_contrib, field_i);
                        field_value(result, interface_cells[1], field_i) += scheme().cell_coeff(right_cell_contrib, field_i);
                    }
                });

            // Boundary interfaces
            scheme().for_each_boundary_interface(f,
                                                 [&](auto& cell, auto& contrib)
                                                 {
                                                     for (std::size_t field_i = 0; field_i < output_field_size; ++field_i)
                                                     {
                                                         field_value(result, cell, field_i) += scheme().cell_coeff(contrib, field_i);
                                                     }
                                                 });

            return result;
        }
    };
} // end namespace samurai
