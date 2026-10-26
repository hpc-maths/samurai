#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * Specialization of @ref FluxBasedScheme for LINEAR and HOMOGENEOUS schemes.
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : public FluxBasedSchemeBase<cfg, bdry_cfg>
    {
      public:

        using base_class = FluxBasedSchemeBase<cfg, bdry_cfg>;

        using base_class::dim;
        using typename base_class::input_field_t;

        using base_class::base_class;
        using base_class::flux_definition;
        using base_class::for_each_interior_interface_and_coeffs;

      private:

        using base_class::h_factor;

      public:

        /**
         * Iterates for each interior interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Func>
        void for_each_interior_interface_and_coeffs(std::size_t d, input_field_t& field, Func&& apply_coeffs) const
        {
            auto& flux_def = flux_definition()[d];

            // The flux only depends on the face length: it is computed once per group of interfaces.
            this->template for_each_interior_interface_group<run_type, get_type, include_periodic>(
                d,
                field.mesh(),
                [&](double h_face, double left_factor, double right_factor, auto&& for_each_interface)
                {
                    FluxStencilCoeffs<cfg> flux_coeffs;
                    FluxStencilCoeffs<cfg> left_cell_coeffs;
                    FluxStencilCoeffs<cfg> right_cell_coeffs;

                    flux_def.cons_flux_function(flux_coeffs, h_face);

                    left_cell_coeffs  = left_factor * flux_coeffs;
                    right_cell_coeffs = -right_factor * flux_coeffs;

                    for_each_interface(
                        [&](auto& interface, auto& stencil)
                        {
                            apply_coeffs(interface, stencil, left_cell_coeffs, right_cell_coeffs);
                        });
                });
        }

        /**
         * Iterates for each boundary interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_boundary_interface_and_coeffs(std::size_t d, input_field_t& field, Func&& apply_coeffs) const
        {
            auto& mesh = field.mesh();

            if (mesh.periodicity()[d])
            {
                return; // no boundary in this direction
            }

            auto& flux_def = flux_definition()[d];

            FluxStencilCoeffs<cfg> flux_coeffs;

            for_each_level(mesh,
                           [&](auto level)
                           {
                               auto h = mesh.cell_length(level);

                               auto factor = h_factor(h, h);

                               // Boundary in direction
                               flux_def.cons_flux_function(flux_coeffs, h);
                               flux_coeffs *= factor;
                               for_each_boundary_interface__direction<run_type, get_type>(mesh,
                                                                                          level,
                                                                                          flux_def.direction,
                                                                                          flux_def.stencil,
                                                                                          [&](auto& cell, auto& comput_cells)
                                                                                          {
                                                                                              apply_coeffs(cell, comput_cells, flux_coeffs);
                                                                                          });

                               // Boundary in opposite direction
                               flux_coeffs *= -1;
                               for_each_boundary_interface__opposite_direction<run_type, get_type>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& cell, auto& comput_cells)
                                   {
                                       apply_coeffs(cell, comput_cells, flux_coeffs);
                                   });
                           });
        }

        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_boundary_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                for_each_boundary_interface_and_coeffs<run_type, get_type>(d, field, std::forward<Func>(apply_coeffs));
            }
        }
    };

} // end namespace samurai
