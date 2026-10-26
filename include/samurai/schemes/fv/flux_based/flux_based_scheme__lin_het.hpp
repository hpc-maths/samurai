#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * Specialization of @ref FluxBasedScheme for LINEAR and HETEROGENEOUS schemes.
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHeterogeneous>>
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
         * The flux depends on the cells, so the interfaces are always visited cell by cell (Get::Cells).
         */
        template <Run run_type = Run::Sequential, Get = Get::Cells, bool include_periodic = true, class Func>
        void for_each_interior_interface_and_coeffs(std::size_t d, input_field_t& field, Func&& apply_coeffs) const
        {
            auto& flux_def = flux_definition()[d];

            this->template for_each_interior_interface_group<run_type, Get::Cells, include_periodic>(
                d,
                field.mesh(),
                [&](double h_face, double left_factor, double right_factor, auto&& for_each_interface)
                {
                    FluxStencilCoeffs<cfg> flux_coeffs;
                    FluxStencilCoeffs<cfg> left_cell_coeffs;
                    FluxStencilCoeffs<cfg> right_cell_coeffs;

                    for_each_interface(
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            StencilData<cfg> data(comput_cells);
                            data.cell_length = h_face;
                            flux_def.cons_flux_function(flux_coeffs, data);

                            left_cell_coeffs  = left_factor * flux_coeffs;
                            right_cell_coeffs = -right_factor * flux_coeffs;
                            apply_coeffs(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
                        });
                });
        }

        /**
         * Iterates for each boundary interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
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
                               auto h      = mesh.cell_length(level);
                               auto factor = h_factor(h, h);

                               // Boundary in direction
                               for_each_boundary_interface__direction(mesh,
                                                                      level,
                                                                      flux_def.direction,
                                                                      flux_def.stencil,
                                                                      [&](auto& cell, auto& comput_cells)
                                                                      {
                                                                          StencilData<cfg> data(comput_cells);
                                                                          data.cell_length = h;
                                                                          flux_def.cons_flux_function(flux_coeffs, data);
                                                                          flux_coeffs *= factor;
                                                                          apply_coeffs(cell, comput_cells, flux_coeffs);
                                                                      });

                               // Boundary in opposite direction
                               for_each_boundary_interface__opposite_direction(mesh,
                                                                               level,
                                                                               flux_def.direction,
                                                                               flux_def.stencil,
                                                                               [&](auto& cell, auto& comput_cells)
                                                                               {
                                                                                   StencilData<cfg> data(comput_cells);
                                                                                   data.cell_length = h;
                                                                                   flux_def.cons_flux_function(flux_coeffs, data);
                                                                                   flux_coeffs *= -factor;
                                                                                   apply_coeffs(cell, comput_cells, flux_coeffs);
                                                                               });
                           });
        }

        template <class Func>
        void for_each_boundary_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                for_each_boundary_interface_and_coeffs(d, field, std::forward<Func>(apply_coeffs));
            }
        }
    };

} // end namespace samurai
