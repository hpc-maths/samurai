#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * @class FluxBasedScheme
     *    Implementation of LINEAR and HETEROGENEOUS schemes
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHeterogeneous>>
        : public FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
      public:

        using base_class = FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

      private:

        FluxDefinition<cfg> m_flux_definition;
        bool m_include_boundary_fluxes = true;

      public:

        explicit FluxBasedScheme(const FluxDefinition<cfg>& flux_definition)
            : m_flux_definition(flux_definition)
        {
        }

        auto& flux_definition() const
        {
            return m_flux_definition;
        }

        auto& flux_definition()
        {
            return m_flux_definition;
        }

        void include_boundary_fluxes(bool include)
        {
            m_include_boundary_fluxes = include;
        }

        bool include_boundary_fluxes() const
        {
            return m_include_boundary_fluxes;
        }

        FluxStencilCoeffs<cfg> contribution(const FluxStencilCoeffs<cfg>& flux_coeffs, double h_face, double h_cell) const
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            return (face_measure / cell_measure) * flux_coeffs;
        }

        /**
         * Iterates for each interior interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
        void for_each_interior_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h = cell_length(level);

                    for_each_interior_interface__same_level(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            auto flux_coeffs                               = flux_def.cons_flux_function(comput_cells);
                            auto left_cell_contrib                         = contribution(flux_coeffs, h, h);
                            decltype(left_cell_contrib) right_cell_contrib = -left_cell_contrib;
                            apply_coeffs(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                        });
                }

                // Level jumps (level -- level+1)
                for (std::size_t level = min_level; level < max_level; ++level)
                {
                    auto h_l   = cell_length(level);
                    auto h_lp1 = cell_length(level + 1);

                    //         |__|   l+1
                    //    |____|      l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface__level_jump_direction(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_coeffs                        = flux_def.cons_flux_function(comput_cells);
                                decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                                auto left_cell_contrib                  = contribution(flux_coeffs, h_lp1, h_l);
                                auto right_cell_contrib                 = contribution(minus_flux_coeffs, h_lp1, h_lp1);
                                apply_coeffs(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                    //    |__|        l+1
                    //       |____|   l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface__level_jump_opposite_direction(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_coeffs                        = flux_def.cons_flux_function(comput_cells);
                                decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                                auto left_cell_contrib                  = contribution(flux_coeffs, h_lp1, h_lp1);
                                auto right_cell_contrib                 = contribution(minus_flux_coeffs, h_lp1, h_l);
                                apply_coeffs(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                }
            }
        }

        /**
         * Iterates for each boundary interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
        void for_each_boundary_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            auto& mesh = field.mesh();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                for_each_level(mesh,
                               [&](auto level)
                               {
                                   auto h = cell_length(level);

                                   // Boundary in direction
                                   for_each_boundary_interface__direction(mesh,
                                                                          level,
                                                                          flux_def.direction,
                                                                          flux_def.stencil,
                                                                          [&](auto& cell, auto& comput_cells)
                                                                          {
                                                                              auto flux_coeffs  = flux_def.cons_flux_function(comput_cells);
                                                                              auto cell_contrib = contribution(flux_coeffs, h, h);
                                                                              apply_coeffs(cell, comput_cells, cell_contrib);
                                                                          });

                                   // Boundary in opposite direction
                                   for_each_boundary_interface__opposite_direction(
                                       mesh,
                                       level,
                                       flux_def.direction,
                                       flux_def.stencil,
                                       [&](auto& cell, auto& comput_cells)
                                       {
                                           auto flux_coeffs                        = flux_def.cons_flux_function(comput_cells);
                                           decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                                           auto cell_contrib                       = contribution(minus_flux_coeffs, h, h);
                                           apply_coeffs(cell, comput_cells, cell_contrib);
                                       });
                               });
            }
        }
    };

} // end namespace samurai
