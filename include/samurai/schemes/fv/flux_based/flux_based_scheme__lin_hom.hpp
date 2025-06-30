#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * @class FluxBasedScheme
     *    Implementation of LINEAR and HOMOGENEOUS schemes
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : public FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
      public:

        using base_class = FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

        using base_class::dim;
        using base_class::n_comp;
        using base_class::output_n_comp;
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
            double face_measure = std::pow(h_face, dim - 1);
            double cell_measure = std::pow(h_cell, dim);
            return (face_measure / cell_measure) * flux_coeffs;
        }

        /**
         * Iterates for each interior interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_interior_interface_and_coeffs(std::size_t d, input_field_t& field, Func&& apply_coeffs) const
        {
            auto& mesh = field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            auto& flux_def = flux_definition()[d];

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h           = mesh.cell_length(level);
                auto flux_coeffs = flux_def.cons_flux_function(h);

                auto left_cell_coeffs                        = contribution(flux_coeffs, h, h);
                decltype(left_cell_coeffs) right_cell_coeffs = -left_cell_coeffs;

                for_each_interior_interface__same_level<run_type, get_type>(
                    mesh,
                    level,
                    flux_def.direction,
                    flux_def.stencil,
                    [&](auto& interface, auto& stencil)
                    {
                        apply_coeffs(interface, stencil, left_cell_coeffs, right_cell_coeffs);
                    });
            }

            // Level jumps (level -- level+1)
            for (std::size_t level = min_level; level < max_level; ++level)
            {
                auto h_l                                = mesh.cell_length(level);
                auto h_lp1                              = mesh.cell_length(level + 1);
                auto flux_coeffs                        = flux_def.cons_flux_function(h_lp1); // flux computed at level l+1
                decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                {
                    auto left_cell_coeffs  = contribution(flux_coeffs, h_lp1, h_l);
                    auto right_cell_coeffs = contribution(minus_flux_coeffs, h_lp1, h_lp1);

                    for_each_interior_interface__level_jump_direction<run_type, get_type>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface, auto& stencil)
                        {
                            apply_coeffs(interface, stencil, left_cell_coeffs, right_cell_coeffs);
                        });
                }
                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                {
                    auto left_cell_coeffs  = contribution(flux_coeffs, h_lp1, h_lp1);
                    auto right_cell_coeffs = contribution(minus_flux_coeffs, h_lp1, h_l);

                    for_each_interior_interface__level_jump_opposite_direction<run_type, get_type>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface, auto& stencil)
                        {
                            apply_coeffs(interface, stencil, left_cell_coeffs, right_cell_coeffs);
                        });
                }
            }
        }

        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_interior_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                for_each_interior_interface_and_coeffs<run_type, get_type>(d, field, std::forward<Func>(apply_coeffs));
            }
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

            for_each_level(mesh,
                           [&](auto level)
                           {
                               auto h = mesh.cell_length(level);

                               // Boundary in direction
                               auto flux_coeffs = flux_def.cons_flux_function(h);
                               auto cell_coeffs = contribution(flux_coeffs, h, h);
                               for_each_boundary_interface__direction<run_type, get_type>(mesh,
                                                                                          level,
                                                                                          flux_def.direction,
                                                                                          flux_def.stencil,
                                                                                          [&](auto& cell, auto& comput_cells)
                                                                                          {
                                                                                              apply_coeffs(cell, comput_cells, cell_coeffs);
                                                                                          });

                               // Boundary in opposite direction
                               decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                               cell_coeffs                             = contribution(minus_flux_coeffs, h, h);
                               for_each_boundary_interface__opposite_direction<run_type, get_type>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& cell, auto& comput_cells)
                                   {
                                       apply_coeffs(cell, comput_cells, cell_coeffs);
                                   });
                           });
        }

        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Func>
        void for_each_boundary_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            auto& mesh = field.mesh();
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (mesh.periodicity()[d])
                {
                    continue; // no boundary in this direction
                }

                for_each_boundary_interface_and_coeffs<run_type, get_type>(d, field, std::forward<Func>(apply_coeffs));
            }
        }
    };

} // end namespace samurai
