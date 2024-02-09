#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * @class FluxBasedScheme
     *    Implementation of non-linear schemes
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
      public:

        using base_class = FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;

        using typename base_class::field_value_type;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

        using flux_definition_t  = FluxDefinition<cfg>;
        using flux_computation_t = typename flux_definition_t::flux_computation_t;
        using flux_value_t       = typename flux_computation_t::flux_value_t;

      private:

        flux_definition_t m_flux_definition;

      public:

        explicit FluxBasedScheme(const flux_definition_t& flux_definition)
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

        flux_value_t contribution(const flux_value_t& flux_value, double h_face, double h_cell) const
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            return (face_measure / cell_measure) * flux_value;
        }

        inline field_value_type flux_value_cmpnent(const flux_value_t& flux_value, [[maybe_unused]] std::size_t field_i) const
        {
            if constexpr (output_field_size == 1)
            {
                return flux_value;
            }
            else
            {
                return flux_value(field_i);
            }
        }

        /**
         * Iterates for each interior interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
        void for_each_interior_interface(input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h = cell_length(level);

                    for_each_interior_interface___same_level(mesh,
                                                             level,
                                                             flux_def.direction,
                                                             flux_def.stencil,
                                                             [&](auto& interface_cells, auto& comput_cells)
                                                             {
                                                                 auto flux_values        = flux_function(comput_cells, field);
                                                                 auto left_cell_contrib  = contribution(flux_values[0], h, h);
                                                                 auto right_cell_contrib = contribution(flux_values[1], h, h);
                                                                 apply_contrib(interface_cells, left_cell_contrib, right_cell_contrib);
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
                        for_each_interior_interface___level_jump_direction(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_values        = flux_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(flux_values[0], h_lp1, h_l);
                                auto right_cell_contrib = contribution(flux_values[1], h_lp1, h_lp1);
                                apply_contrib(interface_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                    //    |__|        l+1
                    //       |____|   l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface___level_jump_opposite_direction(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_values        = flux_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(flux_values[0], h_lp1, h_lp1);
                                auto right_cell_contrib = contribution(flux_values[1], h_lp1, h_l);
                                apply_contrib(interface_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                }
            }
        }

        /**
         * Iterates for each boundary interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
        void for_each_boundary_interface(input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

                for_each_level(mesh,
                               [&](auto level)
                               {
                                   auto h = cell_length(level);

                                   // Boundary in direction
                                   for_each_boundary_interface___direction(mesh,
                                                                           level,
                                                                           flux_def.direction,
                                                                           flux_def.stencil,
                                                                           [&](auto& cell, auto& comput_cells)
                                                                           {
                                                                               auto flux_values  = flux_function(comput_cells, field);
                                                                               auto cell_contrib = contribution(flux_values[0], h, h);
                                                                               apply_contrib(cell, cell_contrib);
                                                                           });

                                   // Boundary in opposite direction
                                   for_each_boundary_interface___opposite_direction(
                                       mesh,
                                       level,
                                       flux_def.direction,
                                       flux_def.stencil,
                                       [&](auto& cell, auto& comput_cells)
                                       {
                                           auto flux_values  = flux_function(comput_cells, field);
                                           auto cell_contrib = contribution(flux_values[1], h, h);
                                           apply_contrib(cell, cell_contrib);
                                       });
                               });
            }
        }
    };

} // end namespace samurai
