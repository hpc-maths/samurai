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
        : public FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>
    {
      public:

        using field_t = typename cfg::input_field_t;

        using base_class = FVScheme<field_t, cfg::output_field_size, bdry_cfg>;

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;
        using field_value_type = typename base_class::field_value_type;
        using mesh_t           = typename field_t::mesh_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

        using flux_definition_t     = FluxDefinition<cfg>;
        using flux_computation_t    = typename flux_definition_t::flux_computation_t;
        using flux_stencil_coeffs_t = typename flux_computation_t::flux_stencil_coeffs_t;

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

        flux_stencil_coeffs_t contribution(const flux_stencil_coeffs_t& flux_coeffs, double h_face, double h_cell) const
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            return (face_measure / cell_measure) * flux_coeffs;
        }

        auto operator()(field_t& field)
        {
            auto explicit_scheme = make_explicit(*this);
            return explicit_scheme.apply_to(field);
        }

        /**
         * Iterates for each interior interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
        void for_each_interior_interface(const mesh_t& mesh, Func&& apply_coeffs) const
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h = cell_length(level);

                    for_each_interior_interface___same_level(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            auto flux_coeffs                               = flux_def.flux_function(comput_cells);
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
                        for_each_interior_interface___level_jump_direction(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_coeffs                        = flux_def.flux_function(comput_cells);
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
                        for_each_interior_interface___level_jump_opposite_direction(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_coeffs                        = flux_def.flux_function(comput_cells);
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
        void for_each_boundary_interface(const mesh_t& mesh, Func&& apply_coeffs) const
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

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
                                                                               auto flux_coeffs  = flux_def.flux_function(comput_cells);
                                                                               auto cell_contrib = contribution(flux_coeffs, h, h);
                                                                               apply_coeffs(cell, comput_cells, cell_contrib);
                                                                           });

                                   // Boundary in opposite direction
                                   for_each_boundary_interface___opposite_direction(
                                       mesh,
                                       level,
                                       flux_def.direction,
                                       flux_def.stencil,
                                       [&](auto& cell, auto& comput_cells)
                                       {
                                           auto flux_coeffs                        = flux_def.flux_function(comput_cells);
                                           decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                                           auto cell_contrib                       = contribution(minus_flux_coeffs, h, h);
                                           apply_coeffs(cell, comput_cells, cell_contrib);
                                       });
                               });
            }
        }
    };

} // end namespace samurai
