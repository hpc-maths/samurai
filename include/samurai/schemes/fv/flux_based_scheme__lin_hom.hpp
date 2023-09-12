#pragma once
#include "flux_based_scheme.hpp"
#include "flux_implem.hpp"

namespace samurai
{
    /**
     * @class FluxBasedSchemeDefinition for linear and homogeneous fluxes
     * Contains:
     * - how the flux of the field is computed
     * - how the flux contributes to the scheme
     */
    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    class FluxBasedSchemeDefinition<FluxType::LinearHomogeneous, Field, output_field_size, stencil_size>
    {
      public:

        static constexpr std::size_t dim                    = Field::dim;
        static constexpr std::size_t field_size             = Field::size;
        static constexpr std::size_t flux_output_field_size = field_size;

        using flux_definition_t       = FluxDefinition<FluxType::LinearHomogeneous, Field, flux_output_field_size, stencil_size>;
        using flux_computation_t      = typename flux_definition_t::flux_computation_t;
        using field_value_type        = typename Field::value_type;
        using scheme_coeff_matrix_t   = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;
        using scheme_stencil_coeffs_t = xt::xtensor_fixed<scheme_coeff_matrix_t, xt::xshape<stencil_size>>;
        using flux_stencil_coeffs_t   = typename flux_computation_t::flux_stencil_coeffs_t;
        using flux_to_scheme_func_t   = std::function<scheme_stencil_coeffs_t(flux_stencil_coeffs_t&)>;

      private:

        flux_computation_t m_flux;
        flux_to_scheme_func_t m_contribution_func;
        flux_to_scheme_func_t m_contribution_opposite_direction_func = nullptr;

      public:

        ~FluxBasedSchemeDefinition()
        {
            m_contribution_func                    = nullptr;
            m_contribution_opposite_direction_func = nullptr;
        }

        auto& flux() const
        {
            return m_flux;
        }

        auto& flux()
        {
            return m_flux;
        }

        auto& contribution_func() const
        {
            return m_contribution_func;
        }

        auto& contribution_opposite_direction_func() const
        {
            return m_contribution_opposite_direction_func;
        }

        void set_flux(const flux_computation_t& flux)
        {
            m_flux = flux;
        }

        void set_contribution(flux_to_scheme_func_t contribution_func)
        {
            m_contribution_func = contribution_func;
            if (!m_contribution_opposite_direction_func)
            {
                m_contribution_opposite_direction_func = contribution_func;
            }
        }

        void set_contribution_opposite_direction(flux_to_scheme_func_t contribution_func)
        {
            m_contribution_opposite_direction_func = contribution_func;
        }

        /**
         * Computes and returns the contribution coefficients
         */
        scheme_stencil_coeffs_t contribution(flux_stencil_coeffs_t& flux, double h_face, double h_cell) const
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            return (face_measure / cell_measure) * m_contribution_func(flux);
        }

        scheme_stencil_coeffs_t contribution_opposite_direction(flux_stencil_coeffs_t& flux, double h_face, double h_cell) const
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            return (face_measure / cell_measure) * m_contribution_opposite_direction_func(flux);
        }
    };

    /**
     * @class FluxBasedScheme
     *    Implementation of LINEAR and HOMOGENEOUS schemes
     */
    template <class DerivedScheme, class cfg, class bdry_cfg, class Field>
    class FluxBasedScheme<DerivedScheme, cfg, bdry_cfg, Field, std::enable_if_t<cfg::flux_type == FluxType::LinearHomogeneous>>
        : public FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>;

      public:

        using base_class::dim;
        using base_class::field_size;
        using base_class::output_field_size;
        using field_value_type = typename base_class::field_value_type;
        using mesh_t           = typename Field::mesh_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;
        using field_t    = Field;

        static constexpr std::size_t stencil_size = cfg::stencil_size;

        using scheme_definition_t = FluxBasedSchemeDefinition<FluxType::LinearHomogeneous, Field, output_field_size, stencil_size>;
        using flux_definition_t   = typename scheme_definition_t::flux_definition_t;

      protected:

        std::array<scheme_definition_t, dim> m_scheme_definition;

      public:

        explicit FluxBasedScheme(const flux_definition_t& flux_definition)
        {
            add_flux_to_scheme_definition(flux_definition);
        }

      private:

        void add_flux_to_scheme_definition(const flux_definition_t& flux_definition)
        {
            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; d++)
            {
                DirectionVector<dim> direction = xt::view(directions, d);
                assert(direction == flux_definition[d].direction
                       && "The flux definitions must be added in the following order: 1) x-direction, 2) y-direction, 3) z-direction.");
                m_scheme_definition[d].set_flux(flux_definition[d]);
            }
        }

      public:

        auto& definition() const
        {
            return m_scheme_definition;
        }

        auto& definition()
        {
            return m_scheme_definition;
        }

        auto operator()(Field& f)
        {
            auto explicit_scheme = make_explicit(this->derived_cast());
            return explicit_scheme.apply_to(f);
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
                auto& scheme_def = definition()[d];

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h                                  = cell_length(level);
                    auto flux_coeffs                        = scheme_def.flux().flux_function(h);
                    decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;

                    auto left_cell_coeffs  = scheme_def.contribution(flux_coeffs, h, h);
                    auto right_cell_coeffs = scheme_def.contribution_opposite_direction(minus_flux_coeffs, h, h);

                    for_each_interior_interface___same_level(
                        mesh,
                        level,
                        scheme_def.flux().direction,
                        scheme_def.flux().stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            apply_coeffs(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
                        });
                }

                // Level jumps (level -- level+1)
                for (std::size_t level = min_level; level < max_level; ++level)
                {
                    auto h_l                                = cell_length(level);
                    auto h_lp1                              = cell_length(level + 1);
                    auto flux_coeffs                        = scheme_def.flux().flux_function(h_lp1); // flux computed at level l+1
                    decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;

                    //         |__|   l+1
                    //    |____|      l
                    //    --------->
                    //    direction
                    {
                        auto left_cell_coeffs  = scheme_def.contribution(flux_coeffs, h_lp1, h_l);
                        auto right_cell_coeffs = scheme_def.contribution_opposite_direction(minus_flux_coeffs, h_lp1, h_lp1);

                        for_each_interior_interface___level_jump_direction(
                            mesh,
                            level,
                            scheme_def.flux().direction,
                            scheme_def.flux().stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                apply_coeffs(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
                            });
                    }
                    //    |__|        l+1
                    //       |____|   l
                    //    --------->
                    //    direction
                    {
                        auto left_cell_coeffs  = scheme_def.contribution(flux_coeffs, h_lp1, h_lp1);
                        auto right_cell_coeffs = scheme_def.contribution_opposite_direction(minus_flux_coeffs, h_lp1, h_l);

                        for_each_interior_interface___level_jump_opposite_direction(
                            mesh,
                            level,
                            scheme_def.flux().direction,
                            scheme_def.flux().stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                apply_coeffs(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
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
                auto& scheme_def = definition()[d];

                for_each_level(mesh,
                               [&](auto level)
                               {
                                   auto h = cell_length(level);

                                   // Boundary in direction
                                   auto flux_coeffs = scheme_def.flux().flux_function(h);
                                   auto cell_coeffs = scheme_def.contribution(flux_coeffs, h, h);
                                   for_each_boundary_interface___direction(mesh,
                                                                           level,
                                                                           scheme_def.flux().direction,
                                                                           scheme_def.flux().stencil,
                                                                           [&](auto& cell, auto& comput_cells)
                                                                           {
                                                                               apply_coeffs(cell, comput_cells, cell_coeffs);
                                                                           });

                                   // Boundary in opposite direction
                                   decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                                   cell_coeffs = scheme_def.contribution_opposite_direction(minus_flux_coeffs, h, h);
                                   for_each_boundary_interface___opposite_direction(mesh,
                                                                                    level,
                                                                                    scheme_def.flux().direction,
                                                                                    scheme_def.flux().stencil,
                                                                                    [&](auto& cell, auto& comput_cells)
                                                                                    {
                                                                                        apply_coeffs(cell, comput_cells, cell_coeffs);
                                                                                    });
                               });
            }
        }
    };

} // end namespace samurai
