#pragma once
#include "flux_based_scheme.hpp"
#include "flux_implem.hpp"

namespace samurai
{
    /**
     * @class FluxBasedSchemeDefinition for non-linear fluxes
     * Contains:
     * - how the flux of the field is computed
     * - how the flux contributes to the scheme
     */
    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    class FluxBasedSchemeDefinition<FluxType::NonLinear, Field, output_field_size, stencil_size>
    {
      public:

        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;

        using flux_definition_t     = FluxDefinition<FluxType::NonLinear, Field, output_field_size, stencil_size>;
        using flux_computation_t    = typename flux_definition_t::flux_computation_t;
        using field_value_type      = typename Field::value_type;
        using scheme_contrib_t      = typename detail::LocalMatrix<field_value_type, output_field_size, 1>::Type;
        using flux_value_t          = typename flux_computation_t::flux_value_t;
        using flux_to_scheme_func_t = std::function<scheme_contrib_t(flux_value_t&)>;

      private:

        flux_computation_t m_flux;
        flux_to_scheme_func_t m_contribution_func = nullptr;

      public:

        FluxBasedSchemeDefinition()
        {
            if constexpr (std::is_same_v<scheme_contrib_t, flux_value_t>)
            {
                // By default, the contribution is the flux
                m_contribution_func = [](const flux_value_t& flux)
                {
                    return flux;
                };
            }
        }

        ~FluxBasedSchemeDefinition()
        {
            m_contribution_func = nullptr;
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

        void set_flux(const flux_computation_t& flux)
        {
            m_flux = flux;
        }

        void set_contribution(flux_to_scheme_func_t contribution_func)
        {
            m_contribution_func = contribution_func;
        }

        /**
         * Computes and returns the contribution coefficients
         */
        scheme_contrib_t contribution(flux_value_t& flux, double h_face, double h_cell) const
        {
            double face_measure = pow(h_face, dim - 1);
            double cell_measure = pow(h_cell, dim);
            return (face_measure / cell_measure) * m_contribution_func(flux);
        }
    };

    /**
     * @class FluxBasedScheme
     *    Implementation of non-linear schemes
     */
    template <class DerivedScheme, class cfg, class bdry_cfg, class Field>
    class FluxBasedScheme<DerivedScheme, cfg, bdry_cfg, Field, std::enable_if_t<cfg::flux_type == FluxType::NonLinear>>
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

        using scheme_definition_t = FluxBasedSchemeDefinition<FluxType::NonLinear, Field, output_field_size, stencil_size>;
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

        template <class Coeffs>
        inline static double cell_coeff(const Coeffs& coeffs, [[maybe_unused]] std::size_t field_i)
        {
            if constexpr (output_field_size == 1)
            {
                return coeffs;
            }
            else
            {
                return coeffs(field_i);
            }
        }

        /**
         * Iterates for each interior interface and returns (in lambda parameters) the scheme coefficients.
         */
        template <class Func>
        void for_each_interior_interface(Field& f, Func&& apply_contrib) const
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;

            auto& mesh = f.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& scheme_def = definition()[d];

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h = cell_length(level);

                    for_each_interior_interface___same_level(mesh,
                                                             level,
                                                             scheme_def.flux().direction,
                                                             scheme_def.flux().stencil,
                                                             [&](auto& interface_cells, auto& comput_cells)
                                                             {
                                                                 auto flux_value = scheme_def.flux().flux_function(f, comput_cells);
                                                                 decltype(flux_value) minus_flux_value = -flux_value;
                                                                 auto left_cell_contrib  = scheme_def.contribution(flux_value, h, h);
                                                                 auto right_cell_contrib = scheme_def.contribution(minus_flux_value, h, h);
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
                            scheme_def.flux().direction,
                            scheme_def.flux().stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_value                       = scheme_def.flux().flux_function(f, comput_cells);
                                decltype(flux_value) minus_flux_value = -flux_value;
                                auto left_cell_contrib                = scheme_def.contribution(flux_value, h_lp1, h_l);
                                auto right_cell_contrib               = scheme_def.contribution(minus_flux_value, h_lp1, h_lp1);
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
                            scheme_def.flux().direction,
                            scheme_def.flux().stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto flux_value                       = scheme_def.flux().flux_function(f, comput_cells);
                                decltype(flux_value) minus_flux_value = -flux_value;
                                auto left_cell_contrib                = scheme_def.contribution(flux_value, h_lp1, h_lp1);
                                auto right_cell_contrib               = scheme_def.contribution(minus_flux_value, h_lp1, h_l);
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
        void for_each_boundary_interface(Field& f, Func&& apply_contrib) const
        {
            auto& mesh = f.mesh();
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& scheme_def = definition()[d];

                for_each_level(mesh,
                               [&](auto level)
                               {
                                   auto h = cell_length(level);

                                   // Boundary in direction
                                   for_each_boundary_interface___direction(
                                       mesh,
                                       level,
                                       scheme_def.flux().direction,
                                       scheme_def.flux().stencil,
                                       [&](auto& cell, auto& comput_cells)
                                       {
                                           auto flux_value   = scheme_def.flux().flux_function(f, comput_cells);
                                           auto cell_contrib = scheme_def.contribution(flux_value, h, h);
                                           apply_contrib(cell, cell_contrib);
                                       });

                                   // Boundary in opposite direction
                                   for_each_boundary_interface___opposite_direction(
                                       mesh,
                                       level,
                                       scheme_def.flux().direction,
                                       scheme_def.flux().stencil,
                                       [&](auto& cell, auto& comput_cells)
                                       {
                                           auto flux_value                       = scheme_def.flux().flux_function(f, comput_cells);
                                           decltype(flux_value) minus_flux_value = -flux_value;
                                           auto cell_contrib                     = scheme_def.contribution(minus_flux_value, h, h);
                                           apply_contrib(cell, cell_contrib);
                                       });
                               });
            }
        }
    };

} // end namespace samurai
