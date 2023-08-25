#pragma once
#include "../../interface.hpp"
#include "../explicit_scheme.hpp"
#include "flux_definition.hpp"

namespace samurai
{
    /**
     * @class FluxBasedSchemeDefinition
     * Contains:
     * - how the flux of the field is computed
     * - how the flux contributes to the scheme
     */
    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    class FluxBasedSchemeDefinition
    {
      public:

        static constexpr std::size_t dim        = Field::dim;
        static constexpr std::size_t field_size = Field::size;

        using flux_computation_t = LinearNormalFluxDefinition<Field, stencil_size>;
        using field_value_type   = typename Field::value_type; // double
        using coeff_matrix_t     = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;
        using cell_coeffs_t      = xt::xtensor_fixed<coeff_matrix_t, xt::xshape<stencil_size>>;
        using flux_coeffs_t      = typename flux_computation_t::flux_coeffs_t;
        using cell_coeffs_func_t = std::function<cell_coeffs_t(flux_coeffs_t&, double, double)>;

      private:

        flux_computation_t m_flux;
        cell_coeffs_func_t m_contribution_func;
        cell_coeffs_func_t m_contribution_opposite_direction_func = nullptr;

      public:

        auto& flux() const
        {
            return m_flux;
        }

        auto& contribution_func()
        {
            return m_contribution_func;
        }

        auto& contribution_opposite_direction_func()
        {
            return m_contribution_opposite_direction_func;
        }

        void set_flux(const flux_computation_t& flux)
        {
            m_flux = flux;
        }

        void set_contribution(cell_coeffs_func_t contribution_func)
        {
            m_contribution_func = contribution_func;
            if (!m_contribution_opposite_direction_func)
            {
                m_contribution_opposite_direction_func = contribution_func;
            }
        }

        void set_contribution_opposite_direction(cell_coeffs_func_t contribution_func)
        {
            m_contribution_opposite_direction_func = contribution_func;
        }

        /**
         * Computes and returns the contribution coefficients
         */
        cell_coeffs_t contribution(flux_coeffs_t& flux, double h_face, double h_cell) const
        {
            return m_contribution_func(flux, h_face, h_cell);
        }

        cell_coeffs_t contribution_opposite_direction(flux_coeffs_t& flux, double h_face, double h_cell) const
        {
            return m_contribution_opposite_direction_func(flux, h_face, h_cell);
        }
    };

    template <std::size_t output_field_size_, std::size_t stencil_size_, bool is_linear_ = true, bool is_heterogeneous_ = false>
    struct FluxBasedSchemeConfig
    {
        static constexpr std::size_t output_field_size = output_field_size_;
        static constexpr std::size_t stencil_size      = stencil_size_;
        static constexpr bool is_linear                = is_linear_;
        static constexpr bool is_heterogeneous         = is_heterogeneous_;
    };

    template <class DerivedScheme, class cfg, class bdry_cfg, class Field>
    class FluxBasedScheme : public FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>;
        using base_class::dim;
        using base_class::field_size;
        using field_value_type = typename base_class::field_value_type;
        using mesh_t           = typename Field::mesh_t;

      public:

        using cfg_t                                    = cfg;
        using bdry_cfg_t                               = bdry_cfg;
        using field_t                                  = Field;
        static constexpr std::size_t output_field_size = cfg::output_field_size;
        static constexpr std::size_t stencil_size      = cfg::stencil_size;

        using scheme_definition_t = FluxBasedSchemeDefinition<Field, output_field_size, stencil_size>;

        explicit FluxBasedScheme(Field& unknown)
            : base_class(unknown)
        {
        }

        auto operator()(Field& f)
        {
            auto explicit_scheme = make_explicit(this->derived_cast());
            return explicit_scheme.apply_to(f);
        }

        template <class Func>
        void for_each_interior_interface(const mesh_t& mesh, Func&& apply_coeffs) const
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            auto definition = this->derived_cast().definition();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto scheme_def = definition[d];

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h                                  = cell_length(level);
                    auto flux_coeffs                        = scheme_def.flux().get_flux_coeffs(h);
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
                    auto flux_coeffs                        = scheme_def.flux().get_flux_coeffs(h_lp1); // flux computed at level l+1
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

        template <class Func>
        void for_each_boundary_interface(const mesh_t& mesh, Func&& apply_coeffs) const
        {
            auto definition = this->derived_cast().definition();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto scheme_def = definition[d];

                for_each_level(mesh,
                               [&](auto level)
                               {
                                   auto h = cell_length(level);

                                   // Boundary in direction
                                   auto flux_coeffs = scheme_def.flux().get_flux_coeffs(h);
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

    /**
     * is_FluxBasedScheme
     */
    template <class Scheme, typename = void>
    struct is_FluxBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_FluxBasedScheme<
        Scheme,
        std::enable_if_t<
            std::is_base_of_v<FluxBasedScheme<Scheme, typename Scheme::cfg_t, typename Scheme::bdry_cfg_t, typename Scheme::field_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    inline constexpr bool is_FluxBasedScheme_v = is_FluxBasedScheme<Scheme>::value;

} // end namespace samurai
