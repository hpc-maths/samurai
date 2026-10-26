#pragma once
#include "../cell_based/cell_based_scheme__lin_hom.hpp"
#include "../cell_based/cell_based_scheme__nonlin.hpp"
#include "diffusion_bc.hpp"

namespace samurai
{
    /**
     * Assemble the matrix for the problem -Lap(u)=f.
     * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
     */
    template <class Field,
              // scheme config
              std::size_t dim                 = Field::dim,
              std::size_t neighbourhood_width = 1,
              class cfg                       = StarStencilSchemeConfig<SchemeType::LinearHomogeneous, neighbourhood_width, Field, Field>,
              class bdry_cfg                  = BoundaryConfigFV<neighbourhood_width>>
    class DiffusionFV_cell_based : public CellBasedScheme<cfg, bdry_cfg>
    {
        using base_class = CellBasedScheme<cfg, bdry_cfg>;
        using base_class::n_comp;

      public:

        using field_value_type          = typename Field::value_type;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV_cell_based()
        {
            this->set_name("Diffusion");

            this->stencil()           = star_stencil<dim>();
            this->coefficients_func() = [](StencilCoeffs<cfg>& coeffs, double h)
            {
                auto Identity = eye<field_value_type, n_comp, n_comp, Field::is_scalar>();
                for (unsigned int i = 0; i < cfg::stencil_size; ++i)
                {
                    coeffs[i] = -Identity;
                }
                coeffs[cfg::center_index] = (cfg::stencil_size - 1) * Identity;
                coeffs /= (h * h);
            };

            this->is_symmetric(true);
            this->is_spd(true);
            set_dirichlet_config();
            set_neumann_config();
        }

        void set_dirichlet_config()
        {
            detail::set_diffusion_dirichlet_config(this->dirichlet_config());
        }

        void set_neumann_config()
        {
            detail::set_diffusion_neumann_config(this->neumann_config());
        }
    };

    template <class Field>
    auto make_diffusion_cell_based()
    {
        return DiffusionFV_cell_based<Field>();
    }

    template <class Field>
    auto make_diffusion_cell_based_nonlin()
    {
        static constexpr std::size_t neighbourhood_width = 1;
        using input_field_t                              = Field;
        using output_field_t                             = Field;
        using cfg = StarStencilSchemeConfig<SchemeType::NonLinear, neighbourhood_width, output_field_t, input_field_t>;

        auto diff = make_cell_based_scheme<cfg>();

        diff.stencil() = star_stencil<Field::dim, neighbourhood_width>();

        diff.scheme_function() = [](SchemeValue<cfg>& value, const StencilCells<cfg>& cells, const input_field_t& field)
        {
            value  = 0.;
            auto h = cells[0].length;
            for (std::size_t i = 0; i < cfg::stencil_size; ++i)
            {
                value += (i == cfg::center_index ? (cfg::stencil_size - 1.) : -1.) * field[cells[i]];
            }
            value /= (h * h);
        };

        diff.jacobian_function() = [](StencilJacobian<cfg>& jac, const StencilCells<cfg>& cells, const input_field_t& /*field*/)
        {
            auto h = cells[0].length;
            for (std::size_t i = 0; i < cfg::stencil_size; ++i)
            {
                jac[i] = (i == cfg::center_index ? (cfg::stencil_size - 1.) : -1.)
                       * eye<typename Field::value_type, Field::n_comp, Field::n_comp, Field::is_scalar>();
            }
            jac /= (h * h);
        };

        diff.set_name("Diffusion");
        return diff;
    }

} // end namespace samurai
