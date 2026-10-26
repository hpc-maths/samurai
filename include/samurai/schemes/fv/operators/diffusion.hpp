#pragma once
#include "../flux_based/flux_based_scheme__lin_het.hpp"
#include "../flux_based/flux_based_scheme__lin_hom.hpp"
#include "diffusion_bc.hpp"
#include "divergence.hpp"

namespace samurai
{
    /**
     * Finite volume discretization of the -Laplacian, defined by its fluxes.
     */
    template <class cfg, class bdry_cfg = BoundaryConfigFV<cfg::stencil_size / 2>>
    class DiffusionFV : public FluxBasedScheme<cfg, bdry_cfg>
    {
        using base_class = FluxBasedScheme<cfg, bdry_cfg>;

      public:

        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;

        explicit DiffusionFV(const FluxDefinition<cfg>& flux_definition)
            : base_class(flux_definition)
        {
            this->set_name("Diffusion");
            // this->is_symmetric(true);
            // this->is_spd(true);

            set_dirichlet_config();
            set_neumann_config();
        }

        //---------------------------------//
        //       Boundary conditions       //
        //---------------------------------//

        void set_dirichlet_config()
        {
            detail::set_diffusion_dirichlet_config(this->dirichlet_config());
        }

        void set_neumann_config()
        {
            detail::set_diffusion_neumann_config(this->neumann_config());
        }
    };

    template <std::size_t size>
    using DiffCoeff = xt::xtensor_fixed<double, xt::xshape<size>>;

    /**
     * Linear homogeneous diffusion
     */

    template <class cfg>
    auto make_diffusion__(const FluxDefinition<cfg>& flux_definition)
    {
        return DiffusionFV<cfg>(flux_definition);
    }

    namespace detail
    {
        /**
         * Linear homogeneous diffusion with the two-point flux K (u_right - u_left)/h on each component.
         * scale(coeffs, d) multiplies the coefficients of (u_right - u_left)/h in the direction d by -K
         * (minus sign because we want -Laplacian).
         */
        template <class Field, class Scale>
        auto make_two_point_diffusion(Scale scale)
        {
            static constexpr std::size_t dim = Field::dim;

            static constexpr std::size_t stencil_size = 2;
            using input_field_t                       = Field;
            using output_field_t                      = Field;

            using cfg = FluxConfig<SchemeType::LinearHomogeneous, stencil_size, output_field_t, input_field_t>;

            FluxDefinition<cfg> K_grad;

            static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
                [&](auto _d)
                {
                    static constexpr std::size_t d = _d();

                    K_grad[d].cons_flux_function = [scale](FluxStencilCoeffs<cfg>& coeffs, double h)
                    {
                        static constexpr std::size_t left  = 0;
                        static constexpr std::size_t right = 1;

                        // `coeffs` must be filled.
                        // It contains 2 matrices (left, right) of size output_n_comp x input_n_comp.
                        // In this case, of size n_comp x n_comp.

                        if constexpr (Field::is_scalar)
                        {
                            coeffs[left]  = -1 / h;
                            coeffs[right] = 1 / h;
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            for (std::size_t i = 0; i < Field::n_comp; ++i)
                            {
                                coeffs[left](i, i)  = -1 / h;
                                coeffs[right](i, i) = 1 / h;
                            }
                        }
                        scale(coeffs, d);
                    };
                });
            auto scheme = make_diffusion__<cfg>(K_grad);
            scheme.set_name("diffusion");
            return scheme;
        }
    }

    template <class Field>
    auto make_diffusion_order2(const DiffCoeff<Field::dim>& K)
    {
        return detail::make_two_point_diffusion<Field>(
            [K](auto& coeffs, std::size_t d)
            {
                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                coeffs[left] *= -K(d);
                coeffs[right] *= -K(d);
            });
    }

    /**
     * Diffusion operator with a different coefficient for each field component
     */
    template <class Field>
    auto make_multi_diffusion_order2(const DiffCoeff<Field::n_comp>& K)
    {
        return detail::make_two_point_diffusion<Field>(
            [K](auto& coeffs, std::size_t)
            {
                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                if constexpr (Field::is_scalar)
                {
                    coeffs[left] *= -K(0);
                    coeffs[right] *= -K(0);
                }
                else
                {
                    for (std::size_t i = 0; i < Field::n_comp; ++i)
                    {
                        coeffs[left](i, i) *= -K(i);
                        coeffs[right](i, i) *= -K(i);
                    }
                }
            });
    }

    template <class Field>
    auto make_diffusion_order2(double k)
    {
        DiffCoeff<Field::dim> K;
        K.fill(k);
        return make_diffusion_order2<Field>(K);
    }

    template <class Field>
    auto make_diffusion_order2()
    {
        return make_diffusion_order2<Field>(1.);
    }

    template <class Field>
    auto make_laplacian_order2()
    {
        return -make_diffusion_order2<Field>();
    }

    /**
     * Linear heterogeneous diffusion
     */

    template <class field_t,
              class DiffTensorField,
              std::enable_if_t<DiffTensorField::is_scalar && std::is_same_v<typename DiffTensorField::value_type, DiffCoeff<field_t::dim>>, bool> = true>
    auto make_diffusion_order2(const DiffTensorField& K)
    {
        static constexpr std::size_t dim    = field_t::dim;
        static constexpr std::size_t n_comp = field_t::n_comp;

        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = field_t;
        using output_field_t                      = field_t;

        using cfg = FluxConfig<SchemeType::LinearHeterogeneous, stencil_size, output_field_t, input_field_t>;

        FluxDefinition<cfg> K_grad;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                K_grad[d].cons_flux_function = [&](FluxStencilCoeffs<cfg>& coeffs, const StencilData<cfg>& data)
                {
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    double h = data.cell_length;

                    auto k_left  = K[data.cells[left]];
                    auto k_right = K[data.cells[right]];

                    // `coeffs`: 2 matrices (left, right) of size output_n_comp x n_comp.
                    // In this case, of size n_comp x n_comp.
                    if constexpr (field_t::is_scalar)
                    {
                        coeffs[left]  = -k_left(d) / h;
                        coeffs[right] = k_left(d) / h;
                    }
                    else
                    {
                        coeffs[left].fill(0);
                        coeffs[right].fill(0);
                        for (std::size_t i = 0; i < n_comp; ++i)
                        {
                            coeffs[left](i, i)  = -k_left(d) / h;
                            coeffs[right](i, i) = k_left(d) / h;
                        }
                    }
                    // Because we want -Laplacian
                    coeffs[left] *= -1;
                    coeffs[right] *= -1;
                };
            });

        auto scheme = make_diffusion__<cfg>(K_grad);
        scheme.set_name("diffusion");
        return scheme;
    }

} // end namespace samurai
