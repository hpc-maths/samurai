#pragma once
// #include "flux_based_scheme.hpp"
#include "cell_based_scheme.hpp"

namespace samurai
{
    /*template<class Field, class cfg=FluxBasedAssemblyConfig<Field::size, 2>>
    class IdentityFV : public FluxBasedScheme<cfg, Field>
    {
    public:
        using cfg_t = cfg;
        using field_t = Field;
        using Mesh = typename field_t::mesh_t;
        using flux_computation_t = typename FluxBasedScheme<cfg, Field>::flux_computation_t;
        using flux_matrix_t  = typename flux_computation_t::flux_matrix_t;
        using coeff_matrix_t  = typename flux_computation_t::coeff_matrix_t;
        using CellCoeffs = typename flux_computation_t::CellCoeffs;
        using FluxCoeffs = typename flux_computation_t::FluxCoeffs;
        static constexpr std::size_t dim = field_t::dim;
    private:

    public:
        IdentityFV(Field& unknown) :
            FluxBasedScheme<cfg, Field>(unknown, identity_coefficients())
        {}

    private:
        template<std::size_t d>
        static auto flux_coefficients(double h)
        {
            FluxCoeffs flux_coeffs;
            if constexpr (Field::size == 1)
            {
                flux_coeffs[0] = -1/h;
                flux_coeffs[1] =  1/h;
            }
            else
            {
                flux_coeffs[0].fill(0);
                flux_coeffs[1].fill(0);
                flux_coeffs[0](d) = -1/h;
                flux_coeffs[1](d) =  1/h;
            }
            return flux_coeffs;
        }

        static auto get_zero_coeffs(FluxCoeffs&, double, double)
        {
            CellCoeffs coeffs;
            coeffs[0] = zeros<coeff_matrix_t>();
            coeffs[1] = zeros<coeff_matrix_t>();
            return coeffs;
        }



        static auto identity_coefficients()
        {
            static_assert(dim <= 3, "IdentityFV.scheme_coefficients() not implemented for dim > 3.");
            std::array<flux_computation_t, dim> fluxes;
            auto directions = samurai::positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux = fluxes[d];
                flux.direction = xt::view(directions, d);
                flux.computational_stencil = samurai::in_out_stencil<dim>(flux.direction);
                if (d == 0)
                {
                    flux.get_flux_coeffs = flux_coefficients<0>;
                    flux.get_cell1_coeffs = [&](auto&, double, double)
                    {
                        CellCoeffs coeffs;
                        coeffs[0] = eye<coeff_matrix_t>();
                        coeffs[1] = zeros<coeff_matrix_t>();
                        return coeffs;
                    };
                    flux.get_cell2_coeffs = get_zero_coeffs;
                }
                if constexpr (dim >= 2)
                {
                    if (d == 1)
                    {
                        flux.get_flux_coeffs = flux_coefficients<1>;
                        flux.get_cell1_coeffs = get_zero_coeffs;
                        flux.get_cell2_coeffs = get_zero_coeffs;
                    }
                }
                if constexpr (dim >= 3)
                {
                    if (d == 2)
                    {
                        flux.get_flux_coeffs = flux_coefficients<2>;
                        flux.get_cell1_coeffs = get_zero_coeffs;
                        flux.get_cell2_coeffs = get_zero_coeffs;
                    }
                }
            }
            return fluxes;
        }
    };*/

    template <class Field, class cfg = OneCellStencilFV<Field::size>, class bdry_cfg = BoundaryConfigFV<1>>
    class IdentityFV : public CellBasedScheme<IdentityFV<Field>, cfg, bdry_cfg, Field>
    {
        using base_class = CellBasedScheme<IdentityFV<Field>, cfg, bdry_cfg, Field>;
        using base_class::dim;
        using local_matrix_t = typename base_class::local_matrix_t;

      public:

        explicit IdentityFV(Field& unknown)
            : base_class(unknown)
        {
            this->set_name("Identity");
        }

        static constexpr auto stencil()
        {
            return center_only_stencil<dim>();
        }

        static std::array<local_matrix_t, 1> coefficients(double)
        {
            return {eye<local_matrix_t>()};
        }

        bool matrix_is_symmetric(const Field& unknown) const override
        {
            return is_uniform(unknown.mesh());
        }

        bool matrix_is_spd(const Field& unknown) const override
        {
            return matrix_is_symmetric(unknown);
        }
    };

    template <class Field>
    auto make_identity_FV(Field& f)
    {
        return IdentityFV<Field>(f);
    }

} // end namespace samurai
