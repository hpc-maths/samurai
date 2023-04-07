#pragma once
// #include "../flux_based_scheme.hpp"
#include "../cell_based_scheme.hpp"

namespace samurai
{
    namespace petsc
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

        template <class Field, std::size_t dim = Field::dim, std::size_t neighbourhood_width = 0, class cfg = StarStencilFV<dim, dim, neighbourhood_width>>
        class IdentityFV : public CellBasedScheme<cfg, Field>
        {
          public:

            using local_matrix_t = typename CellBasedScheme<cfg, Field>::local_matrix_t;

            IdentityFV(Field& unknown)
                : CellBasedScheme<cfg, Field>(unknown, star_stencil<dim, neighbourhood_width>(), coefficients)
            {
                this->set_name("Identity");
            }

            static std::array<local_matrix_t, cfg::scheme_stencil_size> coefficients(double)
            {
                std::array<local_matrix_t, cfg::scheme_stencil_size> coeffs;

                for (std::size_t i = 0; i < cfg::scheme_stencil_size; i++)
                {
                    if (i == cfg::center_index)
                    {
                        coeffs[i] = eye<local_matrix_t>();
                    }
                    else
                    {
                        coeffs[i] = zeros<local_matrix_t>();
                    }
                }
                return coeffs;
            }
        };

        template <class Field>
        auto make_identity_FV(Field& f)
        {
            return IdentityFV<Field>(f);
        }

    } // end namespace petsc
} // end namespace samurai