#pragma once
#include "flux_based_scheme.hpp"

namespace samurai 
{
    namespace petsc
    {
        /**
         * @class DiffusionFV
         * Assemble the matrix for the problem -Lap(u)=f.
         * The matrix corresponds to the discretization of the operator -Lap by the Finite-Volume method.
        */
        template<class Field, DirichletEnforcement dirichlet_enfcmt=Equation, std::size_t dim=Field::dim, class cfg=FluxBasedAssemblyConfig<Field::size, 2, dirichlet_enfcmt>>
        class DiffusionFV : public FluxBasedScheme<cfg, Field>
        {
        public:
            using cfg_t = cfg;
            using field_t = Field;
            using Mesh = typename Field::mesh_t;
            using flux_computation_t = typename FluxBasedScheme<cfg, Field>::flux_computation_t;
            using flux_matrix_t = typename flux_computation_t::flux_matrix_t;
            using coeff_matrix_t = typename flux_computation_t::coeff_matrix_t;
            static constexpr std::size_t field_size = Field::size;

            DiffusionFV(Field& unknown) : 
                FluxBasedScheme<cfg, Field>(unknown, diffusion_coefficients())
            {
                this->set_name("Diffusion");
            }


            /*static auto flux_coefficients(double h)
            {
                std::array<flux_matrix_t, 2> flux_coeffs;
                if constexpr (field_size == 1)
                {
                    flux_coeffs[0] = -1/h;
                    flux_coeffs[1] =  1/h;
                }
                else
                {
                    for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                    {
                        flux_coeffs[0](field_j) = -1/h;
                        flux_coeffs[1](field_j) =  1/h;
                    }
                }
                return flux_coeffs;
            }*/
            template<std::size_t d>
            static auto flux_coefficients(double h)
            {
                std::array<flux_matrix_t, 2> flux_coeffs;
                if constexpr (field_size == 1)
                {
                    flux_coeffs[0] = -1/h;
                    flux_coeffs[1] =  1/h;
                }
                else
                {
                    /*flux_coeffs[0].fill(0);
                    flux_coeffs[1].fill(0);
                    flux_coeffs[0](d) = -1/h;
                    flux_coeffs[1](d) =  1/h;*/
                    flux_coeffs[0].fill(-1/h);
                    flux_coeffs[1].fill( 1/h);
                }
                return flux_coeffs;
            }

            template<std::size_t d>
            static auto get_laplacian_coeffs_cell1(std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
            {
                // std::cout << "flux = " << std::endl;
                // for (auto& coeff : flux_coeffs)
                // {
                //     std::cout << coeff << std::endl;
                // }
                double h_factor = pow(h_face, dim-1) / pow(h_cell, dim);
                std::array<coeff_matrix_t, 2> coeffs;
                if constexpr (field_size == 1)
                {
                    coeffs[0] = flux_coeffs[0] * h_factor;
                    coeffs[1] = flux_coeffs[1] * h_factor;
                }
                else
                {
                    coeffs[0].fill(0);
                    coeffs[1].fill(0);
                    for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                    {
                        coeffs[0](field_j, field_j) = flux_coeffs[0](field_j) * h_factor;
                        coeffs[1](field_j, field_j) = flux_coeffs[1](field_j) * h_factor;
                        // xt::row(coeffs[0], field_j) = xt::row(flux_coeffs[0], 0) * h_factor;
                        // xt::row(coeffs[1], field_j) = xt::row(flux_coeffs[1], 0) * h_factor;
                    }
                }
                // std::cout << "d = " << d << std::endl;
                // for (auto& coeff : coeffs)
                // {
                //     std::cout << coeff << std::endl;
                // }
                return coeffs;
            }


            static auto diffusion_coefficients()
            {
                std::array<flux_computation_t, dim> fluxes;
                auto directions = positive_cartesian_directions<dim>();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto& flux = fluxes[d];
                    flux.direction = xt::view(directions, d);
                    flux.computational_stencil = in_out_stencil<dim>(flux.direction);
                    /*flux.get_flux_coeffs = flux_coefficients;
                    flux.get_cell1_coeffs = [](std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
                    {
                        double h_factor = pow(h_face, dim-1) / pow(h_cell, dim);
                        std::array<coeff_matrix_t, 2> coeffs;
                        if constexpr (field_size == 1)
                        {
                            coeffs[0] = -flux_coeffs[0] * h_factor;
                            coeffs[1] = -flux_coeffs[1] * h_factor;
                        }
                        else
                        {
                            coeffs[0].fill(0);
                            coeffs[1].fill(0);
                            for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                            {
                                coeffs[0](field_j, field_j) = -flux_coeffs[0](field_j) * h_factor;
                                coeffs[1](field_j, field_j) = -flux_coeffs[1](field_j) * h_factor;
                            }
                        }
                        return coeffs;
                    };
                    flux.get_cell2_coeffs = [](std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
                    {
                        double h_factor = pow(h_face, dim-1) / pow(h_cell, dim);
                        std::array<coeff_matrix_t, 2> coeffs;
                        if constexpr (field_size == 1)
                        {
                            coeffs[0] = flux_coeffs[0] * h_factor;
                            coeffs[1] = flux_coeffs[1] * h_factor;
                        }
                        else
                        {
                            coeffs[0].fill(0);
                            coeffs[1].fill(0);
                            for (std::size_t field_j = 0; field_j < field_size; ++field_j)
                            {
                                coeffs[0](field_j, field_j) = flux_coeffs[0](field_j) * h_factor;
                                coeffs[1](field_j, field_j) = flux_coeffs[1](field_j) * h_factor;
                            }
                        }
                        return coeffs;
                    };*/
                    if (d == 0)
                    {
                        flux.get_flux_coeffs = flux_coefficients<0>;
                        flux.get_cell1_coeffs = [](std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
                        {
                            auto coeffs = get_laplacian_coeffs_cell1<0>(flux_coeffs, h_face, h_cell);
                            for (auto& coeff : coeffs)
                            {
                                coeff *= -1;
                            }
                            return coeffs;
                        };
                        flux.get_cell2_coeffs = get_laplacian_coeffs_cell1<0>;
                    }
                    if constexpr (dim >= 2)
                    {
                        if (d == 1)
                        {
                            flux.get_flux_coeffs = flux_coefficients<1>;
                            flux.get_cell1_coeffs = [](std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
                            {
                                auto coeffs = get_laplacian_coeffs_cell1<1>(flux_coeffs, h_face, h_cell);
                                for (auto& coeff : coeffs)
                                {
                                    coeff *= -1;
                                }
                                return coeffs;
                            };
                            flux.get_cell2_coeffs = get_laplacian_coeffs_cell1<1>;
                        }
                    }
                    if constexpr (dim >= 3)
                    {
                        if (d == 2)
                        {
                            flux.get_flux_coeffs = flux_coefficients<2>;
                            flux.get_cell1_coeffs = [](std::array<flux_matrix_t, 2>& flux_coeffs, double h_face, double h_cell)
                            {
                                auto coeffs = get_laplacian_coeffs_cell1<2>(flux_coeffs, h_face, h_cell);
                                for (auto& coeff : coeffs)
                                {
                                    coeff *= -1;
                                }
                                return coeffs;
                            };
                            flux.get_cell2_coeffs = get_laplacian_coeffs_cell1<2>;
                        }
                    }
                }
                return fluxes;
            }

            bool matrix_is_spd() const override
            {
                if constexpr (cfg::dirichlet_enfcmt == DirichletEnforcement::Elimination)
                {
                    // The projections/predictions kill the symmetry, so the matrix is spd only if the mesh is uniform.
                    return this->mesh().min_level() == this->mesh().max_level();
                }
                else
                {
                    return false;
                }
            }
        };


        template<DirichletEnforcement dirichlet_enfcmt=Equation, class Field>
        auto make_diffusion_FV(Field& f)
        {
            return DiffusionFV<Field, dirichlet_enfcmt>(f);
        }

    } // end namespace petsc
} // end namespace samurai