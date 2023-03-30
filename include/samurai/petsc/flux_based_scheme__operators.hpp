#pragma once
#include "flux_based_scheme.hpp"

namespace samurai 
{
    namespace petsc
    {
        
        template<class Scheme>
        class Scalar_x_FluxBasedScheme : public FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::field_t>
        {
        public:
            using cfg_t = typename Scheme::cfg_t;
            using field_t = typename Scheme::field_t;
            using Mesh = typename field_t::mesh_t;
            using flux_computation_t = typename FluxBasedScheme<cfg_t, field_t>::flux_computation_t;
            static constexpr std::size_t dim = field_t::dim;
        private:
            const Scheme& m_scheme;
            const double m_scalar;

        public:
            Scalar_x_FluxBasedScheme(const Scheme& scheme, double scalar) : 
                FluxBasedScheme<cfg_t, field_t>(scheme.unknown(), scheme_coefficients(scheme)),
                m_scheme(scheme),
                m_scalar(scalar)
            {}

        private:
            auto scheme_coefficients(const Scheme& scheme)
            {
                const std::array<flux_computation_t, dim>& scheme_fluxes = scheme.flux_computations();
                std::array<flux_computation_t, dim> scalar_x_fluxes;
                
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto& scheme_flux = scheme_fluxes[d];
                    auto& scalar_x_flux = scalar_x_fluxes[d];

                    //auto dir = op_flux.direction;
                    scalar_x_flux.direction             = scheme_flux.direction;
                    scalar_x_flux.computational_stencil = scheme_flux.computational_stencil;
                    scalar_x_flux.get_flux_coeffs       = scheme_flux.get_flux_coeffs;
                    scalar_x_flux.get_cell1_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs = scheme_flux.get_cell1_coeffs(flux_coeffs, h_face, h_cell);
                        for (auto& coeff : coeffs)
                        {
                            coeff *= m_scalar;
                        }
                        return coeffs;
                    };
                    scalar_x_flux.get_cell2_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs = scheme_flux.get_cell2_coeffs(flux_coeffs, h_face, h_cell);
                        for (auto& coeff : coeffs)
                        {
                            coeff *= m_scalar;
                        }
                        return coeffs;
                    };
                }
                return scalar_x_fluxes;
            }
        };

        template<class Scheme>
        auto operator*(double scalar, const Scheme& scheme)
        {
            return Scalar_x_FluxBasedScheme<Scheme>(scheme, scalar);
        }




        template<class Scheme1, class Scheme2>
        class Sum_FluxBasedScheme : public FluxBasedScheme<typename Scheme1::cfg_t, typename Scheme1::field_t>
        {
        public:
            using cfg_t = typename Scheme1::cfg_t;
            using field_t = typename Scheme1::field_t;
            using Mesh = typename field_t::mesh_t;
            using flux_computation_t = typename FluxBasedScheme<cfg_t, field_t>::flux_computation_t;
            static constexpr std::size_t dim = field_t::dim;
        private:
            const Scheme1& m_scheme1;
            const Scheme2& m_scheme2;

        public:
            Sum_FluxBasedScheme(const Scheme1& scheme1, const Scheme2& scheme2) : 
                FluxBasedScheme<cfg_t, field_t>(scheme1.unknown(), scheme_coefficients(scheme1, scheme2)),
                m_scheme1(scheme1),
                m_scheme2(scheme2)
            {
                static_assert(std::is_same<typename Scheme1::field_t, typename Scheme2::field_t>::value, "Invalid '+' operation: incompatible field types.");

                if (&scheme1.unknown() != &scheme2.unknown())
                {
                    std::cerr << "Invalid '+' operation: both schemes must be associated to the same unknown." << std::endl;
                    assert(&scheme1.unknown() == &scheme2.unknown());
                }
            }

        private:
            auto scheme_coefficients(const Scheme1& scheme1, const Scheme2& scheme2)
            {
                static_assert(Scheme1::cfg_t::comput_stencil_size == Scheme2::cfg_t::comput_stencil_size);

                auto& scheme1_fluxes = scheme1.flux_computations();
                auto& scheme2_fluxes = scheme2.flux_computations();
                std::array<flux_computation_t, dim> sum_fluxes = scheme1_fluxes;
                
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const auto& scheme1_flux = scheme1_fluxes[d];
                    const auto& scheme2_flux = scheme2_fluxes[d];
                    auto& sum_flux = sum_fluxes[d];
                    sum_flux.direction             = scheme2_flux.direction;
                    sum_flux.computational_stencil = scheme2_flux.computational_stencil;
                    sum_flux.get_flux_coeffs       = scheme2_flux.get_flux_coeffs;
                    sum_flux.get_cell1_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs1 = scheme1_flux.get_cell1_coeffs(flux_coeffs, h_face, h_cell);
                        auto coeffs2 = scheme2_flux.get_cell1_coeffs(flux_coeffs, h_face, h_cell);
                        decltype(coeffs1) coeffs;
                        for (std::size_t i=0; i<cfg_t::comput_stencil_size; ++i)
                        {
                            coeffs[i] = coeffs1[i] + coeffs2[i];
                        }
                        return coeffs;
                    };
                    sum_flux.get_cell2_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs1 = scheme1_flux.get_cell2_coeffs(flux_coeffs, h_face, h_cell);
                        auto coeffs2 = scheme2_flux.get_cell2_coeffs(flux_coeffs, h_face, h_cell);
                        decltype(coeffs1) coeffs;
                        for (std::size_t i=0; i<cfg_t::comput_stencil_size; ++i)
                        {
                            coeffs[i] = coeffs1[i] + coeffs2[i];
                        }
                        return coeffs;
                    };
                }
                return sum_fluxes;
            }
        };

        template <typename, typename = void>
        constexpr bool is_FluxBasedScheme{};

        template <typename T>
        constexpr bool is_FluxBasedScheme<T, std::void_t<decltype(std::declval<T>().flux_computations())> > = true;

        // just sum size of elements
        template <typename Scheme1, typename Scheme2, std::enable_if_t<is_FluxBasedScheme<Scheme1>, bool> = true, std::enable_if_t<is_FluxBasedScheme<Scheme2>, bool> = true>
        auto operator + (const Scheme1& s1, const Scheme2& s2 )
        {
            return Sum_FluxBasedScheme<Scheme1, Scheme2>(s1, s2);
        }

    } // end namespace petsc
} // end namespace samurai