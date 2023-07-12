#pragma once
#include "cell_based_scheme.hpp"
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * Multiplication by a scalar value of the flux-based scheme
     */
    template <class Scheme> //, std::enable_if_t<Scheme::is_flux_based>>
    class Scalar_x_FluxBasedScheme
        : public FluxBasedScheme<Scalar_x_FluxBasedScheme<Scheme>, typename Scheme::cfg_t, typename Scheme::bdry_cfg_t, typename Scheme::field_t>
    {
      public:

        using cfg_t                      = typename Scheme::cfg_t;
        using bdry_cfg_t                 = typename Scheme::bdry_cfg_t;
        using field_t                    = typename Scheme::field_t;
        using base_class                 = FluxBasedScheme<Scalar_x_FluxBasedScheme<Scheme>, cfg_t, bdry_cfg_t, field_t>;
        using coefficients_t             = typename base_class::coefficients_t;
        using flux_coeffs_t              = typename coefficients_t::flux_coeffs_t;
        static constexpr std::size_t dim = field_t::dim;
        using base_class::name;

      private:

        Scheme m_scheme;
        double m_scalar;

      public:

        Scalar_x_FluxBasedScheme(Scheme&& scheme, double scalar)
            : base_class(scheme.unknown())
            , m_scheme(std::move(scheme))
            , m_scalar(scalar)
        {
            this->set_name(std::to_string(m_scalar) + " * " + m_scheme.name());
        }

        Scalar_x_FluxBasedScheme(const Scheme& scheme, double scalar)
            : base_class(scheme.unknown())
            , m_scheme(scheme)
            , m_scalar(scalar)
        {
            this->set_name(std::to_string(m_scalar) + " * " + m_scheme.name());
        }

        auto scalar() const
        {
            return m_scalar;
        }

        auto& scheme()
        {
            return m_scheme;
        }

        auto coefficients() const
        {
            std::array<coefficients_t, dim> scalar_x_fluxes = m_scheme.coefficients();
            static_for<0, dim>::apply(
                [&](auto integral_constant_d)
                {
                    static constexpr int d                  = decltype(integral_constant_d)::value;
                    scalar_x_fluxes[d].get_left_cell_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        return this->scalar_x_get_cell1_coeffs<d>(flux_coeffs, h_face, h_cell);
                    };
                    scalar_x_fluxes[d].get_right_cell_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        return this->scalar_x_get_cell2_coeffs<d>(flux_coeffs, h_face, h_cell);
                    };
                });
            return scalar_x_fluxes;
        }

        template <std::size_t d>
        auto scalar_x_get_cell1_coeffs(flux_coeffs_t& flux_coeffs, double h_face, double h_cell) const
        {
            auto coeffs = m_scheme.coefficients()[d].get_left_cell_coeffs(flux_coeffs, h_face, h_cell);
            for (auto& coeff : coeffs)
            {
                coeff *= m_scalar;
            }
            return coeffs;
        }

        template <std::size_t d>
        auto scalar_x_get_cell2_coeffs(flux_coeffs_t& flux_coeffs, double h_face, double h_cell) const
        {
            auto coeffs = m_scheme.coefficients()[d].get_right_cell_coeffs(flux_coeffs, h_face, h_cell);
            for (auto& coeff : coeffs)
            {
                coeff *= m_scalar;
            }
            return coeffs;
        }

        bool matrix_is_symmetric(const field_t&) const override
        {
            // return m_scheme.matrix_is_symmetric();
            return false;
        }

        bool matrix_is_spd(const field_t&) const override
        {
            // if (m_scheme.matrix_is_spd())
            // {
            //     return m_scalar > 0;
            // }
            return false;
        }
    };

    template <class Scheme>
    auto operator*(double scalar, const Scheme& scheme)
        -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>(scheme, scalar);
    }

    template <class Scheme>
    auto operator*(double scalar, Scheme& scheme)
        -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>(scheme, scalar);
    }

    template <class Scheme>
    auto operator*(double scalar, Scheme&& scheme)
        -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>(std::forward<Scheme>(scheme), scalar);
    }

    template <class Scheme>
    auto operator*(double scalar, Scalar_x_FluxBasedScheme<Scheme>& scalar_x_scheme)
        -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>(scalar_x_scheme.scheme(), scalar * scalar_x_scheme.scalar());
    }

    template <class Scheme>
    auto operator*(double scalar, Scalar_x_FluxBasedScheme<Scheme>&& scalar_x_scheme)
        -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>(scalar_x_scheme.scheme(), scalar * scalar_x_scheme.scalar());
    }

    template <class Scheme>
    auto operator-(const Scheme& scheme)
        -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return (-1) * scheme;
    }

    template <class Scheme>
    auto operator-(Scheme& scheme) -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return (-1) * scheme;
    }

    template <class Scheme>
    auto operator-(Scheme&& scheme) -> std::enable_if_t<is_FluxBasedScheme<Scheme>::value, Scalar_x_FluxBasedScheme<std::decay_t<Scheme>>>
    {
        return (-1) * scheme;
    }

    /**
     * Addition of a flux-based scheme and a cell-based scheme.
     * The cell-based scheme is assembled first, then the flux-based scheme.
     * The boundary conditions are taken from the flux-based scheme.
     */
    template <class FluxScheme, class CellScheme>
    class FluxBasedScheme_Sum_CellBasedScheme
    {
      public:

        using field_t = typename FluxScheme::field_t;

      private:

        std::string m_name = "(unnamed)";
        FluxScheme m_flux_scheme;
        CellScheme m_cell_scheme;

      public:

        FluxBasedScheme_Sum_CellBasedScheme(const FluxScheme& flux_scheme, const CellScheme& cell_scheme)
            : m_flux_scheme(flux_scheme)
            , m_cell_scheme(cell_scheme)
        {
            this->m_name = m_flux_scheme.name() + " + " + m_cell_scheme.name();
            if (&flux_scheme.unknown() != &cell_scheme.unknown())
            {
                std::cerr << "Invalid '+' operation: both schemes must be associated to the same unknown." << std::endl;
                assert(&flux_scheme.unknown() == &cell_scheme.unknown());
            }
        }

        std::string name() const
        {
            return m_name;
        }

        auto& flux_scheme() const
        {
            return m_flux_scheme;
        }

        auto& cell_scheme() const
        {
            return m_cell_scheme;
        }

        auto& unknown() const
        {
            return m_flux_scheme.unknown();
        }

        bool matrix_is_symmetric() const // override
        {
            return m_flux_scheme.matrix_is_symmetric() && m_cell_scheme.matrix_is_symmetric();
        }

        bool matrix_is_spd() const // override
        {
            return m_flux_scheme.matrix_is_spd() && m_cell_scheme.matrix_is_spd();
        }
    };

    // Operator +
    template <typename FluxScheme,
              typename CellScheme,
              std::enable_if_t<is_CellBasedScheme_v<CellScheme> && is_FluxBasedScheme_v<FluxScheme>, bool> = true>
    auto operator+(const FluxScheme& flux_scheme, const CellScheme& cell_scheme)
    {
        return FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>(flux_scheme, cell_scheme);
    }

    // Operator + with reference rvalue
    /*template <typename FluxScheme, typename CellScheme, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true,
    std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true> auto operator + (FluxScheme&& flux_scheme, const CellScheme&
    cell_scheme)
    {
        //return flux_scheme + cell_scheme;
        return FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>(flux_scheme, cell_scheme);
    }*/

    // Operator + in the reverse order
    template <typename CellScheme,
              typename FluxScheme,
              std::enable_if_t<is_CellBasedScheme_v<CellScheme> && is_FluxBasedScheme_v<FluxScheme>, bool> = true>
    auto operator+(const CellScheme& cell_scheme, const FluxScheme& flux_scheme)
    {
        return flux_scheme + cell_scheme;
    }

    // Operator + in the reverse order with reference rvalue
    /*template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true,
    std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true> auto operator + (const CellScheme& cell_scheme, FluxScheme&&
    flux_scheme)
    {
        return flux_scheme + cell_scheme;
    }*/

    // Operator + with reference rvalue
    /*template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true,
    std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true> auto operator + (CellScheme&& cell_scheme, FluxScheme&& flux_scheme)
    {
        return flux_scheme + cell_scheme;
    }*/

} // end namespace samurai
