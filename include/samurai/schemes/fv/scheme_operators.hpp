#pragma once
#include "cell_based_scheme.hpp"
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * Multiplication by a scalar value of the flux-based scheme
     */
    template <class Scheme>
    class Scalar_x_FluxBasedScheme : public FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>
    {
      public:

        using cfg_t                     = typename Scheme::cfg_t;
        using bdry_cfg_t                = typename Scheme::bdry_cfg_t;
        using field_t                   = typename Scheme::field_t;
        using base_class                = FluxBasedScheme<cfg_t, bdry_cfg_t>;
        using flux_definition_t         = typename base_class::flux_definition_t;
        using directional_bdry_config_t = typename base_class::directional_bdry_config_t;
        using base_class::dim;
        using base_class::name;

      private:

        Scheme m_scheme;
        double m_scalar;

      public:

        Scalar_x_FluxBasedScheme(Scheme&& scheme, double scalar)
            : base_class(flux_definition_t())
            , m_scheme(std::move(scheme))
            , m_scalar(scalar)
        {
            build_scheme_definition();
        }

        Scalar_x_FluxBasedScheme(const Scheme& scheme, double scalar)
            : base_class(flux_definition_t())
            , m_scheme(scheme)
            , m_scalar(scalar)
        {
            build_scheme_definition();
        }

        Scalar_x_FluxBasedScheme(const Scalar_x_FluxBasedScheme<Scheme>& other)
            : base_class(flux_definition_t())
            , m_scheme(other.scheme())
            , m_scalar(other.scalar())
        {
            build_scheme_definition();
        }

        Scalar_x_FluxBasedScheme(Scalar_x_FluxBasedScheme<Scheme>&& other) noexcept
            : base_class(flux_definition_t())
            , m_scheme(other.scheme())
            , m_scalar(other.scalar())
        {
            build_scheme_definition();
        }

        Scalar_x_FluxBasedScheme& operator=(const Scalar_x_FluxBasedScheme& other)
        {
            if (this != &other)
            {
                this->m_scheme = other.m_scheme;
                this->m_scalar = other.m_scalar;
                this->build_scheme_definition();
            }
            return *this;
        }

        Scalar_x_FluxBasedScheme& operator=(Scalar_x_FluxBasedScheme&& other)
        {
            if (this != &other)
            {
                this->m_scheme = other.m_scheme;
                this->m_scalar = other.m_scalar;
                this->build_scheme_definition();
            }
            return *this;
        }

        auto scalar() const
        {
            return m_scalar;
        }

        auto& scheme()
        {
            return m_scheme;
        }

        auto& scheme() const
        {
            return m_scheme;
        }

        void build_scheme_definition()
        {
            static_for<0, dim>::apply(
                [&](auto integral_constant_d)
                {
                    static constexpr int d = decltype(integral_constant_d)::value;

                    this->flux_definition()[d] = m_scheme.flux_definition()[d];
                    if (m_scalar != 1)
                    {
                        // Multiply the flux function by the scalar
                        if constexpr (cfg_t::flux_type == FluxType::LinearHomogeneous)
                        {
                            this->flux_definition()[d].flux_function = [&](auto h)
                            {
                                return m_scalar * m_scheme.flux_definition()[d].flux_function(h);
                            };
                        }
                        else
                        {
                            this->flux_definition()[d].flux_function = [&](auto& field, auto& cells)
                            {
                                return m_scalar * m_scheme.flux_definition()[d].flux_function(field, cells);
                            };
                        }
                    }
                });

            this->dirichlet_config() = m_scheme.dirichlet_config();
            this->neumann_config()   = m_scheme.neumann_config();

            this->is_symmetric(m_scheme.is_symmetric());
            this->is_spd(m_scheme.is_spd() && m_scalar != 0);

            this->set_name(std::to_string(m_scalar) + " * " + m_scheme.name());
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
