#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
    /**
     * Multiplication by a scalar value of the flux-based scheme
     */
    // template <class Scheme>
    // class Scalar_x_FluxBasedScheme : public FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>
    // {
    //   public:

    //     using cfg_t                     = typename Scheme::cfg_t;
    //     using bdry_cfg_t                = typename Scheme::bdry_cfg_t;
    //     using field_t                   = typename Scheme::field_t;
    //     using base_class                = FluxBasedScheme<cfg_t, bdry_cfg_t>;
    //     using flux_definition_t         = typename base_class::flux_definition_t;
    //     using directional_bdry_config_t = typename base_class::directional_bdry_config_t;
    //     using base_class::dim;
    //     using base_class::name;

    //   private:

    //     Scheme m_scheme;
    //     double m_scalar;

    //   public:

    //     Scalar_x_FluxBasedScheme(Scheme&& scheme, double scalar)
    //         : base_class(flux_definition_t())
    //         , m_scheme(std::move(scheme))
    //         , m_scalar(scalar)
    //     {
    //         build_scheme_definition();
    //     }

    //     Scalar_x_FluxBasedScheme(const Scheme& scheme, double scalar)
    //         : base_class(flux_definition_t())
    //         , m_scheme(scheme)
    //         , m_scalar(scalar)
    //     {
    //         build_scheme_definition();
    //     }

    //     Scalar_x_FluxBasedScheme(const Scalar_x_FluxBasedScheme<Scheme>& other)
    //         : base_class(flux_definition_t())
    //         , m_scheme(other.scheme())
    //         , m_scalar(other.scalar())
    //     {
    //         build_scheme_definition();
    //     }

    //     Scalar_x_FluxBasedScheme(Scalar_x_FluxBasedScheme<Scheme>&& other) noexcept
    //         : base_class(flux_definition_t())
    //         , m_scheme(other.scheme())
    //         , m_scalar(other.scalar())
    //     {
    //         build_scheme_definition();
    //     }

    //     Scalar_x_FluxBasedScheme& operator=(const Scalar_x_FluxBasedScheme& other)
    //     {
    //         if (this != &other)
    //         {
    //             this->m_scheme = other.m_scheme;
    //             this->m_scalar = other.m_scalar;
    //             this->build_scheme_definition();
    //         }
    //         return *this;
    //     }

    //     Scalar_x_FluxBasedScheme& operator=(Scalar_x_FluxBasedScheme&& other)
    //     {
    //         if (this != &other)
    //         {
    //             this->m_scheme = other.m_scheme;
    //             this->m_scalar = other.m_scalar;
    //             this->build_scheme_definition();
    //         }
    //         return *this;
    //     }

    //     auto scalar() const
    //     {
    //         return m_scalar;
    //     }

    //     auto& scheme()
    //     {
    //         return m_scheme;
    //     }

    //     auto& scheme() const
    //     {
    //         return m_scheme;
    //     }

    //     void build_scheme_definition()
    //     {
    //         static_for<0, dim>::apply(
    //             [&](auto integral_constant_d)
    //             {
    //                 static constexpr int d = decltype(integral_constant_d)::value;

    //                 this->flux_definition()[d] = m_scheme.flux_definition()[d];
    //                 if (m_scalar != 1)
    //                 {
    //                     // Multiply the flux function by the scalar
    //                     if constexpr (cfg_t::scheme_type == SchemeType::LinearHomogeneous)
    //                     {
    //                         this->flux_definition()[d].flux_function = [&](auto h)
    //                         {
    //                             return m_scalar * m_scheme.flux_definition()[d].flux_function(h);
    //                         };
    //                     }
    //                     else if constexpr (cfg_t::scheme_type == SchemeType::LinearHeterogeneous)
    //                     {
    //                         this->flux_definition()[d].flux_function = [&](auto& cells)
    //                         {
    //                             return m_scalar * m_scheme.flux_definition()[d].flux_function(cells);
    //                         };
    //                     }
    //                     else // SchemeType::NonLinear
    //                     {
    //                         this->flux_definition()[d].flux_function = [&](auto& cells, auto& field)
    //                         {
    //                             return m_scalar * m_scheme.flux_definition()[d].flux_function(cells, field);
    //                         };
    //                     }
    //                 }
    //             });

    //         this->dirichlet_config() = m_scheme.dirichlet_config();
    //         this->neumann_config()   = m_scheme.neumann_config();

    //         this->is_symmetric(m_scheme.is_symmetric());
    //         this->is_spd(m_scheme.is_spd() && m_scalar != 0);

    //         this->set_name(std::to_string(m_scalar) + " * " + m_scheme.name());
    //     }
    // };

    template <class cfg, class bdry_cfg>
    auto operator*(double scalar, const FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        // return Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>(scheme, scalar);
        static constexpr std::size_t dim = cfg::dim;

        FluxBasedScheme<cfg, bdry_cfg> multiplied_scheme(scheme); // copy

        static_for<0, dim>::apply(
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                multiplied_scheme.flux_definition()[d] = scheme.flux_definition()[d];
                if (scalar != 1)
                {
                    // Multiply the flux function by the scalar
                    if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
                    {
                        multiplied_scheme.flux_definition()[d].flux_function = [=](auto h)
                        {
                            return scalar * scheme.flux_definition()[d].flux_function(h);
                        };
                    }
                    else if constexpr (cfg::scheme_type == SchemeType::LinearHeterogeneous)
                    {
                        multiplied_scheme.flux_definition()[d].flux_function = [=](auto& cells)
                        {
                            return scalar * scheme.flux_definition()[d].flux_function(cells);
                        };
                    }
                    else // SchemeType::NonLinear
                    {
                        multiplied_scheme.flux_definition()[d].flux_function = [=](auto& cells, auto& field)
                        {
                            return scalar * scheme.flux_definition()[d].flux_function(cells, field);
                        };
                    }
                }
            });

        multiplied_scheme.is_spd(scheme.is_spd() && scalar != 0);
        multiplied_scheme.set_name(std::to_string(scalar) + " * " + scheme.name());
        return multiplied_scheme;
    }

    // template <class cfg, class bdry_cfg>
    // auto operator*(double scalar, FluxBasedScheme<cfg, bdry_cfg>& scheme)
    // {
    //     return Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>(scheme, scalar);
    // }

    // template <class cfg, class bdry_cfg>
    // auto operator*(double scalar, FluxBasedScheme<cfg, bdry_cfg>&& scheme)
    // {
    //     return Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>(std::forward<FluxBasedScheme<cfg, bdry_cfg>>(scheme), scalar);
    // }

    // template <class cfg, class bdry_cfg>
    // auto operator*(double scalar, Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>& scalar_x_scheme)
    // {
    //     return Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>(scalar_x_scheme.scheme(), scalar * scalar_x_scheme.scalar());
    // }

    // template <class cfg, class bdry_cfg>
    // auto operator*(double scalar, Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>&& scalar_x_scheme)
    // {
    //     return Scalar_x_FluxBasedScheme<FluxBasedScheme<cfg, bdry_cfg>>(scalar_x_scheme.scheme(), scalar * scalar_x_scheme.scalar());
    // }

    template <class cfg, class bdry_cfg>
    auto operator-(const FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return (-1) * scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(FluxBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return (-1) * scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(FluxBasedScheme<cfg, bdry_cfg>&& scheme)
    {
        return (-1) * scheme;
    }

} // end namespace samurai
