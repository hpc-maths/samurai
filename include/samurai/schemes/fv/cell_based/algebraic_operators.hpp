#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    /**
     * Multiplication by a scalar value of a cell-based scheme
     */
    template <class cfg, class bdry_cfg>
    auto operator*(double scalar, const CellBasedScheme<cfg, bdry_cfg>& scheme)
    {
        CellBasedScheme<cfg, bdry_cfg> multiplied_scheme(scheme); // copy

        multiplied_scheme.set_name(std::to_string(scalar) + " * " + scheme.name());
        if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
        {
            multiplied_scheme.coefficients_func() = [=](double h)
            {
                return scalar * scheme.coefficients(h);
            };
        }
        else // SchemeType::NonLinear
        {
            using stencil_cells_t = typename CellBasedScheme<cfg, bdry_cfg>::stencil_cells_t;
            using field_t         = typename CellBasedScheme<cfg, bdry_cfg>::field_t;

            multiplied_scheme.scheme_function() = [=](stencil_cells_t& cells, const field_t& field)
            {
                return scalar * scheme.scheme_function()(cells, field);
            };

            multiplied_scheme.local_scheme_function() = nullptr;
            if (scheme.local_scheme_function())
            {
                multiplied_scheme.local_scheme_function() = [=](stencil_cells_t& cells, const auto& field)
                {
                    return scalar * scheme.local_scheme_function()(cells, field);
                };
            }

            multiplied_scheme.jacobian_function() = nullptr;
            if (scheme.jacobian_function())
            {
                multiplied_scheme.jacobian_function() = [=](stencil_cells_t& cells, const field_t& field)
                {
                    return scalar * scheme.jacobian_function()(cells, field);
                };
            }

            multiplied_scheme.local_jacobian_function() = nullptr;
            if (scheme.local_jacobian_function())
            {
                multiplied_scheme.local_jacobian_function() = [=](stencil_cells_t& cells, const auto& field)
                {
                    return scalar * scheme.local_jacobian_function()(cells, field);
                };
            }
        }

        std::ostringstream name;
        if (scalar == static_cast<int>(scalar))
        {
            name << static_cast<int>(scalar) << " * " << scheme.name();
        }
        else
        {
            name << std::setprecision(1) << std::scientific << scalar << " * " << scheme.name();
        }
        multiplied_scheme.set_name(name.str());
        return multiplied_scheme;
    }

    /**
     * Binary '+' operator if same config
     */
    template <class cfg, class bdry_cfg>
    auto operator+(const CellBasedScheme<cfg, bdry_cfg>& scheme1, const CellBasedScheme<cfg, bdry_cfg>& scheme2)
    {
        CellBasedScheme<cfg, bdry_cfg> addition_scheme(scheme1); // copy

        addition_scheme.set_name(scheme1.name() + " + " + scheme2.name());
        if constexpr (cfg::scheme_type == SchemeType::LinearHomogeneous)
        {
            addition_scheme.coefficients_func() = [=](double h)
            {
                return scheme1.coefficients(h) + scheme2.coefficients(h);
            };
        }
        else // SchemeType::NonLinear
        {
            using stencil_cells_t = typename CellBasedScheme<cfg, bdry_cfg>::stencil_cells_t;
            using field_t         = typename CellBasedScheme<cfg, bdry_cfg>::field_t;

            addition_scheme.scheme_function() = nullptr;
            if (scheme1.scheme_function() && scheme2.scheme_function())
            {
                addition_scheme.scheme_function() = [=](stencil_cells_t& cells, const field_t& field)
                {
                    return scheme1.scheme_function()(cells, field) + scheme2.scheme_function()(cells, field);
                };
            }

            addition_scheme.local_scheme_function() = nullptr;
            if (scheme1.local_scheme_function() && scheme2.local_scheme_function())
            {
                addition_scheme.local_scheme_function() = [=](stencil_cells_t& cells, const field_t& field)
                {
                    return scheme1.local_scheme_function()(cells, field) + scheme2.local_scheme_function()(cells, field);
                };
            }

            addition_scheme.jacobian_function() = nullptr;
            if (scheme1.jacobian_function() && scheme2.jacobian_function())
            {
                addition_scheme.jacobian_function() = [=](stencil_cells_t& cells, const field_t& field)
                {
                    return scheme1.jacobian_function()(cells, field) + scheme2.jacobian_function()(cells, field);
                };
            }

            addition_scheme.local_jacobian_function() = nullptr;
            if (scheme1.local_jacobian_function() && scheme2.local_jacobian_function())
            {
                addition_scheme.local_jacobian_function() = [=](stencil_cells_t& cells, const field_t& field)
                {
                    return scheme1.local_jacobian_function()(cells, field) + scheme2.local_jacobian_function()(cells, field);
                };
            }
        }
        return addition_scheme;
    }

    /**
     * Binary '+' operator if different SchemeType (NonLinear and Linear) but same stencil of size 1
     */
    template <class lin_cfg,
              class nonlin_cfg,
              class bdry_cfg,
              std::enable_if_t<nonlin_cfg::scheme_type == SchemeType::NonLinear && lin_cfg::scheme_type == SchemeType::LinearHomogeneous
                                   && lin_cfg::stencil_size == nonlin_cfg::stencil_size && lin_cfg::stencil_size == 1,
                               bool> = true>
    auto operator+(const CellBasedScheme<lin_cfg, bdry_cfg>& lin_scheme, const CellBasedScheme<nonlin_cfg, bdry_cfg>& nonlin_scheme)
    {
        using stencil_cells_t = typename CellBasedScheme<nonlin_cfg, bdry_cfg>::stencil_cells_t;
        using field_t         = typename CellBasedScheme<nonlin_cfg, bdry_cfg>::field_t;

        CellBasedScheme<nonlin_cfg, bdry_cfg> addition_scheme(nonlin_scheme); // copy

        addition_scheme.set_name(lin_scheme.name() + " + " + nonlin_scheme.name());
        if constexpr (lin_cfg::scheme_type == SchemeType::LinearHomogeneous)
        {
            addition_scheme.scheme_function() = [=](stencil_cells_t& cell, const field_t& field)
            {
                auto value = nonlin_scheme.scheme_function()(cell, field);

                auto h      = cell.length;
                auto coeffs = lin_scheme.coefficients(h);
                value += mat_vec<field_t::is_soa>(coeffs[0], field[cell]);
                return value;
            };

            addition_scheme.local_scheme_function() = nullptr;
            if (nonlin_scheme.local_scheme_function())
            {
                addition_scheme.local_scheme_function() = [=](stencil_cells_t& cell, const auto& field)
                {
                    auto value = nonlin_scheme.local_scheme_function()(cell, field);

                    auto h      = cell.length;
                    auto coeffs = lin_scheme.coefficients(h);
                    value += mat_vec<field_t::is_soa>(coeffs[0], field[cell]);
                    return value;
                };
            }

            addition_scheme.jacobian_function() = nullptr;
            if (nonlin_scheme.jacobian_function())
            {
                addition_scheme.jacobian_function() = [=](stencil_cells_t& cell, const field_t& field)
                {
                    auto jac = nonlin_scheme.jacobian_function()(cell, field);

                    auto h = cell.length;
                    jac += lin_scheme.coefficients(h);
                    return jac;
                };
            }

            addition_scheme.local_jacobian_function() = nullptr;
            if (nonlin_scheme.local_jacobian_function())
            {
                addition_scheme.local_jacobian_function() = [=](stencil_cells_t& cell, const auto& field)
                {
                    auto jac = nonlin_scheme.local_jacobian_function()(cell, field);

                    auto h = cell.length;
                    jac += lin_scheme.coefficients(h);
                    return jac;
                };
            }
        }
        return addition_scheme;
    }

    /**
     * Unary '-' operator
     */
    template <class cfg, class bdry_cfg>
    auto operator-(const CellBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return (-1) * scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(CellBasedScheme<cfg, bdry_cfg>& scheme)
    {
        return (-1) * scheme;
    }

    template <class cfg, class bdry_cfg>
    auto operator-(CellBasedScheme<cfg, bdry_cfg>&& scheme)
    {
        return (-1) * scheme;
    }

    /**
     * Binary '-' operator if same config
     */
    template <class cfg, class bdry_cfg>
    auto operator-(const CellBasedScheme<cfg, bdry_cfg>& scheme1, const CellBasedScheme<cfg, bdry_cfg>& scheme2)
    {
        return scheme1 + ((-1) * scheme2);
    }

    /**
     * Binary '-' operator if different SchemeType (NonLinear and LinearHomogeneous) but same stencil
     */
    template <class cfg1, class cfg2, class bdry_cfg>
    auto operator-(const CellBasedScheme<cfg1, bdry_cfg>& scheme1, const CellBasedScheme<cfg2, bdry_cfg>& scheme2)
    {
        return scheme1 + ((-1) * scheme2);
    }

} // end namespace samurai
