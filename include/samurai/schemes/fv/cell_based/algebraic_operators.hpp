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

            multiplied_scheme.scheme_function() = [=](stencil_cells_t& cells, field_t& field)
            {
                return scalar * scheme.scheme_function()(cells, field);
            };
        }
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

            addition_scheme.scheme_function() = [=](stencil_cells_t& cells, field_t& field)
            {
                return scheme1.scheme_function()(cells, field) + scheme2.scheme_function()(cells, field);
            };
        }
        return addition_scheme;
    }

    /**
     * Binary '+' operator if different configs
     */
    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    struct Sum_CellBasedSchemes
    {
        using Scheme1 = CellBasedScheme<cfg1, bdry_cfg1>;
        using Scheme2 = CellBasedScheme<cfg2, bdry_cfg2>;

        Scheme1 scheme1;
        Scheme2 scheme2;

        Sum_CellBasedSchemes(const Scheme1& s1, const Scheme2& s2)
            : scheme1(s1)
            , scheme2(s2)
        {
        }
    };

    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator+(const CellBasedScheme<cfg1, bdry_cfg1>& scheme1, const CellBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return Sum_CellBasedSchemes<cfg1, bdry_cfg1, cfg2, bdry_cfg2>(scheme1, scheme2);
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
    template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2>
    auto operator-(const CellBasedScheme<cfg1, bdry_cfg1>& scheme1, const CellBasedScheme<cfg2, bdry_cfg2>& scheme2)
    {
        return scheme1 + ((-1) * scheme2);
    }

} // end namespace samurai
