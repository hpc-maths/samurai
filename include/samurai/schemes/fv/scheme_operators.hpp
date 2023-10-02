#pragma once
#include "cell_based/algebraic_operators.hpp"
#include "flux_based/algebraic_operators.hpp"

namespace samurai
{
    /**
     * Addition of a flux-based scheme and a cell-based scheme.
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

    // FluxBasedScheme_Sum_CellBasedScheme + flux_scheme2
    template <typename CellScheme, typename FluxScheme, typename FluxScheme2, std::enable_if_t<is_FluxBasedScheme_v<FluxScheme2>, bool> = true>
    auto operator+(const FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>& sum_scheme, const FluxScheme2& flux_scheme2)
    {
        return (sum_scheme.flux_scheme() + flux_scheme2) + sum_scheme.cell_scheme();
    }

    // FluxBasedScheme_Sum_CellBasedScheme + cell_scheme2
    template <typename CellScheme, typename FluxScheme, typename CellScheme2, std::enable_if_t<is_CellBasedScheme_v<CellScheme2>, bool> = true>
    auto operator+(const FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>& sum_scheme, const CellScheme2& cell_scheme2)
    {
        return (sum_scheme.cell_scheme() + cell_scheme2) + sum_scheme.flux_scheme();
    }

    // FluxBasedScheme_Sum_CellBasedScheme - cell_scheme2
    template <typename CellScheme, typename FluxScheme, typename CellScheme2, std::enable_if_t<is_CellBasedScheme_v<CellScheme2>, bool> = true>
    auto operator-(const FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>& sum_scheme, const CellScheme2& cell_scheme2)
    {
        return (sum_scheme.cell_scheme() - cell_scheme2) + sum_scheme.flux_scheme();
    }

    // Sum_CellBasedScheme + flux_scheme
    // template <class cfg1, class bdry_cfg1, class cfg2, class bdry_cfg2, class cfg3, class bdry_cfg3>
    // auto
    // operator+(const Sum_CellBasedScheme<cfg1, bdry_cfg1, cfg2, bdry_cfg2>& sum_scheme, const FluxBasedScheme<cfg3, bdry_cfg3>&
    // flux_scheme)
    // {
    //     return (sum_scheme.cell_scheme() + cell_scheme2) + sum_scheme.flux_scheme();
    // }

} // end namespace samurai
