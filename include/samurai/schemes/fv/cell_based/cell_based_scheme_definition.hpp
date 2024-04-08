#pragma once
#include "../utils.hpp"
#include "local_field.hpp"
#include <functional>

namespace samurai
{
    template <SchemeType scheme_type_,
              std::size_t output_field_size_,
              std::size_t neighbourhood_width_,
              std::size_t scheme_stencil_size_,
              std::size_t center_index_,
              std::size_t contiguous_indices_start_,
              std::size_t contiguous_indices_size_,
              class InputField_>
    struct CellBasedSchemeConfig
    {
        static constexpr SchemeType scheme_type               = scheme_type_;
        static constexpr std::size_t output_field_size        = output_field_size_;
        static constexpr std::size_t neighbourhood_width      = neighbourhood_width_;
        static constexpr std::size_t scheme_stencil_size      = scheme_stencil_size_;
        static constexpr std::size_t center_index             = center_index_;
        static constexpr std::size_t contiguous_indices_start = contiguous_indices_start_;
        static constexpr std::size_t contiguous_indices_size  = contiguous_indices_size_;
        using input_field_t                                   = std::decay_t<InputField_>;
    };

    template <SchemeType scheme_type, std::size_t output_field_size, std::size_t neighbourhood_width, class InputField>
    using StarStencilSchemeConfig = CellBasedSchemeConfig<scheme_type,
                                                          output_field_size,
                                                          neighbourhood_width,
                                                          // ---- Stencil size
                                                          // Cell-centered Finite Volume scheme:
                                                          // center + 'neighbourhood_width' neighbours in each Cartesian direction (2*dim
                                                          // directions) --> 1+2=3 in 1D
                                                          //                 1+4=5 in 2D
                                                          1 + 2 * InputField::dim * neighbourhood_width,
                                                          // ---- Index of the stencil center
                                                          // (as defined in star_stencil())
                                                          neighbourhood_width,
                                                          // ---- Start index and size of contiguous cell indices
                                                          // (as defined in star_stencil())
                                                          0,
                                                          1 + 2 * neighbourhood_width,
                                                          // ---- Input field
                                                          InputField>;

    template <SchemeType scheme_type, std::size_t output_field_size, class InputField>
    using LocalCellSchemeConfig = StarStencilSchemeConfig<scheme_type, output_field_size, 0, InputField>;

    template <class cfg>
    struct CellBasedSchemeDefinitionBase
    {
        static constexpr std::size_t dim = cfg::input_field_t::dim;
        using scheme_stencil_t           = Stencil<cfg::scheme_stencil_size, dim>;
        scheme_stencil_t stencil;

        CellBasedSchemeDefinitionBase()
        {
            if constexpr (cfg::scheme_stencil_size == 1 + 2 * dim * cfg::neighbourhood_width && cfg::neighbourhood_width <= 2)
            {
                stencil = samurai::star_stencil<dim, cfg::neighbourhood_width>();
            }
        }
    };

    /**
     * @class CellBasedSchemeDefinition defines how to compute the scheme.
     * This struct inherits from @class CellBasedSchemeDefinitionBase and is specialized for all scheme types (see below).
     */
    template <class cfg, class enable = void>
    struct CellBasedSchemeDefinition
    {
    };

    /**
     * Specialization of @class CellBasedSchemeDefinition.
     * Defines how to compute a NON-LINEAR cell-based scheme.
     */
    template <class cfg>
    struct CellBasedSchemeDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>> : CellBasedSchemeDefinitionBase<cfg>
    {
        using field_t                           = typename cfg::input_field_t;
        using field_value_type                  = typename field_t::value_type;
        using cell_t                            = typename field_t::cell_t;
        static constexpr std::size_t field_size = field_t::size;

        using stencil_cells_t = CollapsArray<cell_t, cfg::scheme_stencil_size>;

        using scheme_value_t = CollapsArray<field_value_type, cfg::output_field_size>;
        using scheme_func    = std::function<scheme_value_t(stencil_cells_t&, const field_t&)>;

        using jac_coeffs_t         = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        using jac_stencil_coeffs_t = xt::xtensor_fixed<jac_coeffs_t, xt::xshape<cfg::scheme_stencil_size>>;
        using jacobian_func        = std::function<jac_stencil_coeffs_t(stencil_cells_t&, const field_t&)>;

        // Specific to implicit local schemes (unused otherwise)
        using local_field_t     = LocalField<field_t>;
        using local_scheme_func = std::function<scheme_value_t(stencil_cells_t&, const local_field_t&)>; // same as 'scheme_func', but with
                                                                                                         // 'local_field_t' instead of
                                                                                                         // 'field_t'
        using local_jacobian_func = std::function<jac_stencil_coeffs_t(stencil_cells_t&, const local_field_t&)>; // same as 'jacobian_func',
                                                                                                                 // but with 'local_field_t'
                                                                                                                 // instead of 'field_t'

        scheme_func scheme_function     = nullptr;
        jacobian_func jacobian_function = nullptr;

        // Specific to implicit local schemes (unused otherwise)
        local_scheme_func local_scheme_function     = nullptr;
        local_jacobian_func local_jacobian_function = nullptr;

        ~CellBasedSchemeDefinition()
        {
            scheme_function   = nullptr;
            jacobian_function = nullptr;
        }
    };

    /**
     * Specialization of @class CellBasedSchemeDefinition.
     * Defines how to compute a LINEAR and HETEROGENEOUS cell-based scheme.
     */
    template <class cfg>
    struct CellBasedSchemeDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHeterogeneous>>
        : CellBasedSchemeDefinitionBase<cfg>
    {
        using field_t                           = typename cfg::input_field_t;
        using field_value_type                  = typename field_t::value_type;
        using cell_t                            = typename field_t::cell_t;
        static constexpr std::size_t field_size = field_t::size;

        using stencil_cells_t       = std::array<cell_t, cfg::scheme_stencil_size>;
        using local_matrix_t        = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        using stencil_coeffs_t      = xt::xtensor_fixed<local_matrix_t, xt::xshape<cfg::stencil_size>>;
        using get_coefficients_func = std::function<stencil_coeffs_t(stencil_cells_t&)>;

        get_coefficients_func get_coefficients_function = nullptr;

        ~CellBasedSchemeDefinition()
        {
            get_coefficients_function = nullptr;
        }
    };

    /**
     * Specialization of @class CellBasedSchemeDefinition.
     * Defines how to compute a LINEAR and HOMOGENEOUS cell-based scheme.
     */
    template <class cfg>
    struct CellBasedSchemeDefinition<cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : CellBasedSchemeDefinitionBase<cfg>
    {
        using field_t                           = typename cfg::input_field_t;
        using field_value_type                  = typename field_t::value_type;
        static constexpr std::size_t field_size = field_t::size;

        using local_matrix_t        = CollapsMatrix<field_value_type, cfg::output_field_size, field_size>;
        using stencil_coeffs_t      = xt::xtensor_fixed<local_matrix_t, xt::xshape<cfg::scheme_stencil_size>>;
        using get_coefficients_func = std::function<stencil_coeffs_t(double)>;

        get_coefficients_func get_coefficients_function = nullptr;

        ~CellBasedSchemeDefinition()
        {
            get_coefficients_function = nullptr;
        }
    };

    template <class cfg>
    using SchemeValue = typename CellBasedSchemeDefinition<cfg>::scheme_value_t;

    template <class cfg>
    using JacobianMatrix = typename CellBasedSchemeDefinition<cfg>::jac_coeffs_t;

    template <class cfg>
    using StencilCoeffs = typename CellBasedSchemeDefinition<cfg>::stencil_coeffs_t;

} // end namespace samurai
