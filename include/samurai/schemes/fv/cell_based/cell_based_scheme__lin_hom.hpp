#pragma once
#include "cell_based_scheme.hpp"

namespace samurai
{
    /**
     * Useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
     */
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

    template <SchemeType scheme_type, std::size_t output_field_size, std::size_t neighbourhood_width, class InputField_>
    using StarStencilFV = CellBasedSchemeConfig<scheme_type,
                                                output_field_size,
                                                neighbourhood_width,
                                                // ---- Stencil size
                                                // Cell-centered Finite Volume scheme:
                                                // center + 'neighbourhood_width' neighbours in each Cartesian direction (2*dim
                                                // directions) --> 1+2=3 in 1D
                                                //                 1+4=5 in 2D
                                                1 + 2 * InputField_::dim * neighbourhood_width,
                                                // ---- Index of the stencil center
                                                // (as defined in star_stencil())
                                                neighbourhood_width,
                                                // ---- Start index and size of contiguous cell indices
                                                // (as defined in star_stencil())
                                                0,
                                                1 + 2 * neighbourhood_width,
                                                // ---- Input field
                                                InputField_>;

    template <SchemeType scheme_type, std::size_t output_field_size, class InputField_>
    using OneCellStencilFV = StarStencilFV<scheme_type, output_field_size, 0, InputField_>;

    // template <SchemeType scheme_type, std::size_t output_field_size, class InputField_>
    // using EmptyStencilFV = CellBasedSchemeConfig<scheme_type,
    //                                              output_field_size,
    //                                              0, // Neighbourhood width
    //                                              0, // Stencil size
    //                                              0, // Index of the stencil center
    //                                              0, // Contiguous indices start
    //                                              0, // Contiguous indices start
    //                                              InputField_>;

    template <class cfg, class bdry_cfg>
    class CellBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::LinearHomogeneous>>
        : public FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<typename cfg::input_field_t, cfg::output_field_size, bdry_cfg>;
        using base_class::dim;
        using base_class::field_size;

      public:

        using cfg_t                                    = cfg;
        using bdry_cfg_t                               = bdry_cfg;
        using field_t                                  = typename cfg::input_field_t;
        using field_value_type                         = typename field_t::value_type; // double
        static constexpr std::size_t output_field_size = cfg::output_field_size;
        using local_matrix_t                           = typename detail::LocalMatrix<field_value_type,
                                                            output_field_size,
                                                            field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a
                                                                                                         // matrix otherwise
        using scheme_stencil_t      = Stencil<cfg::scheme_stencil_size, dim>;
        using get_coefficients_func = std::function<std::array<local_matrix_t, cfg::scheme_stencil_size>(double)>;

      private:

        scheme_stencil_t m_stencil;
        get_coefficients_func m_get_coefficients;

      public:

        explicit CellBasedScheme()
        {
        }

        auto& stencil() const
        {
            return m_stencil;
        }

        auto& stencil()
        {
            return m_stencil;
        }

        void set_stencil(const scheme_stencil_t& stencil)
        {
            m_stencil = stencil;
        }

        void set_stencil(scheme_stencil_t&& stencil)
        {
            m_stencil = stencil;
        }

        get_coefficients_func& coefficients_func() const
        {
            return m_get_coefficients;
        }

        get_coefficients_func& coefficients_func()
        {
            return m_get_coefficients;
        }

        void set_coefficients_func(get_coefficients_func get_coefficients)
        {
            m_get_coefficients = get_coefficients;
        }

        auto coefficients(double h) const
        {
            return m_get_coefficients(h);
        }
    };

} // end namespace samurai
