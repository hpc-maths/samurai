#pragma once
#include "FV_scheme.hpp"

namespace samurai
{
    /**
     * Useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
     */
    template <std::size_t output_field_size_,
              std::size_t neighbourhood_width_,
              std::size_t scheme_stencil_size_,
              std::size_t center_index_,
              std::size_t contiguous_indices_start_ = 0,
              std::size_t contiguous_indices_size_  = 0>
    struct CellBasedAssemblyConfig
    {
        static constexpr std::size_t output_field_size        = output_field_size_;
        static constexpr std::size_t neighbourhood_width      = neighbourhood_width_;
        static constexpr std::size_t scheme_stencil_size      = scheme_stencil_size_;
        static constexpr std::size_t center_index             = center_index_;
        static constexpr std::size_t contiguous_indices_start = contiguous_indices_start_;
        static constexpr std::size_t contiguous_indices_size  = contiguous_indices_size_;
    };

    template <std::size_t dim, std::size_t output_field_size, std::size_t neighbourhood_width = 1>
    using StarStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                  neighbourhood_width,
                                                  // ---- Stencil size
                                                  // Cell-centered Finite Volume scheme:
                                                  // center + 'neighbourhood_width' neighbours in each Cartesian direction (2*dim
                                                  // directions) --> 1+2=3 in 1D
                                                  //                 1+4=5 in 2D
                                                  1 + 2 * dim * neighbourhood_width,
                                                  // ---- Index of the stencil center
                                                  // (as defined in star_stencil())
                                                  neighbourhood_width,
                                                  // ---- Start index and size of contiguous cell indices
                                                  // (as defined in star_stencil())
                                                  0,
                                                  1 + 2 * neighbourhood_width>;

    template <std::size_t output_field_size>
    using OneCellStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                     0,  // Neighbourhood width
                                                     1,  // Stencil size
                                                     0>; // Index of the stencil center

    template <std::size_t output_field_size>
    using EmptyStencilFV = CellBasedAssemblyConfig<output_field_size,
                                                   0,  // Neighbourhood width
                                                   0,  // Stencil size
                                                   0>; // Index of the stencil center

    template <class DerivedScheme, class cfg, class bdry_cfg, class Field>
    class CellBasedScheme : public FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>
    {
      protected:

        using base_class = FVScheme<DerivedScheme, Field, cfg::output_field_size, bdry_cfg>;
        using base_class::dim;
        using base_class::field_size;

      public:

        using cfg_t                                    = cfg;
        using bdry_cfg_t                               = bdry_cfg;
        using field_t                                  = Field;
        using field_value_type                         = typename Field::value_type; // double
        static constexpr std::size_t output_field_size = cfg::output_field_size;
        using local_matrix_t                           = typename detail::LocalMatrix<field_value_type,
                                                            output_field_size,
                                                            field_size>::Type; // 'double' if field_size = 1, 'xtensor' representing a
                                                                                                         // matrix otherwise

        explicit CellBasedScheme()
        {
        }
    };

    template <class Scheme, typename = void>
    struct is_CellBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_CellBasedScheme<
        Scheme,
        std::enable_if_t<
            std::is_base_of_v<CellBasedScheme<Scheme, typename Scheme::cfg_t, typename Scheme::bdry_cfg_t, typename Scheme::field_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    inline constexpr bool is_CellBasedScheme_v = is_CellBasedScheme<Scheme>::value;

} // end namespace samurai
