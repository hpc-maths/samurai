#pragma once

#include "../interval.hpp"

namespace mure
{
    template<std::size_t dim_,
             std::size_t max_stencil_with_ = 1, // nb bits par axis in the coarsest level
             std::size_t graduation_width_ = 1,
             std::size_t max_refinement_level_ = 20
             >
    struct MRConfig
    {
        // dimensions, storage
        static constexpr std::size_t dim = dim_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_; ///< nb bits par axis in the finest level
        static constexpr bool need_pred_from_proj = true; ///< if it's needed to systematically add ghosts in a less
                                                          ///< refined level (+/- prediction_stencil_width) for each
                                                          ///< leaf

        // base types
        using coord_index_t = int; ///< integer for coordinates
        using coord_t = double; ///< floating point type (notably for flt_... coordinates)
        using index_t = long long int; ///<
        using interval_t = Interval<coord_index_t, index_t>;

        // stencils
        static constexpr std::size_t graduation_width = graduation_width_; ///< for graded tree
        static constexpr std::size_t max_stencil_width = max_stencil_with_; ///< used notably to define how much ghost nodes
                                                                            ///< we need
        static constexpr std::size_t default_s_for_prediction = 1; ///< default interpolation width used for fields when mesh is
                                                                   ///< changed
    };
}
