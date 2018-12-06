#pragma once

namespace mure
{
    template<
            unsigned _dim,
            unsigned _max_stencil_with     = 1,  // nb bits par axis in the coarsest level
            unsigned _graduation_width     = 1,  //
            unsigned _max_refinement_level = 16  //
            >
    struct MRConfig {
        // dimensions, storage
        static constexpr int      dim                         = _dim;
        static constexpr unsigned max_refinement_level        = _max_refinement_level; ///< nb bits par axis in the finest level
        static constexpr bool     need_pred_from_proj         = false;                 ///< if it's needed to systematically add ghosts in a less refined level (+/- prediction_stencil_width) for each leaf

        // base types
        using                     coord_index_t               = int;                  ///< integer for coordinates
        using                     coord_t                     = double;               ///< floating point type (notably for flt_... coordinates)
        using                     index_t                     = std::size_t;          ///<

        // stencils
        static constexpr unsigned graduation_width            = _graduation_width;    ///< for graded tree
        static constexpr unsigned max_stencil_width           = _max_stencil_with;    ///< used notably to define how much ghost nodes we need
        static constexpr unsigned default_s_for_prediction    = 1;                    ///< default interpolation width used for fields when mesh is changed
    };
}