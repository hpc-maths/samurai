#pragma once

#include <type_traits>
#include "../field.hpp"
#include "pred_and_proj.hpp"
#include "harten_multi.hpp"

namespace mure
{
    template<class MRConfig>
    inline void adapt(std::vector<Field<MRConfig>> &fields, double eps)
    {
        auto mesh = fields[0].mesh();
        std::size_t max_level = mesh.max_level();
        std::size_t min_level = mesh.min_level();

        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            for (std::size_t ifield=0; ifield<fields.size(); ++ifield)
            {
                mure::mr_projection(fields[ifield]);
                mure::mr_prediction(fields[ifield]);
            }
            mure::harten_multi(fields, eps, i);
        }
    }
}