#pragma once

#include <xtensor/xfixed.hpp>

namespace mure
{
    template<class TCoord_index, class TIndex, std::size_t dim_>
    struct Cell
    {
        using index_t = TIndex;
        using coord_index_t = TCoord_index;
        static constexpr auto dim = dim_;

        std::size_t level;
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> indices;
        index_t index;

        inline double length() const
        {
            return 1./(1 << level);
        }

        inline auto center() const
        {
            return xt::eval(length()*(indices + 0.5));
        }

        inline auto first_corner() const
        {
            return xt::eval(length()*indices);
        }
    };
}