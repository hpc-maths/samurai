#pragma once

#include <xtensor/xfixed.hpp>

namespace mure
{
    /** @class Cell
     *  @brief Define a mesh cell in multi dimensions.
     * 
     *  A cell is defined by its level, its integer coordinates,
     *  and its index in the data array.
     * 
     *  @tparam TCoord_index The type of the coordinates.
     *  @tparam dim_ The dimension of the cell.
     */
    template<class TCoord_index, std::size_t dim_>
    struct Cell
    {
        using coord_index_t = TCoord_index;
        static constexpr auto dim = dim_;

        /// The level of the cell.
        std::size_t level;
        /// The integer coordinates of the cell.
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> indices;
        /// The index where the cell is in the data array.
        std::size_t index;

        /// The length of the cell.
        inline double length() const {
            return 1./(1 << level);
        }

        /// The center of the cell.
        inline auto center() const
        {
            return xt::eval(length()*(indices + 0.5));
        }

        /// The minimum corner of the cell.
        inline auto first_corner() const
        {
            return xt::eval(length()*indices);
        }
    };
}