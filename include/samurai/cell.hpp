#pragma once

#include <array>

namespace samurai
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
        static constexpr std::size_t dim = dim_;
        using coord_index_t = TCoord_index;

        Cell() = default;
        Cell(const Cell&) = default;
        Cell& operator=(const Cell&) = default;

        Cell(Cell&&) = default;
        Cell& operator=(Cell&&) = default;

        template <class T>
        Cell(std::size_t level, const T& indices, std::size_t index);

        xt::xtensor_fixed<double, xt::xshape<dim>> corner() const;
        double corner(std::size_t i) const;

        /// The integer coordinates of the cell.
        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> indices;

        /// The center of the cell.
        xt::xtensor_fixed<double, xt::xshape<dim>> center() const;
        double center(std::size_t i) const;

        /// The level of the cell.
        std::size_t level;

        /// The index where the cell is in the data array.
        std::size_t index;

        /// The length of the cell.
        double length;

        void to_stream(std::ostream& os) const;
    };

    template<class TCoord_index, std::size_t dim_>
    template <class T>
    inline Cell<TCoord_index, dim_>::Cell(std::size_t level, const T& indices, std::size_t index)
    : level(level), indices(indices), index(index)
    {
        length = 1./(1 << level);
        // center = length*(indices + 0.5);
    }

    /**
     * The minimum corner of the cell.
     */
    template<class TCoord_index, std::size_t dim_>
    inline auto Cell<TCoord_index, dim_>::corner() const -> xt::xtensor_fixed<double, xt::xshape<dim>>
    {
        return length*indices;
    }

    template<class TCoord_index, std::size_t dim_>
    inline double Cell<TCoord_index, dim_>::corner(std::size_t i) const
    {
        return length*indices[i];
    }

    /**
     * The minimum corner of the cell.
     */
    template<class TCoord_index, std::size_t dim_>
    inline auto Cell<TCoord_index, dim_>::center() const -> xt::xtensor_fixed<double, xt::xshape<dim>>
    {
        return length*(indices + 0.5);
    }

    template<class TCoord_index, std::size_t dim_>
    inline double Cell<TCoord_index, dim_>::center(std::size_t i) const
    {
        return length*(indices[i] + 0.5);
    }

    template<class TCoord_index, std::size_t dim_>
    inline void Cell<TCoord_index, dim_>::to_stream(std::ostream& os) const
    {
        os << "Cell -> level: " << level << " indices: " << indices << " center: " << center << " index: " << index;
    }

    template<class TCoord_index, std::size_t dim>
    inline std::ostream &operator<<(std::ostream &out, const Cell<TCoord_index, dim>& cell)
    {
        cell.to_stream(out);
        return out;
    }
}