// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace samurai
{
    template <typename LevelType, std::enable_if_t<std::is_integral<LevelType>::value, bool> = true>
    inline double cell_length(LevelType level)
    {
        return 1. / (1 << level);
    }

    /** @class Cell
     *  @brief Define a mesh cell in multi dimensions.
     *
     *  A cell is defined by its level, its integer coordinates,
     *  and its index in the data array.
     *
     *  @tparam TCoord_index The type of the coordinates.
     *  @tparam dim_ The dimension of the cell.
     */
    template <class TCoord_index, std::size_t dim_>
    struct Cell
    {
        static constexpr std::size_t dim = dim_;
        using value_t                    = TCoord_index;

        Cell() = default;

        template <class T>
        Cell(std::size_t level, const T& indices, std::size_t index);

        xt::xtensor_fixed<double, xt::xshape<dim>> corner() const;
        double corner(std::size_t i) const;

        /// The center of the cell.
        xt::xtensor_fixed<double, xt::xshape<dim>> center() const;
        double center(std::size_t i) const;

        /// The center of the face in the requested Cartesian direction.
        template <class Vector>
        xt::xtensor_fixed<double, xt::xshape<dim>> face_center(const Vector& direction) const;

        void to_stream(std::ostream& os) const;

        /// The level of the cell.
        std::size_t level = 0;

        /// The integer coordinates of the cell.
        xt::xtensor_fixed<value_t, xt::xshape<dim>> indices;

        /// The index where the cell is in the data array.
        std::size_t index = 0;

        /// The length of the cell.
        double length = 0;
    };

    template <class TCoord_index, std::size_t dim_>
    template <class T>
    inline Cell<TCoord_index, dim_>::Cell(std::size_t level_, const T& indices_, std::size_t index_)
        : level(level_)
        , indices(indices_)
        , index(index_)
        , length(cell_length(level))
    {
    }

    /**
     * The minimum corner of the cell.
     */
    template <class TCoord_index, std::size_t dim_>
    inline auto Cell<TCoord_index, dim_>::corner() const -> xt::xtensor_fixed<double, xt::xshape<dim>>
    {
        return length * indices;
    }

    template <class TCoord_index, std::size_t dim_>
    inline double Cell<TCoord_index, dim_>::corner(std::size_t i) const
    {
        return length * indices[i];
    }

    /**
     * The minimum corner of the cell.
     */
    template <class TCoord_index, std::size_t dim_>
    inline auto Cell<TCoord_index, dim_>::center() const -> xt::xtensor_fixed<double, xt::xshape<dim>>
    {
        return length * (indices + 0.5);
    }

    template <class TCoord_index, std::size_t dim_>
    inline double Cell<TCoord_index, dim_>::center(std::size_t i) const
    {
        return length * (indices[i] + 0.5);
    }

    template <class TCoord_index, std::size_t dim_>
    template <class Vector>
    inline auto Cell<TCoord_index, dim_>::face_center(const Vector& direction) const -> xt::xtensor_fixed<double, xt::xshape<dim>>
    {
        assert(abs(xt::sum(direction)(0)) == 1); // We only want a Cartesian unit vector
        return center() + (length / 2) * direction;
    }

    template <class TCoord_index, std::size_t dim_>
    inline void Cell<TCoord_index, dim_>::to_stream(std::ostream& os) const
    {
        os << "Cell -> level: " << level << " indices: " << indices << " center: " << center() << " index: " << index;
    }

    template <class TCoord_index, std::size_t dim>
    inline std::ostream& operator<<(std::ostream& out, const Cell<TCoord_index, dim>& cell)
    {
        cell.to_stream(out);
        return out;
    }
} // namespace samurai