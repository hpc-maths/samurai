// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

namespace samurai
{
    template <typename LevelType, std::enable_if_t<std::is_integral<LevelType>::value, bool> = true>
    inline double cell_length(double scaling_factor, LevelType level)
    {
        return scaling_factor / (1 << level);
    }

    /** @class Cell
     *  @brief Define a mesh cell in multi dimensions.
     *
     *  A cell is defined by its level, its integer coordinates,
     *  and its index in the data array.
     *
     *  @tparam dim_ The dimension of the cell.
     *  @tparam TInterval The type of the interval.
     */
    template <std::size_t dim_, class TInterval>
    struct Cell
    {
        static constexpr std::size_t dim = dim_;                                        ///< The dimension of the cell.
        using interval_t                 = TInterval;                                   ///< Type of interval.
        using value_t                    = typename interval_t::value_t;                ///< Type of value stored in interval.
        using index_t                    = typename interval_t::index_t;                ///< Type of index stored in interval.
        using indices_t                  = xt::xtensor_fixed<value_t, xt::xshape<dim>>; ///< Type of indices to access to field data.
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;  ///< Type of coordinates.

        Cell() = default;

        Cell(const coords_t& origin_point, double scaling_factor, std::size_t level, const indices_t& indices, index_t index);
        Cell(const coords_t& origin_point,
             double scaling_factor,
             std::size_t level,
             const value_t& i,
             const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> others,
             index_t index);

        // The minimum corner of the cell.
        coords_t corner() const;
        double corner(std::size_t i) const;

        // The center of the cell.
        coords_t center() const;
        double center(std::size_t i) const;

        // The center of the face in the requested Cartesian direction.
        template <class Vector>
        coords_t face_center(const Vector& direction) const;

        void to_stream(std::ostream& os) const;

        coords_t origin_point;

        /// The level of the cell.
        std::size_t level = 0;

        /// The integer coordinates of the cell.
        indices_t indices;

        /// The index where the cell is in the data array.
        index_t index = 0;

        /// The length of the cell.
        double length = 0;
    };

    /**
     * @brief Construct a new Cell<dim_, TInterval>::Cell object
     *
     * @param origin_point_
     * @param scaling_factor
     * @param level_
     * @param indices_
     * @param index_
     */
    template <std::size_t dim_, class TInterval>
    inline Cell<dim_, TInterval>::Cell(const coords_t& origin_point_,
                                       double scaling_factor,
                                       std::size_t level_,
                                       const indices_t& indices_,
                                       index_t index_)
        : origin_point(origin_point_)
        , level(level_)
        , indices(indices_)
        , index(index_)
        , length(cell_length(scaling_factor, level))
    {
    }

    /**
     * @brief Construct a new Cell<dim_, TInterval>::Cell object
     *
     * @param origin_point_
     * @param scaling_factor
     * @param level_
     * @param i
     * @param others
     * @param index_
     */
    template <std::size_t dim_, class TInterval>
    inline Cell<dim_, TInterval>::Cell(const coords_t& origin_point_,
                                       double scaling_factor,
                                       std::size_t level_,
                                       const value_t& i,
                                       const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> others,
                                       index_t index_)
        : origin_point(origin_point_)
        , level(level_)
        , index(index_)
        , length(cell_length(scaling_factor, level))
    {
        using namespace xt::placeholders;

        indices[0]                         = i;
        xt::view(indices, xt::range(1, _)) = others;
    }

    /**
     * @brief The minimum corner of the cell.
     */
    template <std::size_t dim_, class TInterval>
    inline auto Cell<dim_, TInterval>::corner() const -> coords_t
    {
        return origin_point + length * indices;
    }

    /**
     * @brief The ith coordinate of minimum corner of the cell.
     *
     * @param i Component number.
     */
    template <std::size_t dim_, class TInterval>
    inline double Cell<dim_, TInterval>::corner(std::size_t i) const
    {
        return origin_point[i] + length * indices[i];
    }

    /**
     * @brief The center of the cell.
     */
    template <std::size_t dim_, class TInterval>
    inline auto Cell<dim_, TInterval>::center() const -> coords_t
    {
        return origin_point + length * (indices + 0.5);
    }

    /**
     * @brief The ith coordinate of center of the cell.
     *
     * @param i Component number.
     */
    template <std::size_t dim_, class TInterval>
    inline double Cell<dim_, TInterval>::center(std::size_t i) const
    {
        return origin_point[i] + length * (indices[i] + 0.5);
    }

    /**
     * @brief The center of the face in the requested Cartesian direction.
     *
     * @tparam Vector   Type of direction, must be addable with a `coords_t`.
     * @param direction Cartesian unit vector of direction where we request center.
     */
    template <std::size_t dim_, class TInterval>
    template <class Vector>
    inline auto Cell<dim_, TInterval>::face_center(const Vector& direction) const -> coords_t
    {
        assert(abs(xt::sum(direction)(0)) == 1); // We only want a Cartesian unit vector
        return center() + (length / 2) * direction;
    }

    /**
     * @brief Insert formatted cell into an output stream.
     *
     * @param os output stream
     */
    template <std::size_t dim_, class TInterval>
    inline void Cell<dim_, TInterval>::to_stream(std::ostream& os) const
    {
        os << "Cell -> level: " << level << " indices: " << indices << " center: " << center() << " index: " << index;
    }

    /**
     * @brief Insert formatted cell into an output stream.
     */
    template <std::size_t dim, class TInterval>
    inline std::ostream& operator<<(std::ostream& out, const Cell<dim, TInterval>& cell)
    {
        cell.to_stream(out);
        return out;
    }

    /**
     * @brief Test equality between cells.
     */
    template <std::size_t dim, class TInterval>
    inline bool operator==(const Cell<dim, TInterval>& c1, const Cell<dim, TInterval>& c2)
    {
        return !(c1.level != c2.level || c1.indices != c2.indices || c1.index != c2.index || c1.length != c2.length);
    }

    /**
     * @brief Test inequality between cells.
     */
    template <std::size_t dim, class TInterval>
    inline bool operator!=(const Cell<dim, TInterval>& c1, const Cell<dim, TInterval>& c2)
    {
        return !(c1 == c2);
    }
} // namespace samurai
