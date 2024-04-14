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
     *  @tparam dim_ The dimension of the cell.
     *  @tparam TInterval The type of the interval.
     */
    template <std::size_t dim_, class TInterval>
    struct Cell
    {
        static constexpr std::size_t dim = dim_;
        using interval_t                 = TInterval;
        using value_t                    = typename interval_t::value_t;
        using index_t                    = typename interval_t::index_t;
        using indices_t                  = xt::xtensor_fixed<value_t, xt::xshape<dim>>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;

        Cell() = default;

        Cell(std::size_t level, const indices_t& indices, index_t index);
        Cell(std::size_t level, const value_t& i, const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> others, index_t index);

        coords_t corner() const;
        double corner(std::size_t i) const;

        /// The center of the cell.
        coords_t center() const;
        double center(std::size_t i) const;

        /// The center of the face in the requested Cartesian direction.
        template <class Vector>
        coords_t face_center(const Vector& direction) const;

        void to_stream(std::ostream& os) const;

        /// The level of the cell.
        std::size_t level = 0;

        /// The integer coordinates of the cell.
        indices_t indices;

        /// The index where the cell is in the data array.
        index_t index = 0;

        /// The length of the cell.
        double length = 0;
    };

    template <std::size_t dim_, class TInterval>
    inline Cell<dim_, TInterval>::Cell(std::size_t level_, const indices_t& indices_, index_t index_)
        : level(level_)
        , indices(indices_)
        , index(index_)
        , length(cell_length(level))
    {
    }

    template <std::size_t dim_, class TInterval>
    inline Cell<dim_, TInterval>::Cell(std::size_t level_,
                                       const value_t& i,
                                       const xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> others,
                                       index_t index_)
        : level(level_)
        , index(index_)
        , length(cell_length(level))
    {
        using namespace xt::placeholders;

        indices[0]                         = i;
        xt::view(indices, xt::range(1, _)) = others;
    }

    /**
     * The minimum corner of the cell.
     */
    template <std::size_t dim_, class TInterval>
    inline auto Cell<dim_, TInterval>::corner() const -> coords_t
    {
        return length * indices;
    }

    template <std::size_t dim_, class TInterval>
    inline double Cell<dim_, TInterval>::corner(std::size_t i) const
    {
        return length * indices[i];
    }

    /**
     * The minimum corner of the cell.
     */
    template <std::size_t dim_, class TInterval>
    inline auto Cell<dim_, TInterval>::center() const -> coords_t
    {
        return length * (indices + 0.5);
    }

    template <std::size_t dim_, class TInterval>
    inline double Cell<dim_, TInterval>::center(std::size_t i) const
    {
        return length * (indices[i] + 0.5);
    }

    template <std::size_t dim_, class TInterval>
    template <class Vector>
    inline auto Cell<dim_, TInterval>::face_center(const Vector& direction) const -> coords_t
    {
        assert(abs(xt::sum(direction)(0)) == 1); // We only want a Cartesian unit vector
        return center() + (length / 2) * direction;
    }

    template <std::size_t dim_, class TInterval>
    inline void Cell<dim_, TInterval>::to_stream(std::ostream& os) const
    {
        os << "Cell -> level: " << level << " indices: " << indices << " center: " << center() << " index: " << index;
    }

    template <std::size_t dim, class TInterval>
    inline std::ostream& operator<<(std::ostream& out, const Cell<dim, TInterval>& cell)
    {
        cell.to_stream(out);
        return out;
    }

    template <std::size_t dim, class TInterval>
    inline bool operator==(const Cell<dim, TInterval>& c1, const Cell<dim, TInterval>& c2)
    {
        return !(c1.level != c2.level || c1.indices != c2.indices || c1.index != c2.index || c1.length != c2.length);
    }

    template <std::size_t dim, class TInterval>
    inline bool operator!=(const Cell<dim, TInterval>& c1, const Cell<dim, TInterval>& c2)
    {
        return !(c1 == c2);
    }
} // namespace samurai
