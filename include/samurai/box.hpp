// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#if XTENSOR_VERSION_MINOR < 26
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#else
#include <xtensor/containers/xfixed.hpp>
#include <xtensor/io/xio.hpp>
#endif

namespace samurai
{

    ////////////////////
    // Box definition //
    ////////////////////

    /** @class Box
     *  @brief Define a box in multi dimensions.
     *
     *  A box is defined by its minimum and maximum corners.
     *
     *  @tparam value_t The type of the box corners.
     *  @tparam dim_ The dimension of the box
     */
    template <class value_t, std::size_t dim_>
    class Box
    {
      public:

        static constexpr std::size_t dim = dim_;
        using point_t                    = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

        Box() = default;
        Box(const point_t& min_corner, const point_t& max_corner);

        const point_t& min_corner() const;
        point_t& min_corner();

        const point_t& max_corner() const;
        point_t& max_corner();

        auto length() const;
        auto min_length() const;
        bool is_valid() const;

        Box& operator*=(value_t v);

      private:

        point_t m_min_corner{0};
        point_t m_max_corner{0};
    };

    ////////////////////////
    // Box implementation //
    ////////////////////////

    /**
     * Construction of a bounded box.
     *
     * @param min_corner The vertex with the minimum coordinates
     * @param max_corner The vertex with the maximum coordinates
     */
    template <class value_t, std::size_t dim_>
    inline Box<value_t, dim_>::Box(const point_t& min_corner, const point_t& max_corner)
        : m_min_corner{min_corner}
        , m_max_corner{max_corner}
    {
    }

    /**
     * Return the min corner of the box.
     */
    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::min_corner() const -> const point_t&
    {
        return m_min_corner;
    }

    /**
     * Return the min corner of the box.
     */
    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::min_corner() -> point_t&
    {
        return m_min_corner;
    }

    /**
     * Return the max corner of the box.
     */
    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::max_corner() const -> const point_t&
    {
        return m_max_corner;
    }

    /**
     * Return the max corner of the box.
     */
    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::max_corner() -> point_t&
    {
        return m_max_corner;
    }

    /**
     * Return the length of the box.
     */
    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::length() const
    {
        return (m_max_corner - m_min_corner);
    }

    /**
     * Return the minimum length of the box.
     */
    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::min_length() const
    {
        return xt::amin(length())[0];
    }

    /**
     * Check if the box is valid.
     */
    template <class value_t, std::size_t dim_>
    inline bool Box<value_t, dim_>::is_valid() const
    {
        return xt::all(m_min_corner < m_max_corner);
    }

    template <class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::operator*=(value_t v) -> Box&
    {
        m_min_corner *= v;
        m_max_corner *= v;
        return *this;
    }

    template <class value_t, std::size_t dim_>
    inline auto operator*(const Box<value_t, dim_>& box, value_t v)
    {
        Box<value_t, dim_> that(box);
        return that *= v;
    }

    template <class value_t, std::size_t dim_>
    inline auto operator*(value_t v, const Box<value_t, dim_>& box)
    {
        Box<value_t, dim_> that(box);
        return that *= v;
    }

    template <class value_t, std::size_t dim>
    inline std::ostream& operator<<(std::ostream& out, const Box<value_t, dim>& box)
    {
        out << "Box(" << box.min_corner() << ", " << box.max_corner() << ")";
        return out;
    }

    template <class value_t, std::size_t dim>
    Box<value_t, dim> approximate_box(const Box<value_t, dim>& box, double tol, double& subdivision_length)
    {
        bool given_subdivision_length = subdivision_length > 0;
        if (!given_subdivision_length)
        {
            subdivision_length = box.min_length(); // / 2;
        }

        auto approx_length = xt::eval(xt::ceil(box.length() / subdivision_length) * subdivision_length);

        if (!given_subdivision_length)
        {
            while (xt::any(xt::abs(approx_length - box.length()) > tol * box.length()))
            {
                subdivision_length /= 2;
                approx_length = xt::eval(xt::ceil(box.length() / subdivision_length) * subdivision_length);
            }
        }

        Box<value_t, dim> approx_box;
        approx_box.min_corner() = box.min_corner();
        approx_box.max_corner() = box.min_corner() + approx_length;
        return approx_box;
    }
} // namespace samurai
