#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace mure
{
    /** @class Box
     *  @brief Define a box in multi dimensions.
     * 
     *  A box is defined by its minimum and maximum corners.
     * 
     *  @tparam value_t The type of the box corners.
     *  @tparam dim_ The dimension of the box
     */
    template<class value_t, std::size_t dim_>
    class Box
    {
    public:

        static constexpr std::size_t dim = dim_;
        using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

        Box() = default;
        Box(Box const&) = default;
        Box(Box&&) = default;
        Box& operator=(Box const&) = default;
        Box& operator=(Box&&) = default;

        Box(point_t const& min_corner, point_t const& max_corner);

        inline auto const& min_corner() const;
        inline auto& min_corner();

        inline auto const& max_corner() const;
        inline auto& max_corner();

        inline auto length() const;
        inline bool is_valid() const;

        inline Box& operator*=(value_t v);

    private:

        point_t m_min_corner{0};
        point_t m_max_corner{0};
    };

    /**********************
     * Box implementation *
     **********************/

    template<class value_t, std::size_t dim_> 
    Box<value_t, dim_>::
    Box(point_t const& min_corner, point_t const& max_corner):
        m_min_corner{min_corner}, m_max_corner{max_corner}{}

    /**
     * Return the min corner of the box.
     */
    template<class value_t, std::size_t dim_>
    inline auto const& Box<value_t, dim_>::min_corner() const
    {
        return m_min_corner;
    }

    /**
     * Return the min corner of the box.
     */
    template<class value_t, std::size_t dim_>
    inline auto& Box<value_t, dim_>::min_corner()
    {
        return m_min_corner;
    }

    /**
     * Return the max corner of the box.
     */
    template<class value_t, std::size_t dim_>
    inline auto const& Box<value_t, dim_>::max_corner() const
    {
        return m_max_corner;
    }

    /**
     * Return the max corner of the box.
     */
    template<class value_t, std::size_t dim_>
    inline auto& Box<value_t, dim_>::max_corner()
    {
        return m_max_corner;
    }

    /**
     * Return the length of the box.
     */
    template<class value_t, std::size_t dim_>
    inline auto Box<value_t, dim_>::length() const
    {
        return (m_max_corner - m_min_corner);
    }

    /**
     * Check if the box is valid.
     */
    template<class value_t, std::size_t dim_>
    inline bool Box<value_t, dim_>::is_valid() const
    {
        return xt::all(m_min_corner < m_max_corner);
    }

    template<class value_t, std::size_t dim_>
    inline Box<value_t, dim_>& Box<value_t, dim_>::operator*=(value_t v)
    {
        m_min_corner *= v;
        m_max_corner *= v;
        return *this;
    }

    template<class value_t, std::size_t dim_>
    inline auto operator*(Box<value_t, dim_> const& box, value_t v)
    {
        Box<value_t, dim_> that(box);
        return that *= v;
    }

    template<class value_t, std::size_t dim_>
    inline auto operator*(value_t v, Box<value_t, dim_> const& box)
    {
        Box<value_t, dim_> that(box);
        return that *= v;
    }

    template<class value_t, std::size_t dim>
    std::ostream& operator<<(std::ostream& out, Box<value_t, dim> const& box)
    {
        out << "Box(" << box.min_corner() << ", " << box.max_corner() << ")";
        return out;
    }
}
