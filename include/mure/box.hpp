#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace mure
{
    template<typename value_t, std::size_t dim_>
    class Box
    {
    public:

        static constexpr std::size_t dim = dim_;
        using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

        Box()
        {
            m_min_corner.fill(0);
            m_max_corner.fill(0);
        }

        inline Box(point_t const& min_corner, point_t const& max_corner):
            m_min_corner{min_corner}, m_max_corner{max_corner}{}

        inline point_t const& min_corner() const
        {
            return m_min_corner;
        }

        inline point_t& min_corner()
        {
            return m_min_corner;
        }

        inline point_t const& max_corner() const
        {
            return m_max_corner;
        }

        inline point_t& max_corner()
        {
            return m_max_corner;
        }

        inline bool is_valid() const
        {
            return xt::all(m_min_corner < m_max_corner);
        }

    private:

        point_t m_min_corner;
        point_t m_max_corner;
    };

    template<typename value_t, std::size_t dim>
    std::ostream& operator<<(std::ostream& out, Box<value_t, dim> const& box)
    {
        out << "Box(" << box.min_corner() << ", " << box.max_corner() << ")";
        return out;
    }
}
