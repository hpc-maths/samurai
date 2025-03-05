// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "utils.hpp"
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

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

        bool intersects(const Box& other) const;
        Box intersection(const Box& other) const;
        std::vector<Box> difference(const Box& other) const;

      private:

        void difference_impl_rec(Box& box, const Box& intersection, std::size_t d, std::vector<Box>& boxes) const;

      public:

        bool operator==(const Box& other) const;
        bool operator!=(const Box& other) const;
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

    /**
     * Check if the box intersects with another box.
     */
    template <class value_t, std::size_t dim_>
    inline bool Box<value_t, dim_>::intersects(const Box& other) const
    {
        return xt::all(m_min_corner < other.m_max_corner) && xt::all(m_max_corner > other.m_min_corner);
    }

    /**
     * Return the intersection of the box with another box.
     */
    template <class value_t, std::size_t dim_>
    inline Box<value_t, dim_> Box<value_t, dim_>::intersection(const Box& other) const
    {
        Box<value_t, dim_> box;
        box.min_corner() = xt::maximum(m_min_corner, other.m_min_corner);
        box.max_corner() = xt::minimum(m_max_corner, other.m_max_corner);
        return box;
    }

    template <class value_t, std::size_t dim_>
    void Box<value_t, dim_>::difference_impl_rec(Box& box, const Box& intersection, std::size_t d, std::vector<Box>& boxes) const
    {
        if (d == dim_)
        {
            return;
        }

        box.min_corner()[d] = this->min_corner()[d];
        box.max_corner()[d] = intersection.min_corner()[d];
        if (d == dim - 1 && box.is_valid())
        {
            boxes.push_back(box);
            // std::cout << box << std::endl;
        }

        difference_impl_rec(box, intersection, d + 1, boxes);

        box.min_corner()[d] = intersection.min_corner()[d];
        box.max_corner()[d] = intersection.max_corner()[d];
        if (d == dim - 1 && box.is_valid() && box != intersection) // The intersection is what we want to remove, so we don't add it
        {
            boxes.push_back(box);
            // std::cout << box << std::endl;
        }

        difference_impl_rec(box, intersection, d + 1, boxes);

        box.min_corner()[d] = intersection.max_corner()[d];
        box.max_corner()[d] = this->max_corner()[d];
        if (d == dim - 1 && box.is_valid())
        {
            boxes.push_back(box);
            // std::cout << box << std::endl;
        }

        difference_impl_rec(box, intersection, d + 1, boxes);
    }

    /**
     * Removes the intersection of the box with another box.
     * The result is a list of boxes.
     */
    template <class value_t, std::size_t dim_>
    inline std::vector<Box<value_t, dim_>> Box<value_t, dim_>::difference(const Box& other) const
    {
        std::vector<Box<value_t, dim_>> boxes;
        if (!intersects(other))
        {
            boxes.push_back(*this);
            return boxes;
        }

        auto intersection = this->intersection(other);

        Box<value_t, dim_> box;
        box.min_corner() = this->min_corner();
        box.max_corner() = intersection.min_corner();

        difference_impl_rec(box, intersection, 0, boxes);
        return boxes;
    }

    /**
     * Check if the box is equal to another box.
     */
    template <class value_t, std::size_t dim_>
    inline bool Box<value_t, dim_>::operator==(const Box& other) const
    {
        return m_min_corner == other.m_min_corner && m_max_corner == other.m_max_corner;
    }

    /**
     * Check if the box is different from another box.
     */
    template <class value_t, std::size_t dim_>
    inline bool Box<value_t, dim_>::operator!=(const Box& other) const
    {
        return !(*this == other);
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

    // Function to compute the GCD of two double values
    double gcd_double(double a, double b)
    {
        if (a == 0.0 && b == 0.0)
        {
            return 0.0; // GCD of 0 and 0 is 0
        }

        if (a == 0.0)
        {
            return std::abs(b); // GCD of 0 and b is |b|
        }

        if (b == 0.0)
        {
            return std::abs(a); // GCD of a and 0 is |a|
        }

        // Scale the doubles to integers by finding a common denominator
        int scale_a = 0;
        int scale_b = 0;

        double temp_a = std::abs(a);
        double temp_b = std::abs(b);

        while (std::floor(temp_a) != temp_a)
        {
            temp_a *= 10;
            scale_a++;
        }

        while (std::floor(temp_b) != temp_b)
        {
            temp_b *= 10;
            scale_b++;
        }

        int scale = std::max(scale_a, scale_b);

        long long int_a = static_cast<long long>(std::abs(a) * std::pow(10, scale));
        long long int_b = static_cast<long long>(std::abs(b) * std::pow(10, scale));

        // Compute the GCD of the scaled integers
        long long int_gcd = std::gcd(int_a, int_b);

        // Scale the GCD back to the original scale
        return static_cast<double>(int_gcd) / std::pow(10, scale);
    }

    template <class value_t, std::size_t dim>
    Box<value_t, dim> approximate_box(const Box<value_t, dim>& box, double tol, double& subdivision_length)
    {
        using length_t = typename Box<value_t, dim>::point_t;

        assert(tol >= 0 || subdivision_length > 0);

        // bool given_subdivision_length = subdivision_length > 0;
        length_t approx_length;
        if (subdivision_length > 0)
        {
            approx_length = xt::ceil(box.length() / subdivision_length) * subdivision_length;
        }
        else
        {
            // The largest possible subdivision length to exactly approximate the box
            // is the Greatest Common Divisor (GCD) of the box's lengths.
            subdivision_length = box.length()[0];
            for (std::size_t d = 1; d < dim; ++d)
            {
                subdivision_length = gcd_float(subdivision_length, box.length()[d]);
            }

            approx_length = xt::ceil(box.length() / subdivision_length) * subdivision_length;

            length_t error        = xt::abs(approx_length - box.length());
            length_t relative_tol = tol * box.length();

            // If the subdivision length is really too small...
            const double small_subdivision_length_tol = 1e-5;
            if (xt::any(subdivision_length < small_subdivision_length_tol * box.length()))
            {
                // ... and no tolerance is allowed, we raise an error.
                if (tol == 0)
                {
                    std::cerr << "The box " << box << " cannot be exactly represented with a reasonable cell length. ";
                    std::cerr << "You can modify the box's dimensions or you can set a tolerance so it can be approximately represented."
                              << std::endl;
                    std::exit(1);
                }

                // ... we set it to the smallest length of the box...
                subdivision_length = box.min_length();
                approx_length      = xt::ceil(box.length() / subdivision_length) * subdivision_length;
                error              = xt::abs(approx_length - box.length());
                // ... and reduce it to fit the tolerance.
                while (xt::any(error > relative_tol))
                {
                    subdivision_length /= 2;
                    approx_length = xt::ceil(box.length() / subdivision_length) * subdivision_length;
                    error         = xt::abs(approx_length - box.length());
                    // std::cout << "Approximation error: " << error << std::endl;
                }
            }
            else if (tol > 0)
            {
                // Since a tolerance is allowed, we try to find a larger subdivision within that tolerance.
                // To do so, we successively double the subdivision length until the approximation error exceeds the tolerance.

                if (xt::all(error < relative_tol))
                {
                    while (xt::all(error < relative_tol))
                    {
                        subdivision_length *= 2;
                        approx_length = xt::ceil(box.length() / subdivision_length) * subdivision_length;
                        error         = xt::abs(approx_length - box.length());
                        // std::cout << "Approximation error: " << error << std::endl;
                    }
                    subdivision_length /= 2;
                    approx_length = xt::ceil(box.length() / subdivision_length) * subdivision_length;
                    error         = xt::abs(approx_length - box.length());
                }
            }

            if (subdivision_length == 0)
            {
                SAMURAI_ASSERT(subdivision_length > 0, "An error occurred while approximating the box.");
            }
        }

        Box<value_t, dim> approx_box;
        approx_box.min_corner() = box.min_corner();
        approx_box.max_corner() = box.min_corner() + approx_length;
        return approx_box;
    }
} // namespace samurai
