// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "box.hpp"

namespace samurai
{
    template <std::size_t dim>
    class DomainBuilder
    {
        using Box     = Box<double, dim>;
        using point_t = typename Box::point_t;

      private:

        std::vector<Box> m_added_boxes;
        std::vector<Box> m_removed_boxes;

      public:

        DomainBuilder() = default;

        explicit DomainBuilder(const Box& box)
        {
            add(box);
        }

        explicit DomainBuilder(const point_t& min_corner, const point_t& max_corner)
        {
            add(Box(min_corner, max_corner));
        }

        auto& added_boxes() const
        {
            return m_added_boxes;
        }

        auto& removed_boxes() const
        {
            return m_removed_boxes;
        }

        point_t origin_point() const
        {
            point_t origin;
            for (const auto& box : m_added_boxes)
            {
                origin = xt::minimum(origin, box.min_corner());
            }
            for (const auto& box : m_removed_boxes)
            {
                origin = xt::minimum(origin, box.min_corner());
            }
            return origin;
        }

        void add(const Box& box)
        {
            m_added_boxes.push_back(box);
        }

        void add(const point_t& min_corner, const point_t& max_corner)
        {
            add(Box(min_corner, max_corner));
        }

        void remove(const Box& box)
        {
            m_removed_boxes.push_back(box);
        }

        void remove(const point_t& min_corner, const point_t& max_corner)
        {
            remove(Box(min_corner, max_corner));
        }

        double largest_subdivision() const
        {
            double largest_subdivision = m_added_boxes[0].min_length();

            // The largest subdivision must be smaller than the smallest legnth of all boxes
            for (const auto& box : m_added_boxes)
            {
                largest_subdivision = gcd_double(largest_subdivision, box.min_length());
            }
            for (const auto& box : m_removed_boxes)
            {
                largest_subdivision = gcd_double(largest_subdivision, box.min_length());
            }

            // The largest subdivision must be smaller than the smallest length of all differences
            for (const auto& box : m_added_boxes)
            {
                for (const auto& rbox : m_removed_boxes)
                {
                    if (rbox.intersects(box))
                    {
                        std::vector<Box> diff = box.difference(rbox);
                        for (const auto& dbox : diff)
                        {
                            largest_subdivision = gcd_float(largest_subdivision, dbox.min_length());
                        }
                    }
                }
            }

            assert(largest_subdivision > 0);
            return largest_subdivision;
        }
    };
}
