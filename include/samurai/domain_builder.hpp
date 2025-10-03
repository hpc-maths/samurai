// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <numeric>

#include "box.hpp"

namespace samurai
{
    template <std::size_t dim>
    class DomainBuilder
    {
        using box_t   = Box<double, dim>;
        using point_t = typename box_t::point_t;

      private:

        std::vector<box_t> m_added_boxes;
        std::vector<box_t> m_removed_boxes;

      public:

        DomainBuilder() = default;

        explicit DomainBuilder(const box_t& box)
        {
            add(box);
        }

        explicit DomainBuilder(const point_t& min_corner, const point_t& max_corner)
        {
            add(box_t(min_corner, max_corner));
        }

        const auto& added_boxes() const
        {
            return m_added_boxes;
        }

        const auto& removed_boxes() const
        {
            return m_removed_boxes;
        }

        point_t origin_point() const
        {
            point_t origin;
            origin.fill(std::numeric_limits<double>::max());
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

        void add(const box_t& box)
        {
            m_added_boxes.push_back(box);
        }

        void add(const point_t& min_corner, const point_t& max_corner)
        {
            add(box_t(min_corner, max_corner));
        }

        void remove(const box_t& box)
        {
            m_removed_boxes.push_back(box);
        }

        void remove(const point_t& min_corner, const point_t& max_corner)
        {
            remove(box_t(min_corner, max_corner));
        }

        double largest_subdivision() const
        {
            double largest_subdivision = m_added_boxes[0].min_length();

            auto gcd_loop = [](double largest_subdiv, const auto& boxes)
            {
                // for (const auto& box : boxes)
                // {
                //     largest_subdiv = gcd_float(largest_subdiv, box.min_length());
                // }
                // return largest_subdiv;

                return std::accumulate(boxes.begin(),
                                       boxes.end(),
                                       largest_subdiv,
                                       [](double l, const auto& box)
                                       {
                                           return gcd_float(l, box.min_length());
                                       });
            };

            // The largest subdivision must be smaller than the smallest legnth of all boxes
            largest_subdivision = gcd_loop(largest_subdivision, m_added_boxes);
            largest_subdivision = gcd_loop(largest_subdivision, m_removed_boxes);
            // for (const auto& box : m_added_boxes)
            // {
            //     largest_subdivision = gcd_float(largest_subdivision, box.min_length());
            // }
            // for (const auto& box : m_removed_boxes)
            // {
            //     largest_subdivision = gcd_float(largest_subdivision, box.min_length());
            // }

            // The largest subdivision must be smaller than the smallest length of all differences
            for (const auto& box : m_added_boxes)
            {
                for (const auto& rbox : m_removed_boxes)
                {
                    if (rbox.intersects(box))
                    {
                        std::vector<box_t> diff = box.difference(rbox);
                        largest_subdivision     = gcd_loop(largest_subdivision, diff);
                        // for (const auto& dbox : diff)
                        // {
                        //     largest_subdivision = gcd_float(largest_subdivision, dbox.min_length());
                        // }
                    }
                }
            }

            assert(largest_subdivision > 0);
            return largest_subdivision;
        }
    };
}
