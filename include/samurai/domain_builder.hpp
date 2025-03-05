#include "box.hpp"

namespace samurai
{
    template <std::size_t dim>
    class DomainBuilder
    {
        using Box     = Box<double, dim>;
        using point_t = typename Box::point_t;

      private:

        point_t m_origin_point;

        std::vector<Box> m_added_boxes;
        std::vector<Box> m_removed_boxes;

      public:

        DomainBuilder(const Box& box, bool fill_domain = true)
            : m_origin_point(box.min_corner())
        {
            if (fill_domain)
            {
                add(box);
            }
        }

        auto& origin_point() const
        {
            return m_origin_point;
        }

        void add(const Box& box)
        {
            m_added_boxes.push_back(box);
        }

        void remove(const Box& box)
        {
            m_removed_boxes.push_back(box);
        }

        double largest_subdivision() const
        {
            double largest_subdivision = m_added_boxes[0].min_length();

            // The largest subdivision must be smaller than the smallest legnth of all boxes
            for (const auto& box : m_added_boxes)
            {
                largest_subdivision = gcd_double(largest_subdivision, box.min_length());
                std::cout << largest_subdivision << std::endl;
            }
            for (const auto& box : m_removed_boxes)
            {
                largest_subdivision = gcd_double(largest_subdivision, box.min_length());
                std::cout << largest_subdivision << std::endl;
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
                            std::cout << "***********" << std::endl;
                            std::cout << dbox << std::endl;
                            // std::cout << dbox.min_length() << std::endl;
                            largest_subdivision = gcd_float(largest_subdivision, dbox.min_length());
                            std::cout << largest_subdivision << std::endl;
                        }
                    }
                }
            }

            assert(largest_subdivision > 0);
            return largest_subdivision;
        }
    };
}
