// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "arguments.hpp"
#include "samurai_config.hpp"
#include <array>

namespace samurai
{

    template <std::size_t dim_, std::size_t prediction_stencil_radius_ = 1>
    class mesh_config
    {
      public:

        static constexpr std::size_t dim                       = dim_;
        static constexpr std::size_t prediction_stencil_radius = prediction_stencil_radius_;

        mesh_config()
        {
            m_periodic.fill(false);
        }

        auto& max_stencil_radius(int stencil_radius)
        {
            m_max_stencil_radius = stencil_radius;
            return *this;
        }

        auto& max_stencil_radius()
        {
            return m_max_stencil_radius;
        }

        auto& max_stencil_radius() const
        {
            return m_max_stencil_radius;
        }

        auto& max_stencil_size(int stencil_size)
        {
            m_max_stencil_radius = stencil_size / 2;
            if (stencil_size % 2 == 1)
            {
                m_max_stencil_radius += 1;
            }
            return *this;
        }

        auto max_stencil_size() const
        {
            return m_max_stencil_radius * 2;
        }

        auto ghost_width() const
        {
            return m_ghost_width;
        }

        auto& graduation_width(std::size_t grad_width)
        {
            m_graduation_width = grad_width;
            return *this;
        }

        auto& graduation_width() const
        {
            return m_graduation_width;
        }

        auto& min_level(std::size_t level)
        {
            m_min_level = level;
            return *this;
        }

        auto& min_level()
        {
            return m_min_level;
        }

        auto& min_level() const
        {
            return m_min_level;
        }

        auto& max_level(std::size_t level)
        {
            m_max_level = level;
            return *this;
        }

        auto& max_level()
        {
            return m_max_level;
        }

        auto& max_level() const
        {
            return m_max_level;
        }

        auto& approx_box_tol(double tol)
        {
            m_approx_box_tol = tol;
            return *this;
        }

        auto& approx_box_tol() const
        {
            return m_approx_box_tol;
        }

        auto& scaling_factor(double factor)
        {
            m_scaling_factor = factor;
            return *this;
        }

        auto& scaling_factor() const
        {
            return m_scaling_factor;
        }

        auto& scaling_factor()
        {
            return m_scaling_factor;
        }

        auto& periodic(std::array<bool, dim> const& periodicity)
        {
            m_periodic = periodicity;
            return *this;
        }

        auto& periodic(bool periodicity)
        {
            m_periodic.fill(periodicity);
            return *this;
        }

        auto& periodic() const
        {
            return m_periodic;
        }

        auto& periodic(std::size_t i) const
        {
            return m_periodic[i];
        }

        auto& disable_args_parse()
        {
            m_disable_args_parse = false;
            return *this;
        }

        void parse_args()
        {
            if (!m_disable_args_parse)
            {
                if (args::max_stencil_radius != std::numeric_limits<int>::max())
                {
                    m_max_stencil_radius = args::max_stencil_radius;
                }
                if (args::graduation_width != std::numeric_limits<std::size_t>::max())
                {
                    m_graduation_width = args::graduation_width;
                }
                if (args::min_level != std::numeric_limits<std::size_t>::max())
                {
                    m_min_level = args::min_level;
                }
                if (args::max_level != std::numeric_limits<std::size_t>::max())
                {
                    m_max_level = args::max_level;
                }
                // if (args::approx_box_tol != std::numeric_limits<double>::infinity())
                // {
                //     m_approx_box_tol = args::approx_box_tol;
                // }
                // if (args::scaling_factor != std::numeric_limits<double>::infinity())
                // {
                //     m_scaling_factor = args::scaling_factor;
                // }
                if (m_max_level < m_min_level)
                {
                    std::cerr << "Max level must be greater than min level." << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            m_ghost_width = std::max(m_max_stencil_radius, static_cast<int>(prediction_stencil_radius));
        }

      private:

        int m_max_stencil_radius       = 1;
        std::size_t m_graduation_width = default_config::graduation_width;
        int m_ghost_width              = default_config::ghost_width;

        std::size_t m_min_level = 0;
        std::size_t m_max_level = 6;

        double m_approx_box_tol = 0.05;
        double m_scaling_factor = 0;

        std::array<bool, dim> m_periodic;

        bool m_disable_args_parse = false;
    };
}
