// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "arguments.hpp"
#include <array>

namespace samurai
{

    template <std::size_t dim_>
    class mesh_config
    {
      public:

        static constexpr std::size_t dim = dim_;

        mesh_config()
        {
            m_periodic.fill(false);
        }

        auto& max_stencil_width(std::size_t stencil_width)
        {
            m_max_stencil_width = stencil_width;
            return *this;
        }

        auto& max_stencil_width() const
        {
            return m_max_stencil_width;
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

        auto& min_level() const
        {
            return m_min_level;
        }

        auto& max_level(std::size_t level)
        {
            m_max_level = level;
            return *this;
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

        auto& periodic(std::array<bool, dim> const& periodicity)
        {
            m_periodic = periodicity;
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

        void parse_args()
        {
            // if (args::max_stencil_width != std::numeric_limits<std::size_t>::max())
            // {
            //     m_max_stencil_width = args::max_stencil_width;
            // }
            // if (args::graduation_width != std::numeric_limits<std::size_t>::max())
            // {
            //     m_graduation_width = args::graduation_width;
            // }
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
        }

      private:

        std::size_t m_max_stencil_width = 1;
        std::size_t m_graduation_width  = 1;

        std::size_t m_min_level;
        std::size_t m_max_level;

        double m_approx_box_tol = 0.05;
        double m_scaling_factor = 0;

        std::array<bool, dim> m_periodic;
    };
}
