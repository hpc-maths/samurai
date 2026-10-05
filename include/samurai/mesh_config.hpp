// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "arguments.hpp"
#include "cell_array.hpp"
#include "samurai_config.hpp"

#include <array>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>

namespace samurai
{

    template <std::size_t dim_,
              int prediction_stencil_radius_    = default_config::prediction_stencil_radius,
              std::size_t max_refinement_level_ = default_config::max_level,
              class interval_t_                 = default_config::interval_t>
    class mesh_config
    {
      public:

        static constexpr std::size_t dim                  = dim_;
        static constexpr int prediction_stencil_radius    = prediction_stencil_radius_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_;

        using interval_t = interval_t_;

        mesh_config()
        {
            m_periodic.fill(false);
        }

        // m_max_stencil_radius ---------------------------

        /**
         * @brief set max stencil radius in chained config
         *
         * @param stencil_radius largest stencil radius of the schemes run on the mesh, in cells
         * @return auto& returns this object
         */
        auto& max_stencil_radius(int stencil_radius)
        {
            m_max_stencil_radius = stencil_radius;
            return *this;
        }

        /**
         * @brief get a reference on max stencil radius
         */
        auto& max_stencil_radius()
        {
            return m_max_stencil_radius;
        }

        /**
         * @brief get a reference on max stencil radius
         */
        const auto& max_stencil_radius() const
        {
            return m_max_stencil_radius;
        }

        /**
         * @brief set max stencil radius from a stencil size: the radius is the size divided by 2, rounded up
         *
         * @param stencil_size number of cells of the largest stencil along one direction
         * @return auto& returns this object
         */
        auto& max_stencil_size(int stencil_size)
        {
            m_max_stencil_radius = stencil_size / 2;
            if (stencil_size % 2 == 1)
            {
                m_max_stencil_radius += 1;
            }
            return *this;
        }

        /**
         * @brief get value of max stencil size
         */
        auto max_stencil_size() const
        {
            return m_max_stencil_radius * 2;
        }

        // m_graduation_width -----------------------------

        /**
         * @brief set graduation width in chained config
         *
         * @param grad_width graduation width of the mesh, in cells
         * @return auto& returns this object
         */
        auto& graduation_width(std::size_t grad_width)
        {
            m_graduation_width = grad_width;
            return *this;
        }

        /**
         * @brief get a reference on graduation width
         */
        auto& graduation_width()
        {
            return m_graduation_width;
        }

        /**
         * @brief get a reference on graduation width
         */
        const auto& graduation_width() const
        {
            return m_graduation_width;
        }

        // m_ghost_width ----------------------------------

        /**
         * @brief get a reference on ghost width
         */
        const auto& ghost_width() const
        {
            return m_ghost_width;
        }

        // m_min_level ------------------------------------

        /**
         * @brief set min level in chained config
         *
         * @param level coarsest level the mesh can hold
         * @return auto& returns this object
         */
        auto& min_level(std::size_t level)
        {
            m_min_level = level;
            return *this;
        }

        /**
         * @brief get a reference on min level
         */
        auto& min_level()
        {
            return m_min_level;
        }

        /**
         * @brief get a reference on min level
         */
        const auto& min_level() const
        {
            return m_min_level;
        }

        // m_max_level ------------------------------------

        /**
         * @brief set max level in chained config
         *
         * @param level finest level the mesh can hold
         * @return auto& returns this object
         */
        auto& max_level(std::size_t level)
        {
            m_max_level = level;
            return *this;
        }

        /**
         * @brief get a reference on max level
         */
        auto& max_level()
        {
            return m_max_level;
        }

        /**
         * @brief get a reference on max level
         */
        const auto& max_level() const
        {
            return m_max_level;
        }

        // m_start_level ------------------------------------

        /**
         * @brief set start level in chained config
         *
         * @param level level of the cells of the initial mesh
         * @return auto& returns this object
         */
        auto& start_level(std::size_t level)
        {
            m_start_level = level;
            return *this;
        }

        /**
         * @brief get the start level: the level set with start_level(level), or the max level if none was set
         */
        std::size_t start_level() const
        {
            return m_start_level == std::numeric_limits<std::size_t>::max() ? m_max_level : m_start_level;
        }

        // m_approx_box_tol -------------------------------

        /**
         * @brief set approximation box tolerance in chained config
         *
         * @param tol relative tolerance allowed when the domain box cannot be represented exactly with cells
         * @return auto& returns this object
         */
        auto& approx_box_tol(double tol)
        {
            m_approx_box_tol = tol;
            return *this;
        }

        /**
         * @brief get a reference on approximation box tolerance
         */
        auto& approx_box_tol()
        {
            return m_approx_box_tol;
        }

        /**
         * @brief get a reference on approximation box tolerance
         */
        const auto& approx_box_tol() const
        {
            return m_approx_box_tol;
        }

        // m_scaling_factor -------------------------------

        /**
         * @brief set scaling factor in chained config
         *
         * @param factor cell length at level 0; 0 (the default) lets the mesh compute it from the domain
         * @return auto& returns this object
         */
        auto& scaling_factor(double factor)
        {
            m_scaling_factor = factor;
            return *this;
        }

        /**
         * @brief get a reference on scaling factor
         */
        auto& scaling_factor()
        {
            return m_scaling_factor;
        }

        /**
         * @brief get a reference on scaling factor
         */
        const auto& scaling_factor() const
        {
            return m_scaling_factor;
        }

        // m_periodic -------------------------------------

        /**
         * @brief set periodicity in chained config
         *
         * @param periodicity one flag per direction, true for a periodic direction
         * @return auto& returns this object
         */
        auto& periodic(const std::array<bool, dim>& periodicity)
        {
            m_periodic = periodicity;
            return *this;
        }

        /**
         * @brief set periodicity in chained config with a value to fill in each direction
         *
         * @param periodicity true to make every direction periodic
         * @return auto& returns this object
         */
        auto& periodic(bool periodicity)
        {
            m_periodic.fill(periodicity);
            return *this;
        }

        /**
         * @brief get a reference on periodicity array
         */
        auto& periodic()
        {
            return m_periodic;
        }

        /**
         * @brief get a reference on periodicity array
         */
        const auto& periodic() const
        {
            return m_periodic;
        }

        /**
         * @brief get a reference on periodicity in direction i
         */
        auto& periodic(std::size_t i)
        {
            return m_periodic[i];
        }

        /**
         * @brief get a reference on periodicity in direction i
         */
        const auto& periodic(std::size_t i) const
        {
            return m_periodic[i];
        }

        // m_disable_args_parse ---------------------------

        /**
         * @brief disable argument parse
         */
        auto& disable_args_parse()
        {
            m_disable_args_parse = true;
            return *this;
        }

        // disable_minimal_ghost_width --------------------
        auto& disable_minimal_ghost_width()
        {
            m_disable_minimal_ghost_width = true;
            return *this;
        }

        /**
         * @brief throw std::invalid_argument unless min level <= start level <= max level
         */
        void check_levels() const
        {
            if (m_max_level < m_min_level)
            {
                throw std::invalid_argument("Max level must be greater than min level.");
            }
            if (start_level() < m_min_level || start_level() > m_max_level)
            {
                throw std::invalid_argument(
                    fmt::format("Start level {} must lie between min level {} and max level {}.", start_level(), m_min_level, m_max_level));
            }
        }

        /**
         * @brief parse arguments and set value to default samurai config value if needed
         */
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
                if (args::start_level != std::numeric_limits<std::size_t>::max())
                {
                    m_start_level = args::start_level;
                }
                check_levels();
            }

            if (!m_disable_minimal_ghost_width)
            {
                // 2 is because prediction_stencil_radius=1, if >1 we don't know what to do...
                // The idea is to have enough ghosts at the boundary for the reconstruction and the transfer to work.
                m_max_stencil_radius = std::max(m_max_stencil_radius, 2);
            }

            m_ghost_width = std::max(m_max_stencil_radius, static_cast<int>(prediction_stencil_radius));
        }

      private:

#ifdef SAMURAI_WITH_MPI
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned long)
        {
            ar & m_max_stencil_radius;
            ar & m_graduation_width;
            ar & m_ghost_width;
            ar & m_min_level;
            ar & m_max_level;
            ar & m_start_level;
            ar & m_approx_box_tol;
            ar & m_scaling_factor;
            ar & m_disable_args_parse;
        }
#endif

        int m_max_stencil_radius       = 1;
        std::size_t m_graduation_width = default_config::graduation_width;
        int m_ghost_width              = default_config::ghost_width;

        std::size_t m_min_level   = 0;
        std::size_t m_max_level   = 6;
        std::size_t m_start_level = std::numeric_limits<std::size_t>::max(); // unset: start at the max level

        double m_approx_box_tol = 0.05;
        double m_scaling_factor = 0;

        std::array<bool, dim> m_periodic;

        bool m_disable_args_parse          = false;
        bool m_disable_minimal_ghost_width = false;
    };
} // namespace samurai
