// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "../arguments.hpp"
#include "../box.hpp"

namespace samurai
{
    class mra_config
    {
      public:

        /**
         * A level constraint: on the region, every cell is at a level `>= level`.
         *
         * The region is kept as the corners of a box in real coordinates, whatever the
         * dimension, so that the config itself carries no dimension.
         */
        struct level_constraint
        {
            std::vector<double> min_corner;
            std::vector<double> max_corner;
            std::size_t level;
        };

        auto& epsilon(double eps)
        {
            m_epsilon = eps;
            return *this;
        }

        auto& epsilon() const
        {
            return m_epsilon;
        }

        auto& regularity(double reg)
        {
            m_regularity = reg;
            return *this;
        }

        auto& regularity() const
        {
            return m_regularity;
        }

        auto& relative_detail(bool rel)
        {
            m_rel_detail = rel;
            return *this;
        }

        auto& relative_detail() const
        {
            return m_rel_detail;
        }

        /**
         * Require every cell of @a region to be at a level of at least @a level.
         *
         * A **guarantee**, not a hint: the constraint is applied as a tag - refine below
         * @a level, keep at @a level - which composes with the multiresolution criterion
         * through the maximum the adaptation already takes, and graduation only ever adds
         * refinement, so it can never lower a level the constraint asked for. Where the
         * region does not touch the boundary this is a plain refinement region; where it
         * does, it is what a boundary condition whose data varies along the boundary needs:
         * the multiresolution detail sees the solution, not the data, and a discontinuity
         * in the data is declared here rather than detected.
         *
         * The region is a value, not a callable: time dependence is expressed by passing a
         * different config to each adaptation, as the epsilon is.
         *
         * This replaces the `--refine-boundary` flag, which could only *keep* the boundary
         * at `max_level` where it was already there, never bring it there, and applied to
         * the whole boundary at once.
         */
        template <std::size_t dim>
        auto& min_level_in(const Box<double, dim>& region, std::size_t level)
        {
            level_constraint constraint;
            constraint.min_corner.assign(region.min_corner().begin(), region.min_corner().end());
            constraint.max_corner.assign(region.max_corner().begin(), region.max_corner().end());
            constraint.level = level;
            m_min_levels.push_back(std::move(constraint));
            return *this;
        }

        const std::vector<level_constraint>& min_levels() const
        {
            return m_min_levels;
        }

        void parse_args()
        {
            if (args::epsilon != std::numeric_limits<double>::infinity())
            {
                m_epsilon = args::epsilon;
            }
            if (args::regularity != std::numeric_limits<double>::infinity())
            {
                m_regularity = args::regularity;
            }
            if (args::rel_detail)
            {
                m_rel_detail = true;
            }
        }

      private:

        double m_epsilon    = 1e-4;
        double m_regularity = 1.;
        bool m_rel_detail   = false;

        std::vector<level_constraint> m_min_levels;
    };
}
