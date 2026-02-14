// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <ranges>

#include "../box.hpp"
#include "../samurai_config.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#endif

namespace samurai
{
    namespace mpi_neighbor
    {

        /**
         * @brief Compact representation of a subdomain for neighbor discovery.
         *
         * This structure contains only the essential geometric information needed
         * for the initial screening phase of neighbor discovery, dramatically
         * reducing communication volume compared to broadcasting full mesh structures.
         *
         * Size: ~64 bytes (2×dim doubles + metadata) vs. ~KB for full mesh
         *
         * @tparam dim Spatial dimension
         */
        template <std::size_t dim>
        struct SubdomainBoundingBox
        {
            /// MPI rank owning this subdomain
            int rank = -1;

            /// Cell length used for expansion in neighbor detection
            double cell_length = 0.0;

            /// Tight axis-aligned bounding box of subdomain
            Box<double, dim> bbox;

            SubdomainBoundingBox(int rank_, double cell_length_, const Box<double, dim>& bbox_)
                : rank(rank_)
                , cell_length(cell_length_)
                , bbox(bbox_)
            {
            }

#ifdef SAMURAI_WITH_MPI
            /// Boost.Serialization support for MPI communication - save
            template <class Archive>
            void save(Archive& ar, [[maybe_unused]] const unsigned int version) const
            {
                ar & rank;
                ar & cell_length;
                // Serialize bbox corners element-by-element to avoid xtensor operator& issues
                for (std::size_t d = 0; d < dim; ++d)
                {
                    double min_val = bbox.min_corner()[d];
                    double max_val = bbox.max_corner()[d];
                    ar & min_val;
                    ar & max_val;
                }
            }

            /// Boost.Serialization support for MPI communication - load
            template <class Archive>
            void load(Archive& ar, [[maybe_unused]] const unsigned int version)
            {
                ar & rank;
                ar & cell_length;
                // Deserialize bbox corners element-by-element
                typename Box<double, dim>::point_t min_c, max_c;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    double min_val, max_val;
                    ar & min_val;
                    ar & max_val;
                    min_c[d] = min_val;
                    max_c[d] = max_val;
                }
                bbox = Box<double, dim>{min_c, max_c};
            }

            BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

            /// Check if this bbox could intersect with another (with expansion)
            bool could_be_neighbor(const SubdomainBoundingBox& other) const
            {
                auto expanded  = bbox;
                auto expansion = std::min(cell_length, other.cell_length);

                for (std::size_t d = 0; d < dim; ++d)
                {
                    expanded.min_corner()[d] -= expansion;
                    expanded.max_corner()[d] += expansion;
                }

                return expanded.intersects(other.bbox);
            }
        };

        /**
         * @brief Compute tight bounding box from interval-based level cell array.
         *
         * This function traverses all intervals in the subdomain and computes
         * the minimal axis-aligned bounding box that contains all cells.
         *
         * @tparam LCA_type Level cell array type (e.g., LevelCellArray)
         * @param lca The subdomain to compute bbox for
         * @return Bounding box in physical coordinates
         */
        template <class LCA_type>
        auto compute_subdomain_bbox(const LCA_type& lca)
        {
            static constexpr std::size_t dim = LCA_type::dim;
            using point_t                    = typename Box<double, dim>::point_t;

            point_t min_corner;
            point_t max_corner;

            // Initialize to extreme values
            min_corner.fill(std::numeric_limits<double>::max());
            max_corner.fill(std::numeric_limits<double>::lowest());

            // Check if level is empty
            if (lca.empty())
            {
                // Empty subdomain - return degenerate bbox
                min_corner.fill(0.0);
                max_corner.fill(0.0);
                return SubdomainBoundingBox<dim>{
                    -1, // invalid rank
                    0.0, // invalid cell length
                    Box<double, dim>{min_corner, max_corner}
                };
            }

            double cell_length = lca.cell_length();

            for (std::size_t d = 0; d < dim; ++d)
            {
                min_corner[d] = std::ranges::min(
                                    lca[d],
                                    [](const auto& a, const auto& b)
                                    {
                                        return a.start < b.start;
                                    }).start
                                  * cell_length
                              + lca.origin_point()[d];
                max_corner[d] = std::ranges::max(
                                    lca[d],
                                    [](const auto& a, const auto& b)
                                    {
                                        return a.end < b.end;
                                    }).end
                                  * cell_length
                              + lca.origin_point()[d];
            }

            return SubdomainBoundingBox<dim>{
                -1, // rank will be set by caller
                cell_length,
                Box<double, dim>{min_corner, max_corner}
            };
        }

    } // namespace mpi_neighbor
} // namespace samurai
