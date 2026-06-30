// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#endif

#include "../box.hpp"
#include "../samurai_config.hpp"

namespace samurai::mpi_neighbor
{

    /** @class SubdomainBoundingBox
     *  @brief Compact representation of a subdomain for scalable neighbor discovery.
     *
     * Contains essential geometric information for the screening phase of
     * neighbor discovery, reducing communication from \f$O(N)\f$ full meshes
     * to \f$O(N)\f$ compact bboxes (~64 bytes vs. KB per subdomain).
     *
     * The bounding box is tight (minimal) and uses cell_length for
     * conservative expansion during intersection tests.
     *
     * @tparam dim Spatial dimension.
     */
    template <std::size_t dim>
    struct SubdomainBoundingBox
    {
        int rank           = -1;  ///< MPI rank owning this subdomain.
        double cell_length = 0.0; ///< Cell length for conservative expansion.
        double ghost_reach = 0.0; ///< Physical reach of this rank's ghost construction
                                  ///< (see Mesh_base::ghost_physical_reach()).
        Box<double, dim> bbox;    ///< Tight axis-aligned bounding box.

        SubdomainBoundingBox() = default;

        /**
         * Construct a subdomain bounding box.
         *
         * @param rank_        MPI rank owning the subdomain.
         * @param cell_length_ Cell length at the subdomain's finest level.
         * @param bbox_        Tight bounding box in physical coordinates.
         */
        SubdomainBoundingBox(int rank_, double cell_length_, const Box<double, dim>& bbox_)
            : rank(rank_)
            , cell_length(cell_length_)
            , bbox(bbox_)
        {
        }

#ifdef SAMURAI_WITH_MPI
        /**
         * Serialize the bounding box for MPI communication.
         *
         * @param ar      Archive for serialization.
         * @param version Serialization version (unused).
         */
        template <class Archive>
        void save(Archive& ar, [[maybe_unused]] const unsigned int version) const
        {
            ar & rank;
            ar & cell_length;
            ar & ghost_reach;
            // Serialize bbox corners element-by-element to avoid xtensor operator& issues
            for (std::size_t d = 0; d < dim; ++d)
            {
                double min_val = bbox.min_corner()[d];
                double max_val = bbox.max_corner()[d];
                ar & min_val;
                ar & max_val;
            }
        }

        /**
         * Deserialize the bounding box from MPI communication.
         *
         * @param ar      Archive for deserialization.
         * @param version Serialization version (unused).
         */
        template <class Archive>
        void load(Archive& ar, [[maybe_unused]] const unsigned int version)
        {
            ar & rank;
            ar & cell_length;
            ar & ghost_reach;
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

        /**
         * Test if this subdomain could be a neighbor of another.
         *
         * Expands the bounding box by the largest ghost reach of the two
         * subdomains before testing intersection: a rank must be detected as
         * a neighbor as soon as it may contribute values to our ghosts (or we
         * to its), even when a thin third subdomain lies in between. Using the
         * max of both reaches keeps the decision symmetric. Falls back to one
         * cell length when no reach is available.
         *
         * @param other Another subdomain's bounding box.
         * @return      True if the expanded boxes intersect.
         */
        bool could_be_neighbor(const SubdomainBoundingBox& other) const
        {
            if (cell_length == 0.0 || other.cell_length == 0.0)
            {
                // If either subdomain is empty, they cannot be neighbors
                return false;
            }

            // The decision is taken in integer cell units rather than physical
            // coordinates. Subdomains live at the configuration max level, so
            // cell_length is identical on every rank and is an exact common
            // unit. In physical doubles, two subdomains separated by exactly the
            // expansion distance fall on a knife-edge where rounding can make
            // A.could_be_neighbor(B) != B.could_be_neighbor(A); that asymmetry
            // then deadlocks the (symmetry-assuming) candidate exchange in
            // Mesh_base::find_neighbourhood(). Integer arithmetic is exactly
            // symmetric, so the two ranks always agree.
            const double cl = std::min(cell_length, other.cell_length);

            // Expansion in whole cells, mirroring the physical version's
            // max(my reach, other reach, one cell). ghost_reach / cl is an exact
            // integer count (ghost_reach is a whole number of coarse cells);
            // llround clears the floating-point noise.
            const std::int64_t expansion = std::max<std::int64_t>(
                {std::llround(ghost_reach / cl), std::llround(other.ghost_reach / cl), std::int64_t{1}});

            for (std::size_t d = 0; d < dim; ++d)
            {
                // Integer cell gaps along axis d. The shared domain origin baked
                // into both corners cancels in the differences, leaving exact
                // index gaps; llround removes the residual float noise.
                const std::int64_t lo_gap = std::llround((bbox.min_corner()[d] - other.bbox.max_corner()[d]) / cl);
                const std::int64_t hi_gap = std::llround((other.bbox.min_corner()[d] - bbox.max_corner()[d]) / cl);

                // Reproduces Box::intersects (strict, half-open) on the box
                // expanded by `expansion`: overlap on this axis requires both
                // gaps strictly below the expansion. Symmetric under A<->B swap.
                if (lo_gap >= expansion || hi_gap >= expansion)
                {
                    return false;
                }
            }

            return true;
        }
    };

    /**
     * Compute tight bounding box from interval-based cell array.
     *
     * Traverses all intervals to find extrema coordinates, then converts
     * to physical space using cell length and origin. Returns a degenerate
     * box for empty subdomains.
     *
     * Complexity: \f$O(n \cdot d)\f$ where \f$n\f$ is the number of intervals
     * per dimension and \f$d\f$ is the spatial dimension.
     *
     * @tparam LCA_type Level cell array type (e.g., LevelCellArray).
     * @param  lca      Cell array to compute bounding box for.
     * @return          Bounding box with rank=-1 (to be set by caller).
     */
    template <class LCA_type>
    auto compute_subdomain_bbox(const LCA_type& lca)
    {
        static constexpr std::size_t dim = LCA_type::dim;
        using point_t                    = typename Box<double, dim>::point_t;
        using coord_t                    = typename LCA_type::interval_t::value_t;
        point_t min_corner;
        point_t max_corner;

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
            auto min_start = std::numeric_limits<coord_t>::max();
            auto max_end   = std::numeric_limits<coord_t>::lowest();

            for (const auto& interval : lca[d])
            {
                min_start = std::min(min_start, interval.start);
                max_end   = std::max(max_end, interval.end);
            }

            min_corner[d] = min_start * cell_length + lca.origin_point()[d];
            max_corner[d] = max_end * cell_length + lca.origin_point()[d];
        }

        return SubdomainBoundingBox<dim>{
            -1, // rank will be set by caller
            cell_length,
            Box<double, dim>{min_corner, max_corner}
        };
    }

} // namespace samurai::mpi_neighbor
