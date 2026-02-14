// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../box.hpp"
#include "../samurai_config.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/mpi.hpp>
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
     * @tparam T Coordinate type (typically double)
     * @tparam dim Spatial dimension
     */
    template <class T, std::size_t dim>
    struct SubdomainBoundingBox
    {
        /// MPI rank owning this subdomain
        int rank = -1;
        
        /// Tight axis-aligned bounding box of subdomain
        Box<T, dim> bbox;
        
        /// Minimum refinement level in subdomain
        std::size_t min_level = 0;
        
        /// Maximum refinement level in subdomain
        std::size_t max_level = 0;
        
        /// Number of cells (for load balancing info)
        std::size_t num_cells = 0;

        SubdomainBoundingBox() = default;
        
        SubdomainBoundingBox(int rank_, 
                            const Box<T, dim>& bbox_,
                            std::size_t min_level_,
                            std::size_t max_level_,
                            std::size_t num_cells_)
            : rank(rank_)
            , bbox(bbox_)
            , min_level(min_level_)
            , max_level(max_level_)
            , num_cells(num_cells_)
        {}

#ifdef SAMURAI_WITH_MPI
        /// Boost.Serialization support for MPI communication - save
        template<class Archive>
        void save(Archive& ar, [[maybe_unused]] const unsigned int version) const
        {
            ar & rank;
            // Serialize bbox corners element-by-element to avoid xtensor operator& issues
            for (std::size_t d = 0; d < dim; ++d)
            {
                double min_val = bbox.min_corner()[d];
                double max_val = bbox.max_corner()[d];
                ar & min_val;
                ar & max_val;
            }
            ar & min_level;
            ar & max_level;
            ar & num_cells;
        }
        
        /// Boost.Serialization support for MPI communication - load
        template<class Archive>
        void load(Archive& ar, [[maybe_unused]] const unsigned int version)
        {
            ar & rank;
            // Deserialize bbox corners element-by-element
            typename Box<T, dim>::point_t min_c, max_c;
            for (std::size_t d = 0; d < dim; ++d)
            {
                double min_val, max_val;
                ar & min_val;
                ar & max_val;
                min_c[d] = min_val;
                max_c[d] = max_val;
            }
            bbox = Box<T, dim>{min_c, max_c};
            ar & min_level;
            ar & max_level;
            ar & num_cells;
        }
        
        BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

        /// Check if this bbox could intersect with another (with expansion)
        bool could_be_neighbor(const SubdomainBoundingBox& other, 
                              double expansion_factor) const
        {
            auto expanded = bbox;
            auto expansion = expansion_factor * compute_max_cell_size();
            
            for (std::size_t d = 0; d < dim; ++d)
            {
                expanded.min_corner()[d] -= expansion;
                expanded.max_corner()[d] += expansion;
            }
            
            return expanded.intersects(other.bbox);
        }
        
        /// Compute maximum cell size in this subdomain
        T compute_max_cell_size() const
        {
            // At minimum level, cells are largest
            auto lengths = bbox.length();
            T max_length = lengths[0];
            for (std::size_t d = 1; d < dim; ++d)
            {
                max_length = std::max(max_length, lengths[d]);
            }
            // Approximate: assume domain spans roughly 2^min_level cells
            if (min_level == 0)
            {
                return max_length;
            }
            return max_length / (1 << min_level);
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
        -> SubdomainBoundingBox<double, LCA_type::dim>
    {
        static constexpr std::size_t dim = LCA_type::dim;
        using point_t = typename Box<double, dim>::point_t;
        
        point_t min_corner;
        point_t max_corner;
        
        // Initialize to extreme values
        min_corner.fill(std::numeric_limits<double>::max());
        max_corner.fill(std::numeric_limits<double>::lowest());
        
        std::size_t level = lca.level();
        std::size_t total_cells = 0;
        
        // Check if level is empty
        if (lca.offsets(0).size() <= 1)
        {
            // Empty subdomain - return degenerate bbox
            min_corner.fill(0.0);
            max_corner.fill(0.0);
            return SubdomainBoundingBox<double, dim>{
                -1, // invalid rank
                Box<double, dim>{min_corner, max_corner},
                level,
                level,
                0
            };
        }
        
        double cell_length = lca.cell_length();
        
        // Iterate over all intervals at this level
        for_each_interval(lca,
            [&](std::size_t, const auto& interval, const auto& index_yz)
            {
                // Convert integer interval to physical coordinates
                point_t cell_min;
                point_t cell_max;
                
                // X-direction from interval
                cell_min[0] = lca.origin_point()[0] + interval.start * cell_length;
                cell_max[0] = lca.origin_point()[0] + interval.end * cell_length;
                
                // Other dimensions from index
                for (std::size_t d = 1; d < dim; ++d)
                {
                    cell_min[d] = lca.origin_point()[d] + index_yz[d-1] * cell_length;
                    cell_max[d] = lca.origin_point()[d] + (index_yz[d-1] + 1) * cell_length;
                }
                
                // Update global bbox
                for (std::size_t d = 0; d < dim; ++d)
                {
                    min_corner[d] = std::min(min_corner[d], cell_min[d]);
                    max_corner[d] = std::max(max_corner[d], cell_max[d]);
                }
                
                total_cells += interval.size();
            });
        
        return SubdomainBoundingBox<double, dim>{
            -1, // rank will be set by caller
            Box<double, dim>{min_corner, max_corner},
            level,
            level,
            total_cells
        };
    }

} // namespace mpi_neighbor
} // namespace samurai
