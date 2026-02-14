// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

/**
 * Unit test for bounding box-based neighbor discovery (non-MPI version)
 *
 * This test validates the bbox computation and intersection logic
 * without requiring MPI.
 */

#include "samurai/box.hpp"
#include "samurai/cell_array.hpp"
#include "samurai/cell_list.hpp"
#include "samurai/mpi/subdomain_bbox.hpp"
#include <gtest/gtest.h>

// Test 1: Basic bbox computation from cell array
TEST(SubdomainBBox, BasicComputation)
{
    constexpr std::size_t dim = 2;

    constexpr std::size_t level = 3;
    samurai::LevelCellList<dim> lcl(level);

    lcl[{0}].add_interval({0, 8});

    samurai::LevelCellArray<dim> lca(lcl);

    // Compute bounding box
    auto bbox = samurai::mpi_neighbor::compute_subdomain_bbox(lca);

    // Verify bbox encompasses the domain
    EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[0], 0.0);
    EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[1], 0.0);
    EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[0], 1.0);
    EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[1], 0.125);

    // Recompute bbox with new origin
    lca.set_origin_point({-1.0, 0.5});
    bbox = samurai::mpi_neighbor::compute_subdomain_bbox(lca);
    EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[0], -1.0);
    EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[1], 0.5);
    EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[0], 0.0);
    EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[1], 0.625);

    // Recompute bbox with new scaling
    lca.set_scaling_factor(0.5);
    bbox = samurai::mpi_neighbor::compute_subdomain_bbox(lca);
    EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[0], -1.0);
    EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[1], 0.5);
    EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[0], -0.5);
    EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[1], 0.5625);
}

// Test 2: BBox intersection detection
TEST(SubdomainBBox, IntersectionDetection)
{
    constexpr std::size_t dim = 2;
    using BBox                = samurai::mpi_neighbor::SubdomainBoundingBox<dim>;

    // Create two adjacent bboxes
    BBox bbox1(0, 0.125, samurai::Box<double, dim>({0.0, 0.0}, {0.5, 1.0}));
    BBox bbox2(1, 0.125, samurai::Box<double, dim>({0.5, 0.0}, {1.0, 1.0}));

    // They should be neighbors (they touch)
    EXPECT_TRUE(bbox1.could_be_neighbor(bbox2));
    EXPECT_TRUE(bbox2.could_be_neighbor(bbox1));
}

// Test 3: Non-intersecting bboxes
TEST(SubdomainBBox, NonIntersecting)
{
    constexpr std::size_t dim = 2;

    using BBox = samurai::mpi_neighbor::SubdomainBoundingBox<dim>;

    // Create two distant bboxes
    BBox bbox1(0, 0.125, samurai::Box<double, dim>({0.0, 0.0}, {0.3, 0.3}));
    BBox bbox2(1, 0.125, samurai::Box<double, dim>({0.7, 0.7}, {1.0, 1.0}));

    // They should NOT be neighbors (too far apart)
    EXPECT_FALSE(bbox1.could_be_neighbor(bbox2));
    EXPECT_FALSE(bbox2.could_be_neighbor(bbox1));
}

// Demonstration test showing candidate reduction
TEST(SubdomainBBox, CandidateReduction)
{
    constexpr std::size_t dim = 2;
    using BBox                = samurai::mpi_neighbor::SubdomainBoundingBox<dim>;

    // Simulate 16 processes arranged in a 4x4 grid
    constexpr int grid_size     = 100;
    constexpr int num_processes = grid_size * grid_size;

    std::vector<BBox> all_bboxes;

    // Create bboxes for grid decomposition
    double cell_size = 1.0 / grid_size;
    for (std::size_t i = 0; i < grid_size; ++i)
    {
        for (std::size_t j = 0; j < grid_size; ++j)
        {
            int rank     = static_cast<int>((i * grid_size) + j);
            double x_min = static_cast<double>(i) * cell_size;
            double x_max = static_cast<double>(i + 1) * cell_size;
            double y_min = static_cast<double>(j) * cell_size;
            double y_max = static_cast<double>(j + 1) * cell_size;

            all_bboxes.emplace_back(rank, cell_size, samurai::Box<double, dim>({x_min, y_min}, {x_max, y_max}));
        }
    }

    // For process in center (rank 5: position [1,1])
    int test_rank       = 5;
    const auto& my_bbox = all_bboxes[static_cast<std::size_t>(test_rank)];

    // Count candidates using bbox screening
    int bbox_candidates = 0;
    for (const auto& other : all_bboxes)
    {
        if (other.rank != test_rank && my_bbox.could_be_neighbor(other))
        {
            bbox_candidates++;
        }
    }

    // Brute force would check all other processes
    int brute_force_candidates = num_processes - 1; // 15

    // Bbox screening should find only ~8 neighbors (Moore neighborhood)
    EXPECT_LT(bbox_candidates, brute_force_candidates);
    EXPECT_LE(bbox_candidates, 8); // At most 8 neighbors in 2D grid

    // Calculate reduction
    double reduction = 100.0 * (1.0 - static_cast<double>(bbox_candidates) / brute_force_candidates);

    std::cout << "\n=== Candidate Reduction Demo ===\n";
    std::cout << fmt::format("Processes: {} ({}x{} grid)\n", num_processes, grid_size, grid_size);
    std::cout << "Test rank: " << test_rank << " (center process)\n";
    std::cout << "Brute force candidates: " << brute_force_candidates << "\n";
    std::cout << "BBox candidates: " << bbox_candidates << "\n";
    std::cout << "Reduction: " << std::fixed << std::setprecision(1) << reduction << "%\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(1) << (static_cast<double>(brute_force_candidates) / bbox_candidates)
              << "×\n";
    std::cout << "===============================\n\n";

    EXPECT_GT(reduction, 0.0); // Should achieve some reduction
}
