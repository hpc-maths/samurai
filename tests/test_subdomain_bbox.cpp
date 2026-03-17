// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

/**
 * Unit tests for bounding box-based neighbor discovery.
 *
 * These tests validate the bbox computation and intersection logic
 * in 1D, 2D, 3D, and 4D without requiring MPI.
 */
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/mpi/subdomain_bbox.hpp>

namespace samurai
{
    // Fixture for dimension-parametrized tests
    template <typename T>
    class SubdomainBBoxTest : public ::testing::Test
    {
    };

    // Define test types for 1D, 2D, 3D, 4D
    using dim_test_types = ::testing::Types<std::integral_constant<std::size_t, 1>,
                                            std::integral_constant<std::size_t, 2>,
                                            std::integral_constant<std::size_t, 3>,
                                            std::integral_constant<std::size_t, 4>>;

    TYPED_TEST_SUITE(SubdomainBBoxTest, dim_test_types, );

    // Test 1: Basic bbox computation from cell array
    TYPED_TEST(SubdomainBBoxTest, BasicComputation)
    {
        constexpr std::size_t dim = TypeParam::value;
        using point_t             = typename Box<double, dim>::point_t;

        constexpr std::size_t level = 3;
        LevelCellList<dim> lcl(level);

        // Add a single interval along first dimension
        if constexpr (dim == 1)
        {
            lcl[{}].add_interval({0, 8});
        }
        else if constexpr (dim == 2)
        {
            lcl[{0}].add_interval({0, 8});
        }
        else if constexpr (dim == 3)
        {
            lcl[{0, 0}].add_interval({0, 8});
        }
        else // dim == 4
        {
            lcl[{0, 0, 0}].add_interval({0, 8});
        }

        LevelCellArray<dim> lca(lcl);

        // Compute bounding box
        auto bbox = mpi_neighbor::compute_subdomain_bbox(lca);

        // Verify bbox encompasses the domain
        // At level 3, cell size is 1/8 = 0.125
        // Interval [0, 8) spans [0, 1) in physical space
        point_t expected_min;
        expected_min.fill(0.0);
        point_t expected_max;
        expected_max.fill(0.125);
        expected_max[0] = 1.0; // First dimension spans [0, 1)

        for (std::size_t d = 0; d < dim; ++d)
        {
            EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[d], expected_min[d]) << "Dimension " << d;
            EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[d], expected_max[d]) << "Dimension " << d;
        }

        // Verify cell_length
        EXPECT_DOUBLE_EQ(bbox.cell_length, 1.0 / (1 << level));
    }

    // Test 2: Bbox with custom origin and scaling
    TYPED_TEST(SubdomainBBoxTest, OriginAndScaling)
    {
        constexpr std::size_t dim = TypeParam::value;
        using point_t             = typename Box<double, dim>::point_t;

        constexpr std::size_t level = 2;
        LevelCellList<dim> lcl(level);

        // Add interval
        if constexpr (dim == 1)
        {
            lcl[{}].add_interval({0, 4});
        }
        else if constexpr (dim == 2)
        {
            lcl[{0}].add_interval({0, 4});
        }
        else if constexpr (dim == 3)
        {
            lcl[{0, 0}].add_interval({0, 4});
        }
        else
        {
            lcl[{0, 0, 0}].add_interval({0, 4});
        }

        LevelCellArray<dim> lca(lcl);

        // Test with custom origin
        point_t origin;
        origin.fill(-1.0);
        lca.set_origin_point(origin);

        auto bbox = mpi_neighbor::compute_subdomain_bbox(lca);

        // Verify origin is respected
        for (std::size_t d = 0; d < dim; ++d)
        {
            EXPECT_LE(bbox.bbox.min_corner()[d], origin[d] + 1e-10) << "Dimension " << d;
        }

        // Test with custom scaling
        lca.set_scaling_factor(0.5);
        bbox = mpi_neighbor::compute_subdomain_bbox(lca);

        // At level 2 with scaling 0.5: cell size = 0.5/4 = 0.125
        // 4 cells span 4 * 0.125 = 0.5
        double expected_span_x = 0.5;
        EXPECT_NEAR(bbox.bbox.max_corner()[0] - bbox.bbox.min_corner()[0], expected_span_x, 1e-10);
    }

    // Test 3: BBox intersection detection
    TYPED_TEST(SubdomainBBoxTest, IntersectionDetection)
    {
        constexpr std::size_t dim = TypeParam::value;
        using BBox                = mpi_neighbor::SubdomainBoundingBox<dim>;
        using point_t             = typename Box<double, dim>::point_t;

        // Create two adjacent bboxes along first dimension
        point_t min1, max1, min2, max2, min3, max3;
        min1.fill(0.0);
        max1.fill(0.5);
        min2.fill(0.5);
        max2.fill(1.0);
        min3.fill(0.0);
        max3.fill(0.0);

        BBox bbox1(0, 0.125, Box<double, dim>(min1, max1));
        BBox bbox2(1, 0.125, Box<double, dim>(min2, max2));
        BBox bbox3(2, 0., Box<double, dim>(min3, max3)); // Empty bbox for testing

        // They should be neighbors (they touch)
        EXPECT_TRUE(bbox1.could_be_neighbor(bbox2));
        EXPECT_TRUE(bbox2.could_be_neighbor(bbox1));
        EXPECT_FALSE(bbox3.could_be_neighbor(bbox1)); // Empty bbox should not be neighbor
        EXPECT_FALSE(bbox3.could_be_neighbor(bbox2)); // Empty bbox should not be neighbor
        EXPECT_FALSE(bbox1.could_be_neighbor(bbox3)); // Empty bbox should not be neighbor
        EXPECT_FALSE(bbox2.could_be_neighbor(bbox3)); // Empty bbox should not be neighbor
    }

    // Test 4: Non-intersecting bboxes
    TYPED_TEST(SubdomainBBoxTest, NonIntersecting)
    {
        constexpr std::size_t dim = TypeParam::value;
        using BBox                = mpi_neighbor::SubdomainBoundingBox<dim>;
        using point_t             = typename Box<double, dim>::point_t;

        // Create two distant bboxes
        point_t min1, max1, min2, max2;
        min1.fill(0.0);
        max1.fill(0.3);
        min2.fill(0.7);
        max2.fill(1.0);

        BBox bbox1(0, 0.125, Box<double, dim>(min1, max1));
        BBox bbox2(1, 0.125, Box<double, dim>(min2, max2));

        // They should NOT be neighbors (too far apart)
        EXPECT_FALSE(bbox1.could_be_neighbor(bbox2));
        EXPECT_FALSE(bbox2.could_be_neighbor(bbox1));
    }

    // Test 5: Expansion factor influence
    TYPED_TEST(SubdomainBBoxTest, ExpansionFactor)
    {
        constexpr std::size_t dim = TypeParam::value;
        using BBox                = mpi_neighbor::SubdomainBoundingBox<dim>;
        using point_t             = typename Box<double, dim>::point_t;

        // Test that expansion uses min(cell_length, other.cell_length)
        point_t min1, max1, min2, max2;
        min1.fill(0.0);
        max1.fill(0.4);
        min2.fill(0.6);
        max2.fill(1.0);

        // Case 1: Same cell lengths - gap exactly equals cell_length
        BBox bbox1a(0, 0.2, Box<double, dim>(min1, max1)); // cell_length = 0.2
        BBox bbox2a(1, 0.2, Box<double, dim>(min2, max2)); // cell_length = 0.2, gap = 0.2
        EXPECT_TRUE(bbox1a.could_be_neighbor(bbox2a));     // Expansion = min(0.2, 0.2) = 0.2, should touch

        // Case 2: Different cell lengths - expansion uses minimum
        BBox bbox1b(0, 0.1, Box<double, dim>(min1, max1)); // cell_length = 0.1
        BBox bbox2b(1, 0.3, Box<double, dim>(min2, max2)); // cell_length = 0.3, gap = 0.2
        EXPECT_FALSE(bbox1b.could_be_neighbor(bbox2b));    // Expansion = min(0.1, 0.3) = 0.1, gap too large

        // Case 3: Larger cell length reaches across gap
        BBox bbox1c(0, 0.25, Box<double, dim>(min1, max1)); // cell_length = 0.25
        BBox bbox2c(1, 0.25, Box<double, dim>(min2, max2)); // cell_length = 0.25, gap = 0.2
        EXPECT_TRUE(bbox1c.could_be_neighbor(bbox2c));      // Expansion = min(0.25, 0.25) = 0.25 > gap
    }

    // Test 6: Empty subdomain handling
    TYPED_TEST(SubdomainBBoxTest, EmptySubdomain)
    {
        constexpr std::size_t dim = TypeParam::value;

        constexpr std::size_t level = 3;
        LevelCellList<dim> lcl(level);
        // Don't add any intervals

        LevelCellArray<dim> lca(lcl);

        // Compute bbox of empty subdomain
        auto bbox = mpi_neighbor::compute_subdomain_bbox(lca);

        // Should return degenerate bbox
        EXPECT_EQ(bbox.rank, -1);
        EXPECT_DOUBLE_EQ(bbox.cell_length, 0.0);

        for (std::size_t d = 0; d < dim; ++d)
        {
            EXPECT_DOUBLE_EQ(bbox.bbox.min_corner()[d], 0.0);
            EXPECT_DOUBLE_EQ(bbox.bbox.max_corner()[d], 0.0);
        }
    }

    // Test 7: Complex interval pattern
    TYPED_TEST(SubdomainBBoxTest, ComplexPattern)
    {
        constexpr std::size_t dim = TypeParam::value;

        constexpr std::size_t level = 2;
        LevelCellList<dim> lcl(level);

        // Add multiple intervals to create non-convex pattern
        if constexpr (dim == 1)
        {
            lcl[{}].add_interval({0, 2});
            lcl[{}].add_interval({3, 4}); // Gap at index 2
        }
        else if constexpr (dim == 2)
        {
            lcl[{0}].add_interval({0, 2});
            lcl[{0}].add_interval({3, 4}); // Gap at x=2
            lcl[{1}].add_interval({1, 3});
        }
        else if constexpr (dim == 3)
        {
            lcl[{0, 0}].add_interval({0, 2});
            lcl[{1, 0}].add_interval({1, 3});
            lcl[{0, 1}].add_interval({2, 4});
        }
        else // dim == 4
        {
            lcl[{0, 0, 0}].add_interval({0, 2});
            lcl[{1, 0, 0}].add_interval({1, 3});
            lcl[{0, 1, 0}].add_interval({2, 4});
        }

        LevelCellArray<dim> lca(lcl);
        auto bbox = mpi_neighbor::compute_subdomain_bbox(lca);

        // Bbox should be convex hull containing all intervals
        EXPECT_GT(bbox.cell_length, 0.0);

        // Min corner should be at origin
        EXPECT_LE(bbox.bbox.min_corner()[0], 0.0 + 1e-10);

        // Max corner should contain rightmost cell
        EXPECT_GE(bbox.bbox.max_corner()[0], 0.75); // Cell [3,4) at level 2
    }

} // namespace samurai

// Demonstration test showing candidate reduction in 2D
TEST(SubdomainBBox, CandidateReductionDemo2D)
{
    constexpr std::size_t dim = 2;
    using BBox                = samurai::mpi_neighbor::SubdomainBoundingBox<dim>;

    // Simulate processes arranged in a grid
    constexpr int grid_size     = 100;
    constexpr int num_processes = grid_size * grid_size;

    std::vector<BBox> all_bboxes;

    // Create bboxes for grid decomposition
    double cell_size = 1.0 / grid_size;
    for (int i = 0; i < grid_size; ++i)
    {
        for (int j = 0; j < grid_size; ++j)
        {
            int rank     = (i * grid_size) + j;
            double x_min = static_cast<double>(i) * cell_size;
            double x_max = static_cast<double>(i + 1) * cell_size;
            double y_min = static_cast<double>(j) * cell_size;
            double y_max = static_cast<double>(j + 1) * cell_size;

            all_bboxes.emplace_back(rank, cell_size, samurai::Box<double, dim>({x_min, y_min}, {x_max, y_max}));
        }
    }

    // Test process in center
    int test_rank       = grid_size / 2 * grid_size + grid_size / 2;
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
    int brute_force_candidates = num_processes - 1;

    // Bbox screening should find only ~8 neighbors (Moore neighborhood)
    EXPECT_LT(bbox_candidates, brute_force_candidates);
    EXPECT_LE(bbox_candidates, 8);

    double reduction = 100.0 * (1.0 - static_cast<double>(bbox_candidates) / brute_force_candidates);

    std::cout << "\n=== 2D Candidate Reduction Demo ===\n";
    std::cout << fmt::format("Processes: {} ({}x{} grid)\n", num_processes, grid_size, grid_size);
    std::cout << "Test rank: " << test_rank << " (center process)\n";
    std::cout << "Brute force candidates: " << brute_force_candidates << "\n";
    std::cout << "BBox candidates: " << bbox_candidates << "\n";
    std::cout << fmt::format("Reduction: {:.1f}%\n", reduction);
    std::cout << fmt::format("Speedup: {:.1f}×\n", static_cast<double>(brute_force_candidates) / bbox_candidates);
    std::cout << "====================================\n\n";

    EXPECT_GT(reduction, 99.0); // Should achieve >99% reduction
}

// Demonstration test for 3D
TEST(SubdomainBBox, CandidateReductionDemo3D)
{
    constexpr std::size_t dim = 3;
    using BBox                = samurai::mpi_neighbor::SubdomainBoundingBox<dim>;

    // Simulate processes arranged in a 3D grid
    constexpr int grid_size     = 10;
    constexpr int num_processes = grid_size * grid_size * grid_size;

    std::vector<BBox> all_bboxes;

    // Create bboxes for 3D grid decomposition
    double cell_size = 1.0 / grid_size;
    for (int i = 0; i < grid_size; ++i)
    {
        for (int j = 0; j < grid_size; ++j)
        {
            for (int k = 0; k < grid_size; ++k)
            {
                int rank     = (i * grid_size * grid_size) + (j * grid_size) + k;
                double x_min = static_cast<double>(i) * cell_size;
                double x_max = static_cast<double>(i + 1) * cell_size;
                double y_min = static_cast<double>(j) * cell_size;
                double y_max = static_cast<double>(j + 1) * cell_size;
                double z_min = static_cast<double>(k) * cell_size;
                double z_max = static_cast<double>(k + 1) * cell_size;

                all_bboxes.emplace_back(rank, cell_size, samurai::Box<double, dim>({x_min, y_min, z_min}, {x_max, y_max, z_max}));
            }
        }
    }

    // Test process in center
    int test_rank       = (grid_size / 2) * grid_size * grid_size + (grid_size / 2) * grid_size + (grid_size / 2);
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
    int brute_force_candidates = num_processes - 1;

    // Bbox screening should find only ~26 neighbors (3D Moore neighborhood)
    EXPECT_LT(bbox_candidates, brute_force_candidates);
    EXPECT_LE(bbox_candidates, 26);

    double reduction = 100.0 * (1.0 - static_cast<double>(bbox_candidates) / brute_force_candidates);

    std::cout << "\n=== 3D Candidate Reduction Demo ===\n";
    std::cout << fmt::format("Processes: {} ({}x{}x{} grid)\n", num_processes, grid_size, grid_size, grid_size);
    std::cout << "Test rank: " << test_rank << " (center process)\n";
    std::cout << "Brute force candidates: " << brute_force_candidates << "\n";
    std::cout << "BBox candidates: " << bbox_candidates << "\n";
    std::cout << fmt::format("Reduction: {:.1f}%\n", reduction);
    std::cout << fmt::format("Speedup: {:.1f}×\n", static_cast<double>(brute_force_candidates) / bbox_candidates);
    std::cout << "====================================\n\n";

    EXPECT_GT(reduction, 97.0); // Should achieve >97% reduction
}
