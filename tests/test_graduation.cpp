#include <algorithm>

#include <gtest/gtest.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>

namespace samurai
{
    TEST(graduation, dim_1)
    {
        constexpr size_t dim = 1;
        CellList<dim> cl;
        cl[0][{}].add_interval({1, 2});
        cl[5][{}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    TEST(graduation, dim_2)
    {
        constexpr size_t dim = 2;
        CellList<dim> cl;
        cl[0][{}].add_interval({1, 2});
        cl[5][{}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    TEST(graduation, dim_3)
    {
        constexpr size_t dim = 3;
        CellList<dim> cl;
        cl[0][{1, 1}].add_interval({1, 2});
        cl[5][{0, 0}].add_interval({0, 1});
        CellArray<dim> ca{cl};

        samurai::make_graduation(ca);
        EXPECT_TRUE(is_graduated(ca));
    }

    // Build the 1D domain pyramid [0, n) at `top_level`, projected down to level 0, as the
    // multi-level CellArray make_graduation expects (domain[l] = domain at level l).
    static CellArray<1> domain_pyramid_1d(int top_level, int n_cells_top)
    {
        CellList<1> cl;
        for (int l = top_level; l >= 0; --l)
        {
            const int n = n_cells_top >> (top_level - l);
            cl[static_cast<std::size_t>(l)][{}].add_interval({0, n});
        }
        return CellArray<1>{cl};
    }

    // Boundary contiguity exercised end to end through the production make_graduation (the
    // folded single pass). A level-4 run [14,16) touches the right physical boundary and a
    // level-3 cell [6,7) (fine [12,14)) sits one coarse cell inside it. For a wide stencil
    // that L3 cell must be refined to level 4 so the boundary run is thick enough. The mesh
    // tiles the domain (coverage == domain).
    static void expect_boundary_refined(int radius)
    {
        const std::array<bool, 1> is_periodic{false};

        struct DummyMesh
        {
        };

        std::vector<MPI_Subdomain<DummyMesh>> no_neighbours;

        CellList<1> cl;
        cl[4][{}].add_interval({14, 16});
        cl[3][{}].add_interval({0, 7});
        CellArray<1> ca{cl};

        make_graduation(ca, domain_pyramid_1d(4, 16), no_neighbours, is_periodic, size_t{1}, radius);

        EXPECT_TRUE(is_graduated(ca));

        // The boundary-adjacent level-3 cell [6,7) must have been refined to level 4, so
        // nothing remains at level 3 there.
        CellList<1> hole_cl;
        hole_cl[3][{}].add_interval({6, 7});
        const LevelCellArray<1> hole = CellArray<1>{hole_cl}[3];
        size_t remaining             = 0;
        intersection(ca[3], hole)
            .on(3)(
                [&](const auto&, const auto&)
                {
                    ++remaining;
                });
        EXPECT_EQ(remaining, 0u) << "the boundary-adjacent level-3 cell must be refined (radius=" << radius << ")";
    }

    TEST(graduation, boundary_contiguity_radius2)
    {
        expect_boundary_refined(2);
    }

    TEST(graduation, boundary_contiguity_radius3)
    {
        expect_boundary_refined(3);
    }

    static size_t count_intervals(size_t l, const LevelCellArray<1>& a)
    {
        size_t n = 0;
        self(a).on(l)(
            [&](const auto&, const auto&)
            {
                ++n;
            });
        return n;
    }

    static bool same_lca(size_t l, const LevelCellArray<1>& a, const LevelCellArray<1>& b)
    {
        return count_intervals(l, LevelCellArray<1>(difference(a, b).on(l))) == 0
            && count_intervals(l, LevelCellArray<1>(difference(b, a).on(l))) == 0;
    }

    // The fused single-pass boundary helpers (boundary_case1_cells / boundary_case2_cells)
    // must be INDEPENDENT of the MPI partition: a physical-boundary cell owned by a
    // NEIGHBOUR rank must drive exactly the same refinement as if this rank owned it. This
    // is the single-pass analogue of graduation.contiguous_boundary_partition_independent,
    // tested directly on the pure helpers (no real MPI needed) by feeding the boundary run
    // either as this rank's own cells or as a neighbour source.
    TEST(graduation, fused_boundary_case1_partition_independent)
    {
        CellList<1> domain_cl;
        domain_cl[4][{}].add_interval({0, 16});
        domain_cl[3][{}].add_interval({0, 8});
        CellArray<1> domain{domain_cl};
        const std::array<bool, 1> is_periodic{false};
        const int radius   = 3;
        const int n_contig = std::max(radius, 2 * (radius - 2)); // = 3

        CellList<1> bcl;
        bcl[4][{}].add_interval({14, 16}); // level-4 run on the right physical boundary
        const LevelCellArray<1> boundary = CellArray<1>{bcl}[4];
        const LevelCellArray<1> empty4;

        const LevelCellArray<1> all_local = boundary_case1_cells(size_t{4},
                                                                 n_contig,
                                                                 domain,
                                                                 is_periodic,
                                                                 std::vector<LevelCellArray<1>>{boundary});
        const LevelCellArray<1> split     = boundary_case1_cells(size_t{4},
                                                             n_contig,
                                                             domain,
                                                             is_periodic,
                                                             std::vector<LevelCellArray<1>>{empty4, boundary});

        EXPECT_GT(count_intervals(4, all_local), 0u) << "the boundary run must force an inner refinement";
        EXPECT_TRUE(same_lca(4, all_local, split)) << "case-1 refinement must not depend on which rank owns the boundary cell";
    }

    TEST(graduation, fused_boundary_case2_partition_independent)
    {
        CellList<1> domain_cl;
        domain_cl[4][{}].add_interval({0, 16});
        domain_cl[3][{}].add_interval({0, 8});
        CellArray<1> domain{domain_cl};
        const std::array<bool, 1> is_periodic{false};
        const int radius = 3;

        // A level-4 (fine) cell sitting one coarse cell inside the right boundary, so case 2
        // must pull the fine level out to the boundary. Provided either locally or by a neighbour.
        CellList<1> fcl;
        fcl[4][{}].add_interval({12, 14});
        const LevelCellArray<1> fine = CellArray<1>{fcl}[4];
        const LevelCellArray<1> empty4;

        const LevelCellArray<1> all_local = boundary_case2_cells(size_t{4}, radius, domain, is_periodic, std::vector<LevelCellArray<1>>{fine});
        const LevelCellArray<1> split = boundary_case2_cells(size_t{4},
                                                             radius,
                                                             domain,
                                                             is_periodic,
                                                             std::vector<LevelCellArray<1>>{empty4, fine});

        EXPECT_TRUE(same_lca(4, all_local, split)) << "case-2 refinement must not depend on which rank owns the fine cell";
    }
}
