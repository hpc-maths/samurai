#include <algorithm>
#include <stdexcept>

#include <gtest/gtest.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

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

    // Corner coupling (2D). At a domain corner two physical boundary faces meet, so the
    // boundary run extended along one face can create cells that then need extension along
    // the perpendicular face. This reproduces the exact mesh (a central level-5 block inside
    // a level-4 frame, radius 3) that exposed a fused single-pass under-refinement at the
    // corners: the fine level must be pulled into the corner, which needs the per-level
    // extension to be iterated until the corner region stabilises.
    TEST(graduation, boundary_contiguity_corner_2d)
    {
        constexpr size_t dim = 2;

        // domain pyramid: [0, 16)^2 at level 4 (= [0, 32)^2 at level 5), down to level 0.
        CellList<dim> domain_cl;
        for (int l = 5; l >= 0; --l)
        {
            const int n = 32 >> (5 - l);
            for (int j = 0; j < n; ++j)
            {
                domain_cl[static_cast<std::size_t>(l)][{j}].add_interval({0, n});
            }
        }
        CellArray<dim> domain{domain_cl};

        // Tiling input: central level-5 block [4,28)^2 (= [2,14)^2 at level 4) surrounded by
        // a level-4 frame filling [0,16)^2.
        CellList<dim> cl;
        for (int j = 0; j < 16; ++j) // level-4 rows
        {
            if (j < 2 || j >= 14)
            {
                cl[4][{j}].add_interval({0, 16});
            }
            else
            {
                cl[4][{j}].add_interval({0, 2});
                cl[4][{j}].add_interval({14, 16});
            }
        }
        for (int j = 4; j < 28; ++j) // level-5 rows of the central block
        {
            cl[5][{j}].add_interval({4, 28});
        }
        CellArray<dim> ca{cl};

        const std::array<bool, dim> is_periodic{false, false};

        struct DummyMesh
        {
        };

        std::vector<MPI_Subdomain<DummyMesh>> no_neighbours;
        make_graduation(ca, domain, no_neighbours, is_periodic, size_t{1}, 3);

        EXPECT_TRUE(is_graduated(ca));

        // The fine level (5) must be pulled into the bottom-left corner: level 4 must NOT
        // remain at the very corner cell (x=[0,2)@L4, row 0). The buggy single pass left L4
        // there instead of refining to L5.
        CellList<dim> corner_cl;
        corner_cl[4][{0}].add_interval({0, 2});
        const LevelCellArray<dim> corner = CellArray<dim>{corner_cl}[4];
        size_t remaining                 = 0;
        intersection(ca[4], corner)
            .on(4)(
                [&](const auto&, const auto&)
                {
                    ++remaining;
                });
        EXPECT_EQ(remaining, 0u) << "the fine level must be pulled into the domain corner (corner coupling)";
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

    namespace
    {
        // samurai::graduation() (as opposed to make_graduation() above) reads the graduation width
        // from mesh.cfg() and dispatches it through dispatch_static<1, 9>.
        //
        // The tag is built on holder(mesh) so that it owns its mesh: a plain field only points to
        // the mesh it was built on, which here is a local of this helper.
        auto make_graduation_width_test_tag(std::size_t graduation_width)
        {
            static constexpr std::size_t dim = 1;
            using box_t                      = Box<double, dim>;

            auto mesh_cfg = mesh_config<dim>().min_level(1).max_level(3).graduation_width(graduation_width);
            auto mesh     = mra::make_mesh(box_t{xt::zeros<double>({dim}), xt::ones<double>({dim})}, mesh_cfg);
            auto m        = holder(mesh);
            auto tag      = make_scalar_field<int>("tag", m);
            tag.fill(static_cast<int>(CellFlag::keep));
            return tag;
        }
    }

    // A graduation width of 0 means "no graduation constraint" and must not throw.
    TEST(graduation, width_zero_is_a_noop)
    {
        auto tag = make_graduation_width_test_tag(0);
        const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil{{1}, {-1}};
        EXPECT_NO_THROW(samurai::graduation(tag, stencil));
    }

    // A width beyond what dispatch_static<1, 9> is instantiated for must fail loudly
    // (std::out_of_range) instead of silently skipping the graduation step.
    TEST(graduation, width_above_dispatch_range_throws)
    {
        auto tag = make_graduation_width_test_tag(15);
        const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil{{1}, {-1}};
        EXPECT_THROW(samurai::graduation(tag, stencil), std::out_of_range);
    }
}
