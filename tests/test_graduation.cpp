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

#ifdef SAMURAI_WITH_MPI
    // Non-regression test for the partition-dependence bug fixed alongside this test:
    // list_interval_to_refine_for_contiguous_boundary_cells (active only when
    // max_stencil_radius > 1) used to look for physical-boundary cells in the local
    // `ca` only, so a boundary cell owned by a neighbour rank was invisible and the
    // refinement it must trigger on the neighbouring inner cells depended on the MPI
    // partition. The fix takes boundary cells from `ca` AND from every mesh in
    // `mpi_meshes`, while still refining only the local rank's own cells.
    //
    // Domain: x in [0, 16) at level 4. A single coarse cell (level 3, index 6, i.e.
    // fine x in [12, 14)) sits close enough to the right physical boundary
    // (x = 16) that it must be refined to level 4 for a scheme with
    // max_stencil_radius = 3, regardless of which rank owns the level-4 boundary
    // cells (fine x in [14, 16)) that trigger this refinement.
    TEST(graduation, contiguous_boundary_partition_independent)
    {
        constexpr size_t dim         = 1;
        constexpr int stencil_radius = 3;

        using ca_type    = CellArray<dim>;
        using interval_t = typename ca_type::interval_t;
        using coord_type = typename ca_type::lca_type::coord_type;
        using out_t      = std::array<ArrayOfIntervalAndPoint<interval_t, coord_type>, ca_type::max_size>;

        CellList<dim> domain_cl;
        domain_cl[4][{}].add_interval({0, 16});
        LevelCellArray<dim> domain(domain_cl[4]);

        const std::array<bool, dim> is_periodic{false};

        // Scenario A: this rank owns everything, including the boundary cells.
        CellList<dim> cl_local;
        cl_local[4][{}].add_interval({14, 16});
        cl_local[3][{}].add_interval({6, 7});
        ca_type ca_all_local{cl_local};

        out_t out_single_rank{};
        std::vector<ca_type> no_neighbours;
        list_interval_to_refine_for_contiguous_boundary_cells(stencil_radius, ca_all_local, domain, no_neighbours, is_periodic, out_single_rank);

        // Scenario B: this rank owns only the interior coarse cell; a neighbour rank
        // owns the boundary cells (as can happen after load balancing moves a
        // subdomain boundary into an active region).
        CellList<dim> cl_interior_only;
        cl_interior_only[3][{}].add_interval({6, 7});
        ca_type ca_without_boundary{cl_interior_only};

        CellList<dim> cl_neighbour;
        cl_neighbour[4][{}].add_interval({14, 16});
        ca_type neighbour_ca{cl_neighbour};
        std::vector<ca_type> mpi_meshes{neighbour_ca};

        out_t out_split_rank{};
        list_interval_to_refine_for_contiguous_boundary_cells(stencil_radius, ca_without_boundary, domain, mpi_meshes, is_periodic, out_split_rank);

        // The coarse cell must be flagged for refinement in both cases, identically,
        // whether the boundary cells that trigger it are local or owned by a
        // neighbour.
        ASSERT_EQ(out_single_rank[3].size(), 1u);
        ASSERT_EQ(out_split_rank[3].size(), 1u);
        EXPECT_EQ(out_single_rank[3].get_interval(0), out_split_rank[3].get_interval(0));
    }
#endif // SAMURAI_WITH_MPI
}
