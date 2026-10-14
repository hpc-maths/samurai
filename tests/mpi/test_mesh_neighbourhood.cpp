// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// MPI neighbourhood of a mesh rebuilt from a reference mesh.
//
// A mesh built from a reference mesh (Mesh(cells, ref_mesh), as the adaptation
// does at every iteration) may reuse the reference neighbourhood instead of
// running the neighbourhood discovery again. That reuse is valid only when every
// input of the discovery is unchanged: the subdomains, but also the ghost reach
// and the periodicity. The latter two can change after the reference mesh was
// built (the periodicity is set through mesh.cfg(), see the
// burgers_os_2d_mpi demo), so the neighbourhood of the rebuilt mesh must match
// the one of a mesh built with the final configuration from the start.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace mpi = boost::mpi;

namespace
{
    class mesh_neighbourhood : public samurai_test::MpiTest
    {
    };

    constexpr std::size_t dim   = 2;
    constexpr std::size_t level = 4;

    auto make_config(bool periodic)
    {
        // A single level: the ghost reach never changes, so only the
        // periodicity can make the neighbourhoods differ.
        return samurai::mesh_config<dim>().min_level(level).max_level(level).periodic(periodic);
    }

    // One unit square per rank, side by side along x: with 3 ranks or more, the
    // first and the last rank are neighbours through the periodic boundary only.
    samurai::CellArray<dim> strip_cells()
    {
        mpi::communicator world;
        const auto r = static_cast<double>(world.rank());

        const xt::xtensor_fixed<double, xt::xshape<dim>> origin{0., 0.};
        samurai::CellArray<dim> cells;
        for (std::size_t l = 0; l < cells.max_size; ++l)
        {
            cells[l].set_origin_point(origin);
            cells[l].set_scaling_factor(1.);
        }
        cells[level] = {level, samurai::Box<double, dim>({r, 0.}, {r + 1., 1.}), origin, 0.05, 1.};
        return cells;
    }

    template <class Mesh>
    std::vector<int> neighbour_ranks(const Mesh& mesh)
    {
        std::vector<int> ranks;
        for (const auto& neighbour : mesh.mpi_neighbourhood())
        {
            ranks.push_back(neighbour.rank);
        }
        std::sort(ranks.begin(), ranks.end());
        return ranks;
    }

    TEST_F(mesh_neighbourhood, periodicity_set_after_construction)
    {
        auto cells = strip_cells();

        auto expected = samurai::mra::make_mesh(cells, make_config(true));

        auto mesh = samurai::mra::make_mesh(cells, make_config(false));
        mesh.cfg().periodic(true);
        mesh = {cells, mesh};

        EXPECT_TRUE_ALL_RANKS(neighbour_ranks(mesh) == neighbour_ranks(expected));

        // A further rebuild with the same cells, as the adaptation does, keeps it.
        decltype(mesh) rebuilt{cells, mesh};
        EXPECT_TRUE_ALL_RANKS(neighbour_ranks(rebuilt) == neighbour_ranks(expected));
    }
}
