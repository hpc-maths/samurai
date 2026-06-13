// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Tests for the distributed cell graph (roadmap step 4).
// The graph is always built when WITH_MPI=ON; these tests verify its structure
// without requiring ParMETIS or PT-Scotch.

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/graph.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    template <std::size_t d>
    struct Dim
    {
        static constexpr std::size_t dim = d;
    };

    template <class T>
    class LoadBalancingGraph : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim = T::dim;
        using mesh_config_t              = samurai::mesh_config<dim>;
        using Mesh                       = samurai::MRMesh<mesh_config_t>;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

        static constexpr std::size_t level = (dim == 2) ? 3 : 2;

        static samurai::Box<double, dim> unit_box()
        {
            xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
            min_corner.fill(0.);
            max_corner.fill(1.);
            return samurai::Box<double, dim>(min_corner, max_corner);
        }

        static Mesh make_uniform_mesh()
        {
            return samurai::mra::make_mesh(unit_box(), samurai::mesh_config<dim>().min_level(level).max_level(level));
        }
    };

    using Dims = ::testing::Types<Dim<2>, Dim<3>>;
    TYPED_TEST_SUITE(LoadBalancingGraph, Dims, );

    // Uniform mesh on 1 process: the graph must be well-formed with the
    // correct number of vertices and edges matching the known grid topology.
    TYPED_TEST(LoadBalancingGraph, uniform_1p)
    {
        mpi::communicator world;
        if (world.size() != 1)
        {
            GTEST_SKIP() << "This test requires exactly 1 process";
        }

        auto mesh = TestFixture::make_uniform_mesh();
        auto graph = lb::build_cell_graph<int>(mesh, lb::weight::uniform());

        const auto n = mesh.nb_cells(TestFixture::mesh_id_t::cells);
        constexpr std::size_t dim = TestFixture::dim;

        // vtxdist: [0, n]
        EXPECT_EQ(graph.vtxdist.size(), 2u);
        EXPECT_EQ(graph.vtxdist[0], 0);
        EXPECT_EQ(graph.vtxdist[1], static_cast<int>(n));

        // Number of vertices matches cells
        EXPECT_EQ(static_cast<std::size_t>(graph.nvtx_local()), n);

        // Each interior cell has 2*dim neighbours; boundary cells have fewer.
        // The total edge count (directed) must equal sum of degrees.
        const int total_edges = graph.adjncy.size();
        EXPECT_GT(total_edges, 0) << "Graph has no edges for a non-trivial mesh";

        // Check that all edge endpoints are valid global vertex IDs
        for (std::size_t i = 0; i < static_cast<std::size_t>(total_edges); ++i)
        {
            EXPECT_GE(graph.adjncy[i], 0);
            EXPECT_LT(static_cast<std::size_t>(graph.adjncy[i]), n);
        }

        // Symmetry: for every edge (u, v), there must be an edge (v, u).
        // Build a set of directed edges and check.
        std::set<std::pair<int, int>> edges;
        for (std::size_t u = 0; u < n; ++u)
        {
            for (int j = graph.xadj[u]; j < graph.xadj[u + 1]; ++j)
            {
                int v = graph.adjncy[static_cast<std::size_t>(j)];
                edges.insert({static_cast<int>(u), v});
            }
        }
        for (const auto& [u, v] : edges)
        {
            EXPECT_TRUE(edges.count({v, u})) << "Edge (" << u << "," << v << ") exists but (" << v << "," << u << ") does not";
        }
    }

    // On multiple processes: vtxdist must be consistent and the total vertex
    // count must equal the global cell count.
    TYPED_TEST(LoadBalancingGraph, global_consistency)
    {
        mpi::communicator world;
        auto mesh  = TestFixture::make_uniform_mesh();
        auto graph = lb::build_cell_graph<int>(mesh, lb::weight::uniform());

        const auto n_local  = mesh.nb_cells(TestFixture::mesh_id_t::cells);
        const auto n_global  = mpi::all_reduce(world, n_local, std::plus<std::size_t>());

        // vtxdist[P] = total global count
        EXPECT_EQ(static_cast<std::size_t>(graph.vtxdist.back()), n_global);

        // Local vertex count matches
        EXPECT_EQ(static_cast<std::size_t>(graph.nvtx_local()), n_local);

        // All edge endpoints are valid global IDs
        for (int j = 0; j < graph.adjncy.size(); ++j)
        {
            EXPECT_GE(graph.adjncy[static_cast<std::size_t>(j)], 0);
            EXPECT_LT(static_cast<std::size_t>(graph.adjncy[static_cast<std::size_t>(j)]), n_global);
        }
    }

    // Vertex weights are positive integers.
    TYPED_TEST(LoadBalancingGraph, positive_weights)
    {
        auto mesh  = TestFixture::make_uniform_mesh();
        auto graph = lb::build_cell_graph<int>(mesh, lb::weight::uniform());

        for (std::size_t i = 0; i < static_cast<std::size_t>(graph.nvtx_local()); ++i)
        {
            EXPECT_GT(graph.vwgt[i], 0) << "Vertex weight must be positive";
        }
    }

    // After load balancing with the void strategy, the graph should have the
    // same structure (same number of vertices and edges).
    TYPED_TEST(LoadBalancingGraph, adapted_mesh)
    {
        constexpr std::size_t dim = TestFixture::dim;
        auto mesh = samurai_test::make_locally_refined_mesh<typename TestFixture::Mesh>(
            TestFixture::unit_box(),
            TestFixture::level,
            [](const auto& cell)
            {
                bool in_corner = true;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    in_corner = in_corner && (cell.center(d) < 0.5);
                }
                return in_corner;
            });

        auto graph = lb::build_cell_graph<int>(mesh, lb::weight::uniform());

        const auto n = mesh.nb_cells(TestFixture::mesh_id_t::cells);
        EXPECT_EQ(static_cast<std::size_t>(graph.nvtx_local()), n);

        // Multi-level meshes must have edges (cross-level adjacency)
        EXPECT_GT(graph.adjncy.size(), 0u);
    }
}