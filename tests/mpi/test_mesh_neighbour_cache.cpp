// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Tests of the neighbour mesh cache used by the token exchanges of a mesh built
// from a reference mesh (see Mesh_base::exchange_with_neighbours).
//
// A rank whose cells are unchanged may replace its payload by a token, and its
// neighbours then keep the copy carried over from the reference mesh. These
// tests change the neighbour set of a rank whose cells do NOT change, and check
// that the mesh built from the reference holds exactly what a mesh built from
// scratch (always a full exchange) holds: same local mesh ids, same neighbour
// ranks and, for each neighbour, the same mesh ids, subdomain and union.
//
// The partitions are vertical strips of columns of a uniform 2D mesh. A strip
// is wider than the ghost reach, so only adjacent strips are neighbours, except
// for a strip of a single column, which its two sides see across.

#include <algorithm>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace mpi = boost::mpi;

namespace
{
    class MeshNeighbourCache : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim   = 2;
        static constexpr std::size_t level = 6;
        static constexpr int n_columns     = 1 << level;

        using mesh_config_t = samurai::mesh_config<dim>;
        using Mesh          = samurai::MRMesh<mesh_config_t>;
        using mesh_id_t     = typename Mesh::mesh_id_t;
        using cl_type       = typename Mesh::cl_type;

        /// Column range [start, end) owned by each rank.
        using partition_t = std::vector<std::pair<int, int>>;

        static auto config()
        {
            return samurai::mesh_config<dim>().min_level(level).max_level(level);
        }

        /// Strips of (almost) equal width, rank r at position r.
        static partition_t wide_strips(int size)
        {
            partition_t strips;
            for (int r = 0; r < size; ++r)
            {
                strips.emplace_back(r * n_columns / size, (r + 1) * n_columns / size);
            }
            return strips;
        }

        static cl_type local_cells(const partition_t& strips)
        {
            mpi::communicator world;
            const auto [start, end] = strips[static_cast<std::size_t>(world.rank())];

            cl_type cl;
            for (int j = 0; j < n_columns; ++j)
            {
                if (start < end)
                {
                    cl[level][{j}].add_interval({start, end});
                }
            }
            return cl;
        }

        static std::vector<int> neighbour_ranks(const Mesh& mesh)
        {
            std::vector<int> ranks;
            for (const auto& neighbour : mesh.mpi_neighbourhood())
            {
                ranks.push_back(neighbour.rank);
            }
            std::sort(ranks.begin(), ranks.end());
            return ranks;
        }

        static bool contains(const std::vector<int>& ranks, int rank)
        {
            return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
        }

        /// True when `mesh` (built from a reference) holds the same local and
        /// neighbour data as `scratch` (built from the same cells, without
        /// reference).
        static bool same_as_scratch(const Mesh& mesh, const Mesh& scratch)
        {
            for (std::size_t id = 0; id < Mesh::mesh_t::size; ++id)
            {
                if (!(mesh[static_cast<mesh_id_t>(id)] == scratch[static_cast<mesh_id_t>(id)]))
                {
                    return false;
                }
            }

            if (neighbour_ranks(mesh) != neighbour_ranks(scratch))
            {
                return false;
            }

            for (const auto& neighbour : mesh.mpi_neighbourhood())
            {
                const auto& expected = *std::find_if(scratch.mpi_neighbourhood().begin(),
                                                     scratch.mpi_neighbourhood().end(),
                                                     [&](const auto& n)
                                                     {
                                                         return n.rank == neighbour.rank;
                                                     });
                for (std::size_t id = 0; id < Mesh::mesh_t::size; ++id)
                {
                    if (!(neighbour.mesh[static_cast<mesh_id_t>(id)] == expected.mesh[static_cast<mesh_id_t>(id)]))
                    {
                        return false;
                    }
                }
                if (!(neighbour.mesh.subdomain() == expected.mesh.subdomain()) || !(neighbour.mesh.get_union() == expected.mesh.get_union()))
                {
                    return false;
                }
            }
            return true;
        }
    };

    // Rank 2 keeps its cells while ranks 0 and 1 swap their strips: rank 2 trades
    // neighbour 1 for neighbour 0, so its neighbour set keeps its size but not
    // its ranks. Rank 2 must not send a token to rank 0, which holds no copy of
    // rank 2's mesh.
    TEST_F(MeshNeighbourCache, same_size_different_neighbours)
    {
        mpi::communicator world;
        if (world.size() < 3)
        {
            GTEST_SKIP() << "needs at least 3 ranks";
        }

        const auto before = wide_strips(world.size());
        auto after        = before;
        std::swap(after[0], after[1]);

        const auto ref = samurai::mra::make_mesh(local_cells(before), config());
        Mesh mesh(local_cells(after), ref);
        const auto scratch = samurai::mra::make_mesh(local_cells(after), config());

        // Preconditions: the scenario is the one described above.
        const auto ref_neighbours = neighbour_ranks(ref);
        const auto neighbours     = neighbour_ranks(mesh);
        if (world.rank() == 2)
        {
            EXPECT_TRUE(ref[mesh_id_t::cells] == mesh[mesh_id_t::cells]);
            EXPECT_EQ(ref_neighbours.size(), neighbours.size());
            EXPECT_TRUE(contains(ref_neighbours, 1) && !contains(ref_neighbours, 0));
            EXPECT_TRUE(contains(neighbours, 0) && !contains(neighbours, 1));
        }

        EXPECT_TRUE_ALL_RANKS(same_as_scratch(mesh, scratch));
    }

    // Rank 1 shrinks to a single column, rank 0 takes the rest of its strip: rank
    // 2, whose cells do not change, gains neighbour 0. Then rank 1 widens back:
    // rank 2 loses neighbour 0. The second mesh is built from the first one, so
    // the carried-over cache goes through two generations.
    TEST_F(MeshNeighbourCache, neighbour_added_then_removed)
    {
        mpi::communicator world;
        if (world.size() < 3)
        {
            GTEST_SKIP() << "needs at least 3 ranks";
        }

        const auto wide = wide_strips(world.size());
        auto thin       = wide;
        thin[0].second  = wide[1].second - 1;
        thin[1].first   = thin[0].second;

        const auto ref = samurai::mra::make_mesh(local_cells(wide), config());

        // Neighbour added.
        Mesh added(local_cells(thin), ref);
        {
            const auto scratch = samurai::mra::make_mesh(local_cells(thin), config());
            if (world.rank() == 2)
            {
                EXPECT_TRUE(ref[mesh_id_t::cells] == added[mesh_id_t::cells]);
                EXPECT_FALSE(contains(neighbour_ranks(ref), 0));
                EXPECT_TRUE(contains(neighbour_ranks(added), 0));
            }
            EXPECT_TRUE_ALL_RANKS(same_as_scratch(added, scratch));
        }

        // Neighbour removed.
        Mesh removed(local_cells(wide), added);
        {
            const auto scratch = samurai::mra::make_mesh(local_cells(wide), config());
            if (world.rank() == 2)
            {
                EXPECT_TRUE(added[mesh_id_t::cells] == removed[mesh_id_t::cells]);
                EXPECT_FALSE(contains(neighbour_ranks(removed), 0));
            }
            EXPECT_TRUE_ALL_RANKS(same_as_scratch(removed, scratch));
        }
    }
}
