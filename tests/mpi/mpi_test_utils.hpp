// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <gtest/gtest.h>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/subset/node.hpp>

/**
 * An MPI test passes only if it passes on EVERY rank. A bare EXPECT_* that
 * fails on a single rank would let the other ranks report success; this macro
 * reduces the condition globally so the failure is visible on all ranks.
 */
#define EXPECT_TRUE_ALL_RANKS(cond)                                                                                                      \
    do                                                                                                                                   \
    {                                                                                                                                    \
        const bool local_ok_ = static_cast<bool>(cond);                                                                                  \
        boost::mpi::communicator world_;                                                                                                 \
        const bool global_ok_ = boost::mpi::all_reduce(world_, local_ok_, std::logical_and<bool>());                                     \
        EXPECT_TRUE(global_ok_) << "condition '" << #cond << "' failed on at least one rank (locally: " << (local_ok_ ? "ok" : "FAILED") \
                                << ", rank " << world_.rank() << ")";                                                                    \
    } while (0)

namespace samurai_test
{
    /**
     * Base fixture for MPI tests: after each test, synchronize the ranks and
     * verify that no MPI message was left pending. All samurai exchanges share
     * the tag == receiver rank, so an orphan message would silently
     * desynchronize the exchanges of the NEXT test; failing here pinpoints the
     * guilty test instead.
     */
    class MpiTest : public ::testing::Test
    {
      protected:

        void TearDown() override
        {
            boost::mpi::communicator world;
            world.barrier();
            // probe in a short loop: an in-flight message may not have landed
            // yet right after the barrier (MPI progression is not guaranteed
            // without communication activity)
            bool clean = true;
            for (int attempt = 0; attempt < 20; ++attempt)
            {
                auto status = world.iprobe(boost::mpi::any_source, boost::mpi::any_tag);
                if (status.has_value())
                {
                    clean = false;
                    std::cerr << "[rank " << world.rank() << "] pending MPI message after test: source " << status->source() << " tag "
                              << status->tag() << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            EXPECT_TRUE_ALL_RANKS(clean);
        }
    };

    /// Test-only strategy: destination rank computed by a lambda(cell, rank, size).
    /// Lets each test inject hand-crafted flags without writing a real partitioner.
    template <class F>
    struct LambdaStrategy
    {
        F f;

        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& /*weight*/) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            boost::mpi::communicator world;
            auto flags = samurai::make_scalar_field<int>("lb_flags", mesh);
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       flags[cell] = f(cell, world.rank(), world.size());
                                   });
            return flags;
        }

        std::string name() const
        {
            return "test-lambda";
        }
    };

    /**
     * Two-level test mesh: uniform mesh at `level`, then the cells selected
     * by `refine_here(cell)` are replaced by their 2^dim children (graduated
     * as long as the selected region is a half/quadrant-like block).
     *
     * IMPORTANT for MPI: the refinement direction matters. Going the other
     * way (coarsening fine cells with `indices >> 1`) is wrong in parallel:
     * the initial decomposition may split the 2^dim children of one parent
     * across two ranks, and both would emit the same coarse cell — a
     * duplicated cell in the global mesh. Refining a local parent creates
     * every child exactly once.
     */
    template <class Mesh, class Box, class Pred>
    Mesh make_locally_refined_mesh(const Box& box, std::size_t level, const Pred& refine_here)
    {
        constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t           = typename Mesh::mesh_id_t;
        using value_t             = typename Mesh::interval_t::value_t;

        auto coarse = samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(level).max_level(level));

        typename Mesh::cl_type cl;
        samurai::for_each_cell(coarse[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                                   if (!refine_here(cell))
                                   {
                                       cl[level][yz].add_point(cell.indices[0]);
                                       return;
                                   }
                                   const auto i = cell.indices[0];
                                   xt::xtensor_fixed<value_t, xt::xshape<dim - 1>> yz_child;
                                   for (unsigned m = 0; m < (1U << (dim - 1)); ++m)
                                   {
                                       for (std::size_t d = 0; d + 1 < dim; ++d)
                                       {
                                           yz_child(d) = 2 * yz(d) + static_cast<value_t>((m >> d) & 1U);
                                       }
                                       cl[level + 1][yz_child].add_interval({2 * i, 2 * i + 2});
                                   }
                               });
        return samurai::mra::make_mesh(cl, samurai::mesh_config<dim>().min_level(level).max_level(level + 1));
    }

    /**
     * Injective per-cell value: encodes (level, i, j, k) exactly in a double.
     * Used to verify that field values follow their cell through a migration:
     * after load balancing, every cell must still hold analytic(cell).
     * Exact for levels <= 11 and non-negative indices < 2^12 (our test boxes
     * are [0,1]^dim with level <= 6).
     */
    template <class Cell>
    double analytic(const Cell& cell)
    {
        constexpr double M = 4096.;
        auto value         = static_cast<double>(cell.level);
        for (std::size_t d = 0; d < Cell::dim; ++d)
        {
            value = value * M + static_cast<double>(cell.indices[d]);
        }
        return value;
    }

    /// Merge the cell arrays of all ranks into a single global CellArray on
    /// rank 0 (empty result on the other ranks).
    template <class CellArray_t>
    CellArray_t gather_global_cells(const CellArray_t& local_cells)
    {
        namespace mpi = boost::mpi;
        mpi::communicator world;

        std::vector<CellArray_t> all;
        mpi::gather(world, local_cells, all, 0);

        CellArray_t global;
        if (world.rank() == 0)
        {
            typename CellArray_t::cl_type cl;
            for (const auto& ca : all)
            {
                samurai::for_each_interval(ca,
                                           [&](std::size_t level, const auto& interval, const auto& index)
                                           {
                                               cl[level][index].add_interval(interval);
                                           });
            }
            global = {cl, false};
        }
        return global;
    }

    /// True on every rank when the two global cell sets are identical
    /// (level-by-level double inclusion, evaluated on rank 0 then broadcast).
    template <class CellArray_t>
    bool same_global_cells(const CellArray_t& before_local, const CellArray_t& after_local)
    {
        namespace mpi = boost::mpi;
        mpi::communicator world;

        auto before = gather_global_cells(before_local);
        auto after  = gather_global_cells(after_local);

        bool ok = true;
        if (world.rank() == 0)
        {
            if (before.nb_cells() != after.nb_cells())
            {
                ok = false;
            }
            else
            {
                const std::size_t min_level = std::min(before.min_level(), after.min_level());
                const std::size_t max_level = std::max(before.max_level(), after.max_level());
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!samurai::difference(before[level], after[level]).empty()
                        || !samurai::difference(after[level], before[level]).empty())
                    {
                        ok = false;
                        break;
                    }
                }
            }
        }
        mpi::broadcast(world, ok, 0);
        return ok;
    }

    /**
     * Two cells (level, indices) are face-adjacent if, normalized to a common
     * fine level, their boxes overlap in every dimension but one and just touch
     * in that one. Corner/edge-only contact does not count.
     */
    template <std::size_t dim>
    bool face_adjacent(const std::array<long, dim + 1>& a, const std::array<long, dim + 1>& b, std::size_t max_level)
    {
        int touching = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            const long sa = static_cast<long>(max_level) - a[0];
            const long sb = static_cast<long>(max_level) - b[0];
            const long a0 = a[d + 1] << sa, a1 = (a[d + 1] + 1) << sa;
            const long b0 = b[d + 1] << sb, b1 = (b[d + 1] + 1) << sb;
            const bool overlap = a0 < b1 && b0 < a1;
            if (overlap)
            {
                continue;
            }
            if (a1 == b0 || b1 == a0)
            {
                ++touching;
            }
            else
            {
                return false; // separated in this dimension
            }
        }
        return touching == 1;
    }

    /**
     * Number of face-connected components of the cells held locally by this rank
     * (flood fill on the pairwise adjacency graph). A correct diffusion partition
     * cedes only cells connected to the interface, so every rank must keep a
     * single connected component (no islands).
     */
    template <class Mesh>
    std::size_t local_connected_components(const Mesh& mesh)
    {
        using mesh_id_t           = typename Mesh::mesh_id_t;
        constexpr std::size_t dim = Mesh::dim;

        std::vector<std::array<long, dim + 1>> cells;
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   std::array<long, dim + 1> c{};
                                   c[0] = static_cast<long>(cell.level);
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       c[d + 1] = static_cast<long>(cell.indices[d]);
                                   }
                                   cells.push_back(c);
                               });

        const std::size_t n = cells.size();
        std::vector<char> seen(n, 0);
        std::size_t components = 0;
        for (std::size_t s = 0; s < n; ++s)
        {
            if (seen[s])
            {
                continue;
            }
            ++components;
            std::vector<std::size_t> stack{s};
            seen[s] = 1;
            while (!stack.empty())
            {
                const std::size_t u = stack.back();
                stack.pop_back();
                for (std::size_t v = 0; v < n; ++v)
                {
                    if (!seen[v] && face_adjacent<dim>(cells[u], cells[v], mesh.max_level()))
                    {
                        seen[v] = 1;
                        stack.push_back(v);
                    }
                }
            }
        }
        return components;
    }

    /**
     * The invariants of a correct migration (roadmap § 4.3):
     *  1. cells are conserved globally (none lost, none duplicated);
     *  2. every field value followed its cell (checked against analytic());
     *  3. the duplicate-free global cell set is unchanged.
     * `fields_match` is a callable(cell) -> bool provided by the test to check
     * its own fields.
     */
    template <class Mesh, class CellArray_t, class CheckFields>
    void check_lb_invariants(const Mesh& mesh_after,
                             const CellArray_t& cells_before,
                             std::size_t global_count_before,
                             const CheckFields& fields_match)
    {
        namespace mpi   = boost::mpi;
        using mesh_id_t = typename Mesh::mesh_id_t;
        mpi::communicator world;

        // 1. global cell count
        const std::size_t local_count  = mesh_after.nb_cells(mesh_id_t::cells);
        const std::size_t global_count = mpi::all_reduce(world, local_count, std::plus<std::size_t>());
        EXPECT_TRUE_ALL_RANKS(global_count == global_count_before);

        // 2. field values followed their cells
        bool values_ok = true;
        samurai::for_each_cell(mesh_after[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   values_ok = values_ok && fields_match(cell);
                               });
        EXPECT_TRUE_ALL_RANKS(values_ok);

        // 3. identical global cell set (catches duplicates and displacements)
        EXPECT_TRUE_ALL_RANKS(same_global_cells(cells_before, mesh_after[mesh_id_t::cells]));
    }
}
