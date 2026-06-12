// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <gtest/gtest.h>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
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
