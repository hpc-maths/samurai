// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Partition independence of make_graduation.
//
// The graduation sweep exchanges each level with the MPI neighbours so that a
// cascade crossing a subdomain boundary is captured: the graded mesh must not
// depend on the decomposition. Oracle: every rank builds the same ungraded
// global mesh, grades it alone (the sequential reference), then grades its own
// part with the neighbourhood of an MRMesh built on that part, as the
// adaptation does. The union of the graded parts must be the reference.

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace mpi = boost::mpi;

namespace
{
    class graduation_parallel : public samurai_test::MpiTest
    {
    };

    constexpr std::size_t dim       = 2;
    constexpr std::size_t top_level = 3; // 8 x 8 coarse cells, the unit of the partition
    constexpr std::size_t max_level = 7;
    constexpr int n_top             = 1 << top_level;

    using ca_t = samurai::CellArray<dim>;
    using cl_t = samurai::CellList<dim>;

    std::uint32_t hash(std::uint32_t a, std::uint32_t b, std::uint32_t c)
    {
        std::uint32_t h = a * 0x9E3779B1U ^ (b + 0x7F4A7C15U) * 0x85EBCA77U ^ (c + 0x165667B1U) * 0xC2B2AE3DU;
        h ^= h >> 15;
        h *= 0x2C1B3C6DU;
        h ^= h >> 12;
        return h;
    }

    // Leaves of a complete but ungraded quadtree: a cell is refined with
    // probability 1/3, so neighbouring leaves differ by several levels.
    void add_leaves(cl_t& cl, std::size_t level, int i, int j, const std::function<bool(std::size_t, int, int)>& keep)
    {
        if (level < max_level
            && hash(static_cast<std::uint32_t>(level), static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j)) % 3 == 0)
        {
            for (int dj = 0; dj < 2; ++dj)
            {
                for (int di = 0; di < 2; ++di)
                {
                    add_leaves(cl, level + 1, 2 * i + di, 2 * j + dj, keep);
                }
            }
            return;
        }
        if (keep(level, i, j))
        {
            cl[level][{j}].add_point(i);
        }
    }

    ca_t leaves(const std::function<bool(std::size_t, int, int)>& keep)
    {
        cl_t cl;
        for (int j = 0; j < n_top; ++j)
        {
            for (int i = 0; i < n_top; ++i)
            {
                add_leaves(cl, top_level, i, j, keep);
            }
        }
        return {cl, false};
    }

    enum class Decomp
    {
        x_strips,
        blocks,
        hashed,
        leaf_strips
    };

    // Owner of a leaf (level, i, j). The first three decompositions assign whole
    // level-top_level trees; leaf_strips cuts along x at positions that are not
    // aligned with any cell, so that sibling leaves can land on different ranks, as
    // partition_mesh does.
    int owner(Decomp decomp, std::size_t level, int leaf_i, int leaf_j, int size)
    {
        const auto shift = static_cast<int>(level - top_level);
        if (decomp == Decomp::leaf_strips)
        {
            const double x = (leaf_i + 0.5) / static_cast<double>(n_top << shift);
            return static_cast<int>(std::floor(x * size + 0.37)) % size;
        }
        const int i = leaf_i >> shift;
        const int j = leaf_j >> shift;
        switch (decomp)
        {
            case Decomp::x_strips:
                return i * size / n_top;
            case Decomp::blocks:
            {
                int npx = static_cast<int>(std::sqrt(static_cast<double>(size)));
                while (size % npx != 0)
                {
                    --npx;
                }
                const int npy = size / npx;
                return (i * npx / n_top) + npx * (j * npy / n_top);
            }
            case Decomp::hashed:
            default:
                return static_cast<int>(hash(99U, static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j))
                                        % static_cast<std::uint32_t>(size));
        }
    }

    std::string label(Decomp decomp)
    {
        switch (decomp)
        {
            case Decomp::x_strips:
                return "x_strips";
            case Decomp::blocks:
                return "blocks";
            case Decomp::hashed:
                return "hashed";
            case Decomp::leaf_strips:
            default:
                return "leaf_strips";
        }
    }

    bool same_cells(const ca_t& a, const ca_t& b)
    {
        if (a.nb_cells() != b.nb_cells())
        {
            return false;
        }
        for (std::size_t level = 0; level <= max_level; ++level)
        {
            if (!samurai::difference(a[level], b[level]).empty() || !samurai::difference(b[level], a[level]).empty())
            {
                return false;
            }
        }
        return true;
    }

    void expect_partition_independent(bool periodic, std::size_t grad_width, int radius, Decomp decomp)
    {
        mpi::communicator world;
        const int rank = world.rank();
        const int size = world.size();

        const std::array<bool, dim> is_periodic{periodic, periodic};

        auto local_cells = leaves(
            [&](std::size_t level, int i, int j)
            {
                return owner(decomp, level, i, j, size) == rank;
            });

        // The neighbourhood and the domain pyramid of the production path.
        auto config = samurai::mesh_config<dim>()
                          .min_level(top_level)
                          .max_level(max_level)
                          .max_stencil_radius(radius)
                          .graduation_width(grad_width)
                          .periodic(periodic)
                          .disable_args_parse();
        auto mesh = samurai::mra::make_mesh(local_cells, config);

        ca_t reference = leaves(
            [](std::size_t, int, int)
            {
                return true;
            });
        std::remove_cvref_t<decltype(mesh.mpi_neighbourhood())> no_neighbours;
        samurai::make_graduation(reference, mesh.domain_pyramid(), no_neighbours, is_periodic, grad_width, mesh.max_stencil_radius());

        samurai::make_graduation(local_cells,
                                 mesh.domain_pyramid(),
                                 mesh.mpi_neighbourhood(),
                                 is_periodic,
                                 grad_width,
                                 mesh.max_stencil_radius());

        auto graded = samurai_test::gather_global_cells(local_cells);
        bool ok     = (rank != 0) || same_cells(graded, reference);
        mpi::broadcast(world, ok, 0);

        EXPECT_TRUE(ok) << fmt::format("periodic={} grad_width={} radius={} decomposition={} on {} ranks",
                                       periodic,
                                       grad_width,
                                       radius,
                                       label(decomp),
                                       size);
    }

    TEST_F(graduation_parallel, partition_independent)
    {
        for (bool periodic : {false, true})
        {
            for (std::size_t grad_width : {std::size_t{1}, std::size_t{2}})
            {
                for (int radius : {1, 2, 3})
                {
                    for (Decomp decomp : {Decomp::x_strips, Decomp::blocks, Decomp::hashed, Decomp::leaf_strips})
                    {
                        expect_partition_independent(periodic, grad_width, radius, decomp);
                    }
                }
            }
        }
    }
}
