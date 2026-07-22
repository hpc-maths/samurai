// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <string>

#include <boost/mpi.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>

// Kept free of GoogleTest so it can be reused by non-test tooling (e.g. the
// demos/mpi/ghost_cases visualisation tool) without pulling gtest into a demo.

namespace samurai_test
{
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
}
