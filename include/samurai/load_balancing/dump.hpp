// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Kept separate from load_balancer.hpp on purpose: this is the only part of
// the module that depends on the HDF5 I/O stack, and users who never dump
// partitions should not pay for that include.

#include <filesystem>
#include <string>

#include "../field.hpp"
#include "../io/hdf5.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>

namespace samurai::load_balancing
{
    /**
     * Write the current partition as an HDF5/XDMF file holding one scalar
     * field `"rank"` (owning rank of each cell), for visual comparison of the
     * partition shapes in ParaView.
     *
     * The MPI world size is appended to the file name, mirroring the samurai
     * I/O convention: `<filename>_size_<P>.h5`.
     *
     * @note MPI: collective (parallel HDF5 write); every rank must call it.
     */
    template <class Mesh>
    void dump_partition(const std::filesystem::path& path, const std::string& filename, Mesh& mesh)
    {
        boost::mpi::communicator world;
        auto rank_field = make_scalar_field<int>("rank", mesh);
        rank_field.fill(world.rank());
        save(path, fmt::format("{}_size_{}", filename, world.size()), mesh, rank_field);
    }
}
#endif
