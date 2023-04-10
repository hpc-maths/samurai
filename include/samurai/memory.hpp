// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <numeric>

#include <fmt/format.h>

#include "level_cell_array.hpp"

namespace samurai
{
    // Compute the memory usage of samurai data structure for the mesh in bytes
    template <std::size_t Dim, class TInterval>
    std::size_t memory_usage(const LevelCellArray<Dim, TInterval>& lca)
    {
        std::size_t mem = lca.nb_intervals() * sizeof(TInterval);

        for (std::size_t d = 1; d < Dim; ++d)
        {
            mem += lca.offsets(d).size() * sizeof(std::size_t);
        }
        mem += sizeof(std::size_t);
        return mem;
    }

    template <std::size_t Dim, class TInterval, std::size_t max_size>
    std::size_t memory_usage(const CellArray<Dim, TInterval, max_size>& ca)
    {
        std::size_t mem = 0;
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            mem += memory_usage(ca[level]);
        }
        return mem;
    }

    template <class D, class Config>
    std::size_t memory_usage(const Mesh_base<D, Config>& mesh, bool verbose = false)
    {
        using mesh_id_t = typename Mesh_base<D, Config>::mesh_id_t;
        std::size_t mem = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(mesh_id_t::count); ++i)
        {
            auto id            = static_cast<mesh_id_t>(i);
            std::size_t mem_id = memory_usage(mesh[id]);
            if (verbose)
            {
                std::cout << fmt::format("Mesh {}: {}", id, mem_id) << std::endl;
            }
            mem += mem_id;
        }
        return mem;
    }
}
