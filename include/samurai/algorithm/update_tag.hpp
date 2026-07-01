// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>

#include <xtensor/containers/xfixed.hpp>

#include "../algorithm.hpp"
#include "../bc/apply_field_bc.hpp"
#include "../concepts.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../subset/node.hpp"
#include "../timers.hpp"
#include "graduation.hpp"
#include "utils.hpp"

#ifndef NDEBUG
#include "../io/hdf5.hpp"
#endif

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <xtensor/views/xmasked_view.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    template <class Field>
    void check_duplicate_cells([[maybe_unused]] Field& field)
    {
#ifdef SAMURAI_WITH_MPI
        // static constexpr std::size_t dim = Field::dim;
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::cells].min_level();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();
        mpi::communicator world;

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (world.rank() > neighbour.rank)
            {
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto out_interface = intersection(mesh[mesh_id_t::cells][level], neighbour.mesh[mesh_id_t::cells][level]);
                    out_interface(
                        [&](const auto& i, const auto& index)
                        {
                            // delete cell
                            std::cout << fmt::format("fall intersection between {} {} on level {} in {} {}",
                                                     world.rank(),
                                                     neighbour.rank,
                                                     level,
                                                     i,
                                                     index[0])
                                      << std::endl;
                        });
                }
            }
        }
#endif
    }

    template <class Field>
    void keep_only_one_coarse_tag([[maybe_unused]] Field& tag)
    {
#ifdef SAMURAI_WITH_MPI
        constexpr std::size_t dim = Field::dim;
        using mesh_t              = typename Field::mesh_t;
        using mesh_id_t           = typename mesh_t::mesh_id_t;
        std::vector<mpi::request> req;

        auto& mesh            = tag.mesh();
        std::size_t max_level = mesh[mesh_id_t::cells].max_level();
        mpi::communicator world;

        for (auto& neighbour : mesh.mpi_neighbourhood())
        {
            if (world.rank() > neighbour.rank)
            {
                auto min_level = std::max<std::size_t>(1, mesh[mesh_id_t::reference].min_level());

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto out_interface = intersection(mesh[mesh_id_t::cells][level], neighbour.mesh.subdomain()).on(level - 1);
                    out_interface(
                        [&](const auto& i, const auto& index)
                        {
                            if constexpr (dim == 1)
                            {
                                auto mask1 = (tag(level, 2 * i) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1) & static_cast<std::uint8_t>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1) & static_cast<std::uint8_t>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1), mask) = 0;
                            }
                            if constexpr (dim == 2)
                            {
                                auto j     = index[0];
                                auto mask1 = (tag(level, 2 * i, 2 * j) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<std::uint8_t>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i, 2 * j) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<std::uint8_t>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i, 2 * j), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) = 0;
                            }
                            if constexpr (dim == 3)
                            {
                                auto j     = index[0];
                                auto k     = index[1];
                                auto mask1 = (tag(level, 2 * i, 2 * j, 2 * k) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j, 2 * k) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1, 2 * k) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::coarsen))
                                           & (tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::coarsen));
                                auto mask2 = (tag(level, 2 * i, 2 * j, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep))
                                           | (tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<std::uint8_t>(CellFlag::keep));
                                auto mask = xt::eval(mask1 && !mask2);

                                xt::masked_view(tag(level, 2 * i, 2 * j, 2 * k), mask)             = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j, 2 * k), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1, 2 * k), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j, 2 * k + 1), mask)         = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = 0;
                                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = 0;
                            }
                        });
                }
            }
        }
#endif
    }
}
