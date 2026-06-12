// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Weighted space-filling curve partitioning (p4est-like).
 *
 * Algorithm:
 *  1. every cell gets the SFC key of its position normalized to the global
 *     max level (`indices << (max_level - level)`), after a global shift
 *     making all coordinates non-negative (curves require >= 0);
 *  2. local cells are sorted by key: the mesh becomes a set of P chunks of
 *     one global 1D sequence;
 *  3. the weighted cumulative sum along that sequence is computed with one
 *     MPI scan + one all_reduce (no gather of keys or cells: O(1)
 *     communication volume);
 *  4. the curve is cut into P segments of equal weight: cell with cumulative
 *     weight c goes to rank floor(c / (W_total / P)).
 *
 * Guarantees:
 *  - weighted balance up to the heaviest single cell per rank;
 *  - partitions are contiguous segments of the curve: compact for Hilbert
 *    (continuous curve), good for Morton (jumps at power-of-two boundaries);
 *  - deterministic: keys are unique (cells do not overlap), so the global
 *    order is total and two consecutive calls produce the same partition
 *    (second call migrates nothing).
 *
 * Note: on the *first* call after an arbitrary initial decomposition, the
 * local key chunks of different ranks may interleave; the assignment is
 * still exact because the cuts only depend on the global cumulative weight.
 *
 * Communication: 3 all_reduce (2 for the coordinate shift bound, 1 for the
 * total weight) + 1 scan. No data gather.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../../field.hpp"
#include "../sfc/hilbert.hpp"
#include "../sfc/morton.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>

namespace samurai::load_balancing
{
    template <class Curve>
    class SFC
    {
      public:

        SFC() = default;

        explicit SFC(Curve curve)
            : m_curve(std::move(curve))
        {
        }

        std::string name() const
        {
            return "sfc-" + m_curve.name();
        }

        /**
         * Destination rank of each cell by equal-weight cuts of the curve.
         * @note MPI: 3 all_reduce + 1 scan (collective), no data gather.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& weight) const
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            using cell_t              = typename Mesh::cell_t;
            constexpr std::size_t dim = Mesh::dim;

            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());

            // normalization level: the *config* max level, identical on every
            // rank (the local max level is not!)
            const std::size_t max_level = mesh.max_level();

            // -- 1a. global coordinate shift: curves need non-negative input --------
            std::array<std::int64_t, dim> local_min;
            local_min.fill(std::numeric_limits<std::int64_t>::max());
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              const auto shift = max_level - cell.level;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  local_min[d] = std::min(local_min[d], static_cast<std::int64_t>(cell.indices[d]) << shift);
                              }
                          });
            std::array<std::int64_t, dim> global_min;
            for (std::size_t d = 0; d < dim; ++d)
            {
                global_min[d] = boost::mpi::all_reduce(world, local_min[d], boost::mpi::minimum<std::int64_t>());
            }

            // -- 1b/2. keys + local sort --------------------------------------------
            struct Item
            {
                sfc_key_t key;
                double w;
                cell_t cell;
            };

            std::vector<Item> items;
            items.reserve(mesh.nb_cells(mesh_id_t::cells));
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              const auto shift = max_level - cell.level;
                              xt::xtensor_fixed<std::uint32_t, xt::xshape<dim>> p;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  const std::int64_t c = (static_cast<std::int64_t>(cell.indices[d]) << shift) - global_min[d];
                                  assert(c >= 0 && c < (std::int64_t(1) << Curve::max_bits(dim))
                                         && "normalized coordinate exceeds the curve range: max_level too deep for 64-bit keys");
                                  p(d) = static_cast<std::uint32_t>(c);
                              }
                              items.push_back({m_curve.template key<dim>(p), weight(cell), cell});
                          });

            std::sort(items.begin(),
                      items.end(),
                      [](const Item& a, const Item& b)
                      {
                          return a.key < b.key;
                      });

            // -- 3. weighted cumulative sum along the curve --------------------------
            double w_local = 0.;
            for (const auto& item : items)
            {
                w_local += item.w;
            }
            const double w_inclusive = boost::mpi::scan(world, w_local, std::plus<double>());
            const double w_offset    = w_inclusive - w_local;
            const double w_total     = boost::mpi::all_reduce(world, w_local, std::plus<double>());

            if (w_total <= 0.)
            {
                return flags; // nothing to balance on
            }

            // -- 4. equal-weight cuts: rank r owns cumulative [r, r+1) * W/P ---------
            const int last_rank = world.size() - 1;
            double cumulative   = w_offset;
            int rank            = 0;
            for (const auto& item : items)
            {
                while (rank < last_rank && cumulative >= (static_cast<double>(rank) + 1.) * w_total / world.size())
                {
                    ++rank;
                }
                flags[item.cell] = rank;
                cumulative += item.w;
            }
            return flags;
        }

      private:

        Curve m_curve;
    };
}
#endif
