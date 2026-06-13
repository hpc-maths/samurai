// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Load balancing driver.
 *
 * Design (see docs/load_balancing_roadmap.md, step 1):
 *  - a *strategy* is an object exposing `partition(mesh, weight) -> flags`
 *    where `flags[cell]` is the destination rank of `cell`. It performs no
 *    migration itself (see the PartitionStrategy concept below);
 *  - the *driver* (`LoadBalancer<Strategy>`) owns the whole MPI machinery:
 *    a single fused migration moves the cells AND the field values in the
 *    same point-to-point message, routed by `flags` towards arbitrary ranks
 *    (destinations are NOT restricted to the geometric MPI neighbourhood).
 *
 * Migration scheme (normative):
 *  1. sort local cells: kept cells go to `new_cl`, leaving cells go to one
 *     CellList payload per destination rank;
 *  2. one single collective: all_to_all of pairs {cells for you, my total
 *     outgoing count}. The second member lets every rank decide *globally*
 *     whether any migration happens at all (mesh reconstruction is collective,
 *     so this decision must be identical everywhere) without an extra
 *     collective;
 *  3. isend one MigrationPayload (CellArray + one flat value vector per field,
 *     serialized in for_each_interval order) per destination; blocking recv
 *     from each announced source; wait_all;
 *  4. new mesh built from kept + received cells. The Mesh_base(cl, ref_mesh)
 *     constructor re-discovers the MPI neighbourhood from scratch
 *     (find_neighbourhood), nothing else to do here;
 *  5. per field: copy kept values (intersection old∩new, level by level),
 *     insert received values (same for_each_interval order as the sender),
 *     then swap the data arrays into the user's field;
 *  6. mesh.swap(new_mesh): the caller's mesh object now holds the balanced
 *     mesh and every field keeps pointing to it.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <concepts>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../algorithm.hpp"
#include "../algorithm/utils.hpp"
#include "../field.hpp"
#include "../mesh.hpp"
#include "../timers.hpp"

#include "config.hpp"
#include "metrics.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>

namespace samurai::load_balancing
{
    /**
     * Contract of a partitioning strategy.
     *
     * `partition(mesh, weight)` returns an int field on `mesh` holding the
     * destination rank of each cell; `name()` identifies the strategy in the
     * collected statistics. A strategy never communicates field data and never
     * modifies the mesh.
     */
    template <class Strategy, class Mesh, class Weight>
    concept PartitionStrategy = requires(Strategy s, Mesh& mesh, const Weight& w) {
        { s.partition(mesh, w) };
        { s.name() } -> std::convertible_to<std::string>;
    };

    namespace detail
    {
        /**
         * Content of one migration message: the cells leaving for one rank and,
         * for each migrated field, their values flattened in the order produced
         * by `for_each_interval(cells)` (components contiguous per interval).
         * Sender and receiver iterate the very same CellArray, so the order
         * matches by construction.
         */
        template <class CellArray_t, class... Fields>
        struct MigrationPayload
        {
            CellArray_t cells;
            std::tuple<std::vector<typename std::decay_t<Fields>::value_type>...> field_data;

            template <class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & cells;
                std::apply(
                    [&ar](auto&... v)
                    {
                        ((ar & v), ...);
                    },
                    field_data);
            }
        };
    }

    /**
     * Load balancing driver. Owns a strategy and performs the fused
     * cells+fields migration described in the header of this file.
     *
     * Typical use:
     * @code
     * namespace lb = samurai::load_balancing;
     * auto balancer = lb::make_load_balancer<lb::Void>();
     * if (balancer.required(u.mesh(), lb::weight::uniform()))
     * {
     *     auto stats = balancer.load_balance(lb::weight::uniform(), u, v);
     * }
     * @endcode
     */
    template <class Strategy>
    class LoadBalancer
    {
      public:

        explicit LoadBalancer(LoadBalanceConfig config = {}, Strategy strategy = {})
            : m_config(config)
            , m_strategy(std::move(strategy))
        {
        }

        /**
         * Collective decision: is the current imbalance above the configured
         * threshold? Returns the same value on every rank.
         *
         * @note MPI: collective on the world communicator.
         */
        template <class Mesh, class Weight>
        bool required(const Mesh& mesh, const Weight& weight) const
        {
            boost::mpi::communicator world;
            if (world.size() <= 1)
            {
                return false;
            }
            return require_balance(mesh, weight, m_config.imbalance_threshold);
        }

        /**
         * Run one load balancing pass: partition with the strategy, then
         * migrate cells and all given fields, then swap the balanced mesh into
         * the caller's mesh object.
         *
         * All fields must live on the same mesh. With a single MPI process
         * this is a silent no-op.
         *
         * @note MPI: the migration itself performs one all_to_all (routing
         *       discovery) + point-to-point payloads + the collectives of the
         *       mesh constructor (executed only when at least one cell
         *       migrates somewhere, decided globally). The statistics add two
         *       `imbalance()` evaluations (collective) around the migration.
         */
        template <class Weight, class Field, class... Fields>
            requires PartitionStrategy<Strategy, typename Field::mesh_t, Weight>
        LoadBalanceStats load_balance(const Weight& weight, Field& field, Fields&... other_fields)
        {
            using mesh_id_t = typename Field::mesh_t::mesh_id_t;

            assert(((&field.mesh() == &other_fields.mesh()) && ... && true) && "all fields must share the same mesh");

            LoadBalanceStats stats;
            stats.strategy_name = m_strategy.name();
            stats.cells_before  = field.mesh().nb_cells(mesh_id_t::cells);
            stats.cells_after   = stats.cells_before;
            stats.load_before   = local_load(field.mesh(), weight);
            stats.load_after    = stats.load_before;

            boost::mpi::communicator world;
            if (world.size() <= 1)
            {
                return stats;
            }

            times::timers.start("load_balancing");

            stats.imbalance_before = imbalance(field.mesh(), weight);
            stats.imbalance_after  = stats.imbalance_before;

            const auto t0 = std::chrono::steady_clock::now();
            times::timers.start("load_balancing:partition");
            auto flags = m_strategy.partition(field.mesh(), weight);
            times::timers.stop("load_balancing:partition");
            const auto t1 = std::chrono::steady_clock::now();

            // strategies that may fail to shed the requested load (e.g. diffusion)
            // expose the deficit; the others leave unmet_flux at 0.
            if constexpr (requires { m_strategy.last_unmet_flux(); })
            {
                stats.unmet_flux = m_strategy.last_unmet_flux();
            }

            times::timers.start("load_balancing:migration");
            migrate(flags, stats, field, other_fields...);
            times::timers.stop("load_balancing:migration");
            const auto t2 = std::chrono::steady_clock::now();

            stats.partition_time  = std::chrono::duration<double>(t1 - t0).count();
            stats.migration_time  = std::chrono::duration<double>(t2 - t1).count();
            stats.cells_after     = field.mesh().nb_cells(mesh_id_t::cells);
            stats.load_after      = local_load(field.mesh(), weight);
            stats.imbalance_after = imbalance(field.mesh(), weight);

            times::timers.stop("load_balancing");

            if (m_config.verbose)
            {
                std::clog << "[rank " << world.rank() << "] load_balance(" << stats.strategy_name << "): " << stats.cells_before << " -> "
                          << stats.cells_after << " cells (out " << stats.cells_migrated_out << ", in " << stats.cells_migrated_in
                          << "), imbalance " << stats.imbalance_before << " -> " << stats.imbalance_after << std::endl;
            }
            return stats;
        }

        const LoadBalanceConfig& config() const
        {
            return m_config;
        }

        Strategy& strategy()
        {
            return m_strategy;
        }

      private:

        /**
         * Fused cells+fields migration (steps 1-6 of the scheme documented in
         * the file header). `flags[cell]` must hold a valid rank for every
         * local cell.
         */
        template <class Flags, class Field, class... Fields>
        void migrate(const Flags& flags, LoadBalanceStats& stats, Field& field, Fields&... other_fields)
        {
            using Mesh_t    = typename Field::mesh_t;
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            using cl_type   = typename Mesh_t::cl_type;
            using ca_type   = typename Mesh_t::ca_type;
            using payload_t = detail::MigrationPayload<ca_type, Field, Fields...>;

            boost::mpi::communicator world;
            const auto size = static_cast<std::size_t>(world.size());
            const int rank  = world.rank();
            auto& mesh      = field.mesh();

            // -- 1. sort local cells by destination ---------------------------------
            cl_type new_cl;
            std::vector<cl_type> send_cl(size);
            std::vector<long long> send_count(size, 0);

            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              const int dest = flags[cell];
                              assert(dest >= 0 && dest < world.size() && "flags must hold a valid destination rank");
                              auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                              if (dest == rank)
                              {
                                  new_cl[cell.level][yz].add_point(cell.indices[0]);
                              }
                              else
                              {
                                  send_cl[static_cast<std::size_t>(dest)][cell.level][yz].add_point(cell.indices[0]);
                                  ++send_count[static_cast<std::size_t>(dest)];
                              }
                          });

            const long long total_out = std::accumulate(send_count.begin(), send_count.end(), 0LL);

            // -- 2. the single collective: routing discovery + global activity ------
            // To each rank r we send {number of cells for r, my total outgoing
            // count}. The second member exposes every rank's activity to
            // everyone, so the (collective) decision to rebuild the mesh needs
            // no further communication.
            std::vector<std::array<long long, 2>> to_all(size), from_all(size);
            for (std::size_t r = 0; r < size; ++r)
            {
                to_all[r] = {send_count[r], total_out};
            }
            boost::mpi::all_to_all(world, to_all, from_all);

            const bool any_migration = std::any_of(from_all.begin(),
                                                   from_all.end(),
                                                   [](const auto& p)
                                                   {
                                                       return p[1] > 0;
                                                   });
            if (!any_migration)
            {
                return; // perfect status quo everywhere: keep mesh and fields untouched
            }

            stats.cells_migrated_out = static_cast<std::size_t>(total_out);

            // -- 3. build and send payloads -----------------------------------------
            std::size_t n_dest = 0;
            for (std::size_t r = 0; r < size; ++r)
            {
                n_dest += (send_count[r] > 0) ? 1 : 0;
            }

            std::vector<payload_t> outbox;
            outbox.reserve(n_dest); // no reallocation: isend keeps references
            std::vector<boost::mpi::request> requests;
            requests.reserve(n_dest);

            for (std::size_t r = 0; r < size; ++r)
            {
                if (send_count[r] == 0)
                {
                    continue;
                }
                payload_t payload;
                payload.cells = {send_cl[r], false};
                pack_fields(payload, std::index_sequence_for<Field, Fields...>{}, field, other_fields...);
                outbox.push_back(std::move(payload));
                requests.push_back(world.isend(static_cast<int>(r), tag_migration, outbox.back()));
            }

            // -- 4. receive payloads and collect incoming cells ----------------------
            std::vector<payload_t> inbox;
            for (std::size_t r = 0; r < size; ++r)
            {
                if (r == static_cast<std::size_t>(rank) || from_all[r][0] == 0)
                {
                    continue;
                }
                payload_t payload;
                world.recv(static_cast<int>(r), tag_migration, payload);
                for_each_interval(payload.cells,
                                  [&](std::size_t level, const auto& interval, const auto& index)
                                  {
                                      new_cl[level][index].add_interval(interval);
                                  });
                stats.cells_migrated_in += static_cast<std::size_t>(from_all[r][0]);
                inbox.push_back(std::move(payload));
            }

            boost::mpi::wait_all(requests.begin(), requests.end());

            // -- 5. new mesh (re-discovers the MPI neighbourhood, see mesh.hpp) ------
            times::timers.start("load_balancing:rebuild");
            Mesh_t new_mesh(new_cl, mesh);
            times::timers.stop("load_balancing:rebuild");

            // -- 6. rebuild each field on the new mesh -------------------------------
            rebuild_fields(new_mesh, inbox, std::index_sequence_for<Field, Fields...>{}, field, other_fields...);

            mesh.swap(new_mesh);
        }

        /// Flatten the values of every field on the cells of `payload.cells`,
        /// in for_each_interval order (the receiver relies on this exact order).
        template <class Payload, std::size_t... Is, class... Fields>
        static void pack_fields(Payload& payload, std::index_sequence<Is...>, const Fields&... fields)
        {
            (pack_one_field(payload.cells, std::get<Is>(payload.field_data), fields), ...);
        }

        template <class CellArray_t, class Data, class Field>
        static void pack_one_field(const CellArray_t& cells, Data& data, const Field& field)
        {
            data.reserve(cells.nb_cells() * Field::n_comp);
            for_each_interval(cells,
                              [&](std::size_t level, const auto& interval, const auto& index)
                              {
                                  auto values = field(level, interval, index);
                                  std::copy(values.begin(), values.end(), std::back_inserter(data));
                              });
        }

        template <class Mesh_t, class Payloads, std::size_t... Is, class... Fields>
        static void rebuild_fields(Mesh_t& new_mesh, const Payloads& inbox, std::index_sequence<Is...>, Fields&... fields)
        {
            (rebuild_one_field<Is>(new_mesh, inbox, fields), ...);
        }

        /// New field on the new mesh = kept values (old∩new intersection) +
        /// received values (same traversal order as pack_one_field), then swap
        /// the data into the user's field object.
        template <std::size_t I, class Mesh_t, class Payloads, class Field>
        static void rebuild_one_field(Mesh_t& new_mesh, const Payloads& inbox, Field& field)
        {
            using mesh_id_t = typename Mesh_t::mesh_id_t;

            Field new_field(field.name(), new_mesh);
            new_field.fill(0);

            auto& old_mesh = field.mesh();
            for (std::size_t level = old_mesh.min_level(); level <= old_mesh.max_level(); ++level)
            {
                auto kept = intersection(old_mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                kept.apply_op(samurai::copy(new_field, field));
            }

            for (const auto& payload : inbox)
            {
                std::ptrdiff_t offset = 0;
                const auto& data      = std::get<I>(payload.field_data);
                for_each_interval(
                    payload.cells,
                    [&](std::size_t level, const auto& interval, const auto& index)
                    {
                        const auto count = static_cast<std::ptrdiff_t>(interval.size() * Field::n_comp);
                        std::copy(data.begin() + offset, data.begin() + offset + count, new_field(level, interval, index).begin());
                        offset += count;
                    });
            }

            swap(field, new_field); // swaps the data arrays only: `field` keeps
                                    // pointing to the caller's mesh object
        }

        LoadBalanceConfig m_config;
        Strategy m_strategy;
    };

    /// Convenience factory, mirroring the `make_*` idiom of samurai.
    template <class Strategy>
    auto make_load_balancer(LoadBalanceConfig config = {}, Strategy strategy = {})
    {
        return LoadBalancer<Strategy>(config, std::move(strategy));
    }
}
#endif
