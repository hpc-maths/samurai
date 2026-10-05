// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

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
     *
     * A strategy that can fail to shed the load it computed may also provide
     * `double last_unmet_flux() const`: `LoadBalancer::load_balance_with_stats()`
     * copies it into `LoadBalanceStats::unmet_flux`.
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

        /// Cheap by-products of one migration (already known from the routing
        /// collective): how many local cells left and arrived. Returned by
        /// `migrate()` so the diagnostic path can record them without the
        /// production path paying for a stats struct it does not use.
        struct MigrationCounts
        {
            std::size_t out = 0; ///< cells sent to other ranks
            std::size_t in  = 0; ///< cells received from other ranks
        };
    }

    /**
     * Load balancing driver: owns a strategy and moves the cells and the field
     * values to the ranks this strategy chooses.
     *
     * The strategy returns the destination rank of every local cell (see
     * `PartitionStrategy`). The driver then sends each destination rank a
     * single point-to-point message that holds both the cells and the values of
     * the given fields. A destination can be any rank, not only an MPI
     * neighbour. The migration runs these steps:
     *
     * 1. Sort the local cells: kept cells go to the new cell list, leaving
     *    cells to one cell list per destination rank.
     * 2. Run one `all_to_all`: each rank sends every other rank the number of
     *    cells it sends to it and its own total number of leaving cells. The
     *    totals tell every rank whether any cell moves at all, so all ranks
     *    take the same decision without another collective, which the
     *    collective construction of the new mesh requires.
     * 3. Send each destination its cells and, for each field, their values in
     *    `for_each_interval` order; receive one message from each announced
     *    source.
     * 4. Build the new mesh from the kept and received cells. Its constructor
     *    finds the MPI neighbourhood again.
     * 5. For each field, copy the values of the kept cells (the intersection of
     *    the old and new cells, level by level), insert the received values in
     *    the order of the sender, then swap the data arrays into the caller's
     *    field.
     * 6. Swap the new mesh into the caller's mesh object, so every field keeps
     *    pointing to it.
     *
     * When no cell moves on any rank, the migration stops after the
     * `all_to_all` and leaves the mesh and the fields untouched.
     *
     * Typical use:
     * @code
     * namespace lb = samurai::load_balancing;
     * auto balancer = lb::make_load_balancer<lb::Void>();
     * if (balancer.required(u.mesh(), lb::weight::uniform()))
     * {
     *     balancer.load_balance(lb::weight::uniform(), u, v);
     * }
     * @endcode
     *
     * `load_balance()` only partitions and migrates. When you need the quality
     * metrics (imbalance before and after, weighted loads, migrated counts) for
     * diagnostics, a benchmark or a test, call `load_balance_with_stats()`
     * instead: it does the same work and adds collective measurements around
     * it.
     */
    template <class Strategy>
    class LoadBalancer
    {
      public:

        /// Builds a driver from its configuration and its strategy.
        explicit LoadBalancer(LoadBalanceConfig config = {}, Strategy strategy = {})
            : m_config(config)
            , m_strategy(std::move(strategy))
        {
        }

        /**
         * Tells whether the global imbalance exceeds
         * `LoadBalanceConfig::imbalance_threshold`, with the same result on
         * every rank. Returns `false` with a single MPI process.
         *
         * @note MPI: collective on the world communicator.
         */
        template <class Mesh, class Weight>
        bool required(const Mesh& mesh, const Weight& weight) const
        {
            boost::mpi::communicator world;
            if (world.size() > 1)
            {
                return require_balance(mesh, weight, m_config.imbalance_threshold);
            }
            return false;
        }

        /**
         * Run one load balancing pass: partition with the strategy, then
         * migrate cells and all given fields, then swap the balanced mesh into
         * the caller's mesh object.
         *
         * No quality metric is computed, so the call adds no collective
         * beyond the migration itself. Use `load_balance_with_stats()` when you
         * need the metrics.
         *
         * All fields must live on the same mesh. With a single MPI process
         * the call does nothing.
         *
         * @note MPI: the migration runs one `all_to_all`, then point-to-point
         *       messages and the collectives of the mesh constructor, only
         *       when at least one cell moves on some rank.
         */
        template <class Weight, class Field, class... Fields>
            requires PartitionStrategy<Strategy, typename Field::mesh_t, Weight>
        void load_balance(const Weight& weight, Field& field, Fields&... other_fields)
        {
            assert(((&field.mesh() == &other_fields.mesh()) && ... && true) && "all fields must share the same mesh");
            run_load_balance(weight, nullptr, field, other_fields...);
        }

        /**
         * Same work as `load_balance()`, plus the quality measurements returned
         * in a `LoadBalanceStats`: local cell counts, weighted loads and global
         * imbalance before and after, migrated cell counts, unmet flux and
         * strategy name. Meant for diagnostics, benchmarks and tests.
         *
         * With a single MPI process nothing moves, and the returned
         * imbalances are 0.
         *
         * @note MPI: adds two `imbalance()` evaluations (collective) and two
         *       `local_load()` traversals around the migration.
         */
        template <class Weight, class Field, class... Fields>
            requires PartitionStrategy<Strategy, typename Field::mesh_t, Weight>
        LoadBalanceStats load_balance_with_stats(const Weight& weight, Field& field, Fields&... other_fields)
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

            stats.imbalance_before = imbalance(field.mesh(), weight);
            stats.imbalance_after  = stats.imbalance_before;

            run_load_balance(weight, &stats, field, other_fields...);

            stats.cells_after     = field.mesh().nb_cells(mesh_id_t::cells);
            stats.load_after      = local_load(field.mesh(), weight);
            stats.imbalance_after = imbalance(field.mesh(), weight);

            return stats;
        }

        /**
         * Debug helper: moves every cell and all given fields to `dest_rank`,
         * which gives the most unbalanced distribution. It tests the
         * strategies from an extreme initial state, with the same migration as
         * `load_balance()`. Does nothing with a single MPI process.
         *
         * @note MPI: same communication as the migration of `load_balance()`.
         */
        template <class Field, class... Fields>
        void concentrate_on(int dest_rank, Field& field, Fields&... other_fields)
        {
            boost::mpi::communicator world;
            if (world.size() > 1)
            {
                auto flags = make_scalar_field<int>("lb_flags", field.mesh());
                flags.fill(dest_rank);
                migrate(flags, field, other_fields...);
            }
        }

        /// Returns the configuration of the driver.
        const LoadBalanceConfig& config() const
        {
            return m_config;
        }

        /// Returns the strategy, for example to read its state after a call.
        Strategy& strategy()
        {
            return m_strategy;
        }

      private:

        /**
         * Shared core of `load_balance()` and `load_balance_with_stats()`:
         * partition, migrate, swap. When `stats` is non-null, the cheap
         * by-products of the work (timings, migrated counts, unmet flux,
         * strategy name) are recorded into it; the costly before/after metrics
         * are the caller's responsibility. No-op with a single MPI process.
         */
        template <class Weight, class Field, class... Fields>
        void run_load_balance(const Weight& weight, LoadBalanceStats* stats, Field& field, Fields&... other_fields)
        {
            samurai::ScopedTimer t("load_balancing");

            boost::mpi::communicator world;
            if (world.size() == 1)
            {
                return;
            }

            times::timers.start("partition");
            auto flags = m_strategy.partition(field.mesh(), weight);
            times::timers.stop("partition");

            times::timers.start("migration");
            const auto counts = migrate(flags, field, other_fields...);
            times::timers.stop("migration");

            if (stats != nullptr)
            {
                stats->strategy_name      = m_strategy.name();
                stats->cells_migrated_out = counts.out;
                stats->cells_migrated_in  = counts.in;
                // strategies that may fail to shed the requested load (e.g.
                // diffusion) expose the deficit; the others leave it at 0.
                if constexpr (requires { m_strategy.last_unmet_flux(); })
                {
                    stats->unmet_flux = m_strategy.last_unmet_flux();
                }
            }
        }

        /**
         * Migration of the cells and the field values (the steps listed in the
         * description of the class). `flags[cell]` must hold a valid rank for every
         * local cell. Returns how many local cells left and arrived.
         */
        template <class Flags, class Field, class... Fields>
        detail::MigrationCounts migrate(const Flags& flags, Field& field, Fields&... other_fields)
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

            detail::MigrationCounts counts;

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
                return counts; // perfect status quo everywhere: keep mesh and fields untouched
            }

            counts.out = static_cast<std::size_t>(total_out);

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
                counts.in += static_cast<std::size_t>(from_all[r][0]);
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

            return counts;
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

    /// Builds a `LoadBalancer`; the configuration and the strategy default to
    /// their default-constructed values.
    template <class Strategy>
    auto make_load_balancer(LoadBalanceConfig config = {}, Strategy strategy = {})
    {
        return LoadBalancer<Strategy>(config, std::move(strategy));
    }
}
#endif
