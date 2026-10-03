// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// update_ghost_mr: multiresolution ghost update.
//
// Public API (namespace samurai):
//   update_ghost_mr(field, others...)      time and run the MR ghost update
//   update_ghost_mr(tuple / Field_tuple)   tuple dispatch
//   update_ghost_mr_if_needed(...)         skip fields whose ghosts are current
//
// Implementation (namespace samurai::detail):
//   update_ghost_mr_aggregated(...)        the algorithm below
//   exchange_subdomains_merged(...)        one non-blocking, field-merged
//                                          subdomain exchange per neighbour
//   subdomain_exchange_intervals<bool>(...)  the cells sent to / received
//                                          from one neighbour
//
// This header is included last by update.hpp: the algorithm calls
// outer_subdomain_corner<...> (explicit template argument, so the template must
// be visible at definition), update_ghost_periodic and update_outer_ghosts. It
// pulls only those helper headers, so there is no mutual include with update.hpp.
//
// Algorithm. Per level (top-down then bottom-up), the subdomain ghosts of all
// fields are exchanged in a single non-blocking round per neighbour: fields are
// independent, so they are packed into one buffer per neighbour and the
// receives are posted before the sends; a neighbour with which there is nothing
// to exchange gets no message (see neighbour_exchange.hpp). Periodic ghosts
// keep their per-dimension ordering through update_ghost_periodic. What is
// deliberately NOT merged:
//   - across levels: the projection reference[L] -> proj_cells[L-1] reads ghosts
//     synchronised at level L (inter-level wavefront);
//   - subdomain vs periodic, and periodic dim vs dim: a periodic send reads
//     field(level, i - shift), which is not restricted to mesh.subdomain(level) and
//     may thus read a subdomain ghost just synchronised, and periodic dimensions
//     accumulate at corners - hence the subdomain -> periodic -> dim ordering.

#include <algorithm>
#include <iterator>
#include <vector>

#include "../algorithm.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../timers.hpp"

#include "neighbour_exchange.hpp" // exchange_intervals, neighbour_exchange
#include "update_outer_ghost.hpp" // update_outer_ghosts
#include "update_periodic.hpp"    // update_ghost_periodic
#include "update_subdomain.hpp"   // outer_subdomain_corner

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai::detail
{
#ifdef SAMURAI_WITH_MPI
    // The cells of the subdomain ghost exchange with one neighbour, in packing
    // order. to_send: the inner-interface cells this rank owns that the
    // neighbour holds, then the outer-subdomain corners this rank owns that the
    // neighbour holds. !to_send: the same two sets with the roles of the two
    // ranks swapped, i.e. what the neighbour sends to this rank. Evaluated once
    // per neighbour and shared by every field of the exchange.
    template <bool to_send, class Field>
    auto subdomain_exchange_intervals(std::size_t level, Field& field, const typename Field::mesh_t::mpi_subdomain_t& neighbour)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh();

        exchange_intervals<typename Field::mesh_t> intervals;
        auto push = [&](const auto& i, const auto& index)
        {
            intervals.push_back(i, index);
        };

        if constexpr (to_send)
        {
            intersection(mesh[mesh_id_t::reference][level], neighbour.mesh[mesh_id_t::reference][level], mesh.subdomain(level))(push);
        }
        else
        {
            intersection(neighbour.mesh[mesh_id_t::reference][level], mesh[mesh_id_t::reference][level], neighbour.mesh.subdomain(level))(
                push);
        }

        auto subdomain_corners = outer_subdomain_corner<to_send>(level, field, neighbour);
        for_each_interval(subdomain_corners,
                          [&](const auto, const auto& i, const auto& index)
                          {
                              push(i, index);
                          });
        return intervals;
    }
#endif // SAMURAI_WITH_MPI

    // Field-merged, non-blocking subdomain ghost exchange for
    // (level, field, other_fields...). One message per neighbour carries the
    // subdomain ghosts of every field, in pack order [field, other_fields...].
    // Result is identical to exchanging each field on its own.
    template <class Field, class... Fields>
    void
    exchange_subdomains_merged([[maybe_unused]] std::size_t level, [[maybe_unused]] Field& field, [[maybe_unused]] Fields&... other_fields)
    {
#ifdef SAMURAI_WITH_MPI
        using value_t   = typename Field::value_type;
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        static_assert((std::is_same_v<value_t, typename Fields::value_type> && ...),
                      "aggregated ghost update requires all fields to share the same value_type");

        auto& mesh                = field.mesh();
        const auto& neighbourhood = mesh.mpi_neighbourhood();
        if (neighbourhood.empty() || mesh[mesh_id_t::reference][level].empty())
        {
            return;
        }

        // send[k] here and recv[.] on neighbour k are the same cells, evaluated
        // from the two copies of the two meshes: both ranks agree on the size of
        // the message between them, and on whether it is empty.
        const std::size_t n                   = neighbourhood.size();
        constexpr std::size_t values_per_cell = (Field::n_comp + ... + Fields::n_comp);
        std::vector<exchange_intervals<mesh_t>> send(n);
        std::vector<exchange_intervals<mesh_t>> recv(n);
        std::vector<std::size_t> send_count(n);
        std::vector<std::size_t> recv_count(n);
        for (std::size_t k = 0; k < n; ++k)
        {
            if (neighbourhood[k].mesh[mesh_id_t::reference][level].empty())
            {
                continue;
            }
            send[k]       = subdomain_exchange_intervals<true>(level, field, neighbourhood[k]);
            recv[k]       = subdomain_exchange_intervals<false>(level, field, neighbourhood[k]);
            send_count[k] = send[k].n_cells() * values_per_cell;
            recv_count[k] = recv[k].n_cells() * values_per_cell;
        }

        neighbour_exchange<value_t>(
            neighbourhood,
            send_count,
            recv_count,
            [&](std::size_t k, auto& buf)
            {
                pack_intervals(level, field, send[k], buf);
                (pack_intervals(level, other_fields, send[k], buf), ...);
            },
            [&](std::size_t k, auto& it)
            {
                unpack_intervals(level, field, recv[k], it);
                (unpack_intervals(level, other_fields, recv[k], it), ...);
            });
#endif // SAMURAI_WITH_MPI
    }

    // Aggregated counterpart of update_ghost_mr. Same multiresolution
    // top-down / bottom-up structure; only the subdomain exchange is
    // field-merged and non-blocking. The periodic exchange keeps its
    // historic per-dimension ordering (see header note) by delegating to
    // update_ghost_periodic.
    template <class Field, class... Fields>
    void update_ghost_mr_aggregated(Field& field, Fields&... other_fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config_t::prediction_stencil_radius;

        auto& mesh            = field.mesh();
        auto max_level        = mesh.max_level();
        std::size_t min_level = 0;

        for (std::size_t level = max_level + 1; level-- > min_level;)
        {
            exchange_subdomains_merged(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);
            update_outer_ghosts(level, field, other_fields...);
            // Second subdomain sync, required for decomposition independence:
            // update_outer_ghosts recomputes the outer/B.C. ghosts locally, but
            // for non-stripe partitions the owner-computed outer values (see
            // outer_subdomain_corner) must still be redistributed to every rank
            // that references them.
            exchange_subdomains_merged(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);

            if (level > min_level)
            {
                auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
                set_at_levelm1.apply_op(variadic_projection(field, other_fields...));
            }
        }

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            auto expr        = intersection(pred_ghosts, mesh.subdomain(level), mesh[mesh_id_t::all_cells][level - 1]).on(level);

            expr.apply_op(variadic_prediction<pred_order, false>(field, other_fields...));
            exchange_subdomains_merged(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);
        }

        field.ghosts_updated() = true;
        ((other_fields.ghosts_updated() = true), ...);
    }
} // namespace samurai::detail

namespace samurai
{
    template <class Field>
    void update_ghost_mr_if_needed(Field& field)
    {
        if (!field.ghosts_updated())
        {
            update_ghost_mr(field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_mr_if_needed(Field& field, Fields&... other_fields)
    {
        update_ghost_mr_if_needed(field);
        update_ghost_mr_if_needed(other_fields...);
    }

    // Public entry point: times the update and delegates to the aggregated
    // implementation above.
    template <class Field, class... Fields>
    void update_ghost_mr(Field& field, Fields&... other_fields)
    {
        ScopedTimer timer_ghosts("ghost update");
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh();

        detail::update_ghost_mr_aggregated(field, other_fields...);

        timer_ghosts.set_cells(mesh.nb_cells(mesh_id_t::cells));
    }

    SAMURAI_INLINE void update_ghost_mr()
    {
    }

    template <class... T>
    SAMURAI_INLINE void update_ghost_mr(std::tuple<T...>& fields)
    {
        std::apply(
            [](T&... tupleArgs)
            {
                update_ghost_mr(tupleArgs...);
            },
            fields);
    }

    template <class... T>
    SAMURAI_INLINE void update_ghost_mr(Field_tuple<T...>& fields)
    {
        update_ghost_mr(fields.elements());
    }
}
